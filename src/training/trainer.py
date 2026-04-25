from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model.builder import build_transformer
from src.registry import DATASET, LOSS, OPTIMIZER, SCHEDULER

from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from .logging import build_logger


def _strip_name(cfg) -> Dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    out.pop("name", None)
    return out


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_kwargs = _strip_name(cfg.data)
        data_kwargs["batch_size"] = cfg.training.batch_size
        self.data = DATASET.build(cfg.data.name, **data_kwargs)

        # Inject vocab sizes into the model cfg before build.
        OmegaConf.set_struct(cfg, False)
        cfg.model.src_vocab_size = self.data["src_vocab_size"]
        cfg.model.tgt_vocab_size = self.data["tgt_vocab_size"]
        OmegaConf.set_struct(cfg, True)

        self.model = build_transformer(cfg).to(self.device)

        opt_kwargs = _strip_name(cfg.optimizer)
        opt_kwargs.setdefault("lr", cfg.training.lr)
        opt_kwargs.setdefault("weight_decay", cfg.training.weight_decay)
        self.optimizers = OPTIMIZER.build(cfg.optimizer.name, model=self.model, **opt_kwargs)

        sched_kwargs = _strip_name(cfg.scheduler)
        steps_per_epoch = max(1, len(self.data["train_loader"]))
        sched_kwargs.setdefault("total_steps", steps_per_epoch * cfg.training.num_epochs)
        self.schedulers = SCHEDULER.build(cfg.scheduler.name, optimizers=self.optimizers, **sched_kwargs)

        loss_kwargs = _strip_name(cfg.loss)
        loss_kwargs.setdefault("ignore_index", self.data["pad_token_id"])
        self.loss_fn = LOSS.build(cfg.loss.name, **loss_kwargs)

        experiment_name = cfg.get("experiment_name") or "default"
        self.logger = build_logger(cfg, experiment_name)
        self.global_step = 0
        self.start_epoch = 0
        self._maybe_resume()

    def _maybe_resume(self) -> None:
        preload = self.cfg.training.get("preload")
        if not preload:
            return
        path = checkpoint_path(
            self.cfg.training.ckpt_dir,
            self.cfg.training.ckpt_basename,
            int(preload),
        )
        state = load_checkpoint(path, self.model, self.optimizers)
        self.start_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]

    def _step_optimizers(self) -> None:
        for opt in self.optimizers:
            opt.step()
        for sched in self.schedulers:
            if sched is not None:
                sched.step()
        for opt in self.optimizers:
            opt.zero_grad()

    def fit(self) -> None:
        cfg = self.cfg
        ckpt_dir = Path(cfg.training.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tgt_vocab = self.data["tgt_vocab_size"]
        grad_clip = cfg.training.get("grad_clip", 0.0)

        for epoch in range(self.start_epoch, cfg.training.num_epochs):
            self.model.train()
            iterator = tqdm(
                self.data["train_loader"],
                desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}",
                unit="batch",
            )
            for batch in iterator:
                src = batch["encoder_input"].to(self.device)
                tgt = batch["decoder_input"].to(self.device)
                src_mask = batch["encoder_mask"].to(self.device)
                tgt_mask = batch["decoder_mask"].to(self.device)
                label = batch["label"].to(self.device)

                for opt in self.optimizers:
                    opt.zero_grad()

                enc_out = self.model.encode(src, src_mask)
                dec_out = self.model.decode(enc_out, src_mask, tgt, tgt_mask)
                logits = self.model.project(dec_out)
                loss = self.loss_fn(logits.view(-1, tgt_vocab), label.view(-1))
                iterator.set_postfix(loss=f"{loss.item():6.3f}")
                self.logger.scalar("train/loss", loss.item(), self.global_step)
                self.logger.flush()

                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self._step_optimizers()
                self.global_step += 1

            save_checkpoint(
                checkpoint_path(cfg.training.ckpt_dir, cfg.training.ckpt_basename, epoch),
                epoch=epoch,
                model=self.model,
                optimizers=self.optimizers,
                global_step=self.global_step,
            )

        self.logger.close()
