from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich.live import Live
from tqdm import tqdm

from src.model.builder import build_transformer
from src.registry import DATASET, LOSS, OPTIMIZER, SCHEDULER

from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from .hf_push import push_checkpoint
from .logging import build_logger
from .tui import TrainingTUI


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
        self.experiment_name = experiment_name
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

    def _model_summary(self) -> Dict[str, Any]:
        cfg = self.cfg
        attn_name = cfg.attention.get("name", "?")
        return {
            "name": cfg.model.get("name", "transformer"),
            "n_params": sum(p.numel() for p in self.model.parameters()),
            "n_layers": cfg.model.n_layers,
            "d_model": cfg.model.d_model,
            "attention": attn_name,
            "feedforward": cfg.feedforward.get("name", "?"),
        }

    def fit(self) -> None:
        cfg = self.cfg
        ckpt_dir = Path(cfg.training.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tgt_vocab = self.data["tgt_vocab_size"]
        grad_clip = cfg.training.get("grad_clip", 0.0)
        steps_per_epoch = max(1, len(self.data["train_loader"]))
        num_epochs = cfg.training.num_epochs
        total_steps = steps_per_epoch * num_epochs
        use_tui = bool(cfg.training.get("tui", True))

        last_ckpt: Path | None = None
        if use_tui:
            tui = TrainingTUI(
                experiment_name=self.experiment_name,
                model_summary=self._model_summary(),
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                num_epochs=num_epochs,
                device=self.device,
            )
            tui.event(f"start: {num_epochs} epochs × {steps_per_epoch} steps", "bold green3")
            live_ctx = Live(tui.render(), console=tui.console, refresh_per_second=10, screen=False)
        else:
            tui = None
            live_ctx = nullcontext()

        with live_ctx as live:
            for epoch in range(self.start_epoch, num_epochs):
                self.model.train()
                if tui is not None:
                    tui.reset_epoch(epoch)

                if tui is None:
                    iterator = tqdm(
                        self.data["train_loader"],
                        desc=f"Epoch {epoch + 1}/{num_epochs}",
                        unit="batch",
                    )
                else:
                    iterator = self.data["train_loader"]

                for step_in_epoch, batch in enumerate(iterator):
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
                    loss_val = loss.item()
                    self.logger.scalar("train/loss", loss_val, self.global_step)
                    self.logger.flush()

                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self._step_optimizers()

                    if tui is not None:
                        lr = self.optimizers[0].param_groups[0]["lr"]
                        tui.update_step(
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            global_step=self.global_step,
                            loss=loss_val,
                            lr=lr,
                        )
                        live.update(tui.render())
                    else:
                        iterator.set_postfix(loss=f"{loss_val:6.3f}")
                    self.global_step += 1

                last_ckpt = checkpoint_path(
                    cfg.training.ckpt_dir, cfg.training.ckpt_basename, epoch
                )
                save_checkpoint(
                    last_ckpt,
                    epoch=epoch,
                    model=self.model,
                    optimizers=self.optimizers,
                    global_step=self.global_step,
                )
                if tui is not None:
                    tui.event(f"epoch {epoch + 1} done · saved {last_ckpt.name}", "green3")
                    live.update(tui.render())

            if tui is not None:
                tui.event("training complete", "bold green3")
                live.update(tui.render())

        self.logger.close()
        self._maybe_push_to_hub(last_ckpt, tui)

    def _maybe_push_to_hub(self, ckpt: Path | None, tui: TrainingTUI | None) -> None:
        hf_cfg = self.cfg.training.get("hf", None)
        if hf_cfg is None or not hf_cfg.get("push", False):
            return
        if ckpt is None or not Path(ckpt).exists():
            return

        run_dir = Path.cwd()
        cfg_snapshot = run_dir / ".hydra" / "config.yaml"

        try:
            url = push_checkpoint(
                ckpt_path=ckpt,
                repo_id=hf_cfg.get("repo_id", None),
                private=bool(hf_cfg.get("private", False)),
                commit_message=hf_cfg.get(
                    "commit_message", f"{self.experiment_name}: trained checkpoint"
                ),
                extra_files=[cfg_snapshot] if cfg_snapshot.exists() else [],
            )
            msg = f"pushed → {url}"
            if tui is not None:
                tui.event(msg, "bold magenta")
            else:
                print(msg)
        except Exception as e:  # pragma: no cover - network/auth
            msg = f"hf push failed: {e}"
            if tui is not None:
                tui.event(msg, "bold red")
            else:
                print(msg)
