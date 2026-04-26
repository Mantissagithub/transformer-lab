from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich.live import Live
from tqdm import tqdm

from src.model.builder import build_causal_lm, build_transformer
from src.registry import DATASET, LOSS, OPTIMIZER, SCHEDULER

from .checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from .distributed import DistEnv, init_distributed, wrap_model
from .hf_push import push_checkpoint
from .logging import build_logger
from .tui import TrainingTUI


def _strip_name(cfg) -> Dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    out.pop("name", None)
    return out


def _autocast_dtype(name: str | None) -> torch.dtype | None:
    if name in (None, "fp32"):
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unknown precision: {name}")


def _try_len(loader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _maybe_unwrap(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.kind = cfg.model.get("kind", "encoder_decoder")

        self.dist_env: DistEnv = init_distributed(
            force_disable=not cfg.training.get("distributed", {}).get("enabled", False)
        )
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.dist_env.local_rank}")
        else:
            self.device = torch.device("cpu")

        data_kwargs = _strip_name(cfg.data)
        data_kwargs["batch_size"] = cfg.training.batch_size
        if self.kind == "causal_lm":
            data_kwargs.setdefault("rank", self.dist_env.rank)
            data_kwargs.setdefault("world_size", self.dist_env.world_size)
        self.data = DATASET.build(cfg.data.name, **data_kwargs)

        OmegaConf.set_struct(cfg, False)
        cfg.model.src_vocab_size = self.data["src_vocab_size"]
        cfg.model.tgt_vocab_size = self.data["tgt_vocab_size"]
        OmegaConf.set_struct(cfg, True)

        if self.kind == "causal_lm":
            base_model = build_causal_lm(cfg)
        else:
            base_model = build_transformer(cfg)
        base_model = base_model.to(self.device)

        dist_cfg = cfg.training.get("distributed", {})
        self.model = wrap_model(
            base_model,
            self.dist_env,
            strategy=dist_cfg.get("strategy", "ddp"),
            fsdp_sharding=dist_cfg.get("fsdp", {}).get("sharding_strategy", "full_shard"),
            fsdp_mixed_precision=dist_cfg.get("fsdp", {}).get("mixed_precision", None),
        )

        compile_cfg = cfg.training.get("compile", False)
        if compile_cfg:
            opts = {} if compile_cfg is True else dict(compile_cfg)
            self.model = torch.compile(self.model, **opts)

        opt_kwargs = _strip_name(cfg.optimizer)
        opt_kwargs.setdefault("lr", cfg.training.lr)
        opt_kwargs.setdefault("weight_decay", cfg.training.weight_decay)
        self.optimizers = OPTIMIZER.build(cfg.optimizer.name, model=self.model, **opt_kwargs)

        sched_kwargs = _strip_name(cfg.scheduler)
        len_loader = _try_len(self.data["train_loader"])
        max_steps = cfg.training.get("max_steps", 0)
        if max_steps and max_steps > 0:
            total_steps = int(max_steps)
        elif len_loader is not None:
            total_steps = len_loader * cfg.training.num_epochs
        else:
            total_steps = 100_000
        sched_kwargs.setdefault("total_steps", total_steps)
        self.schedulers = SCHEDULER.build(cfg.scheduler.name, optimizers=self.optimizers, **sched_kwargs)
        self.total_steps = total_steps

        loss_kwargs = _strip_name(cfg.loss)
        loss_kwargs.setdefault("ignore_index", self.data["pad_token_id"])
        self.loss_fn = LOSS.build(cfg.loss.name, **loss_kwargs)

        precision = cfg.training.get("precision", "fp32")
        self.autocast_dtype = _autocast_dtype(precision)
        self.use_scaler = precision == "fp16" and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        experiment_name = cfg.get("experiment_name") or "default"
        self.experiment_name = experiment_name
        if self.dist_env.is_main:
            self.logger = build_logger(cfg, experiment_name)
        else:
            self.logger = _NullLogger()
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
        state = load_checkpoint(path, _maybe_unwrap(self.model), self.optimizers)
        self.start_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]

    def _step_optimizers(self) -> None:
        if self.scaler is not None:
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            for opt in self.optimizers:
                opt.step()
        for sched in self.schedulers:
            if sched is not None:
                sched.step()
        for opt in self.optimizers:
            opt.zero_grad()

    def _autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def _model_summary(self) -> Dict[str, Any]:
        cfg = self.cfg
        attn_name = cfg.attention.get("name", "?")
        return {
            "name": cfg.model.get("name", self.kind),
            "n_params": sum(p.numel() for p in self.model.parameters()),
            "n_layers": cfg.model.n_layers,
            "d_model": cfg.model.d_model,
            "attention": attn_name,
            "feedforward": cfg.feedforward.get("name", "?"),
        }

    def _encoder_decoder_step(self, batch, vocab: int) -> torch.Tensor:
        src = batch["encoder_input"].to(self.device)
        tgt = batch["decoder_input"].to(self.device)
        src_mask = batch["encoder_mask"].to(self.device)
        tgt_mask = batch["decoder_mask"].to(self.device)
        label = batch["label"].to(self.device)
        m = _maybe_unwrap(self.model)
        enc_out = m.encode(src, src_mask)
        dec_out = m.decode(enc_out, src_mask, tgt, tgt_mask)
        logits = m.project(dec_out)
        return self.loss_fn(logits.view(-1, vocab), label.view(-1))

    def _causal_step(self, batch, vocab: int) -> torch.Tensor:
        ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        logits = self.model(ids)
        return self.loss_fn(logits.view(-1, vocab), labels.view(-1))

    def fit(self) -> None:
        cfg = self.cfg
        ckpt_dir = Path(cfg.training.ckpt_dir)
        if self.dist_env.is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        vocab = self.data["tgt_vocab_size"]
        grad_clip = cfg.training.get("grad_clip", 0.0)
        accum = max(1, int(cfg.training.get("gradient_accumulation_steps", 1)))
        max_steps = int(cfg.training.get("max_steps", 0))
        num_epochs = cfg.training.num_epochs
        len_loader = _try_len(self.data["train_loader"])
        steps_per_epoch = len_loader if len_loader is not None else max(1, max_steps)
        total_steps = self.total_steps
        use_tui = bool(cfg.training.get("tui", True)) and self.dist_env.is_main

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

        stop = False
        with live_ctx as live:
            for epoch in range(self.start_epoch, num_epochs):
                if stop:
                    break
                self.model.train()
                if tui is not None:
                    tui.reset_epoch(epoch)

                if tui is None and self.dist_env.is_main and len_loader is not None:
                    iterator = tqdm(
                        self.data["train_loader"],
                        desc=f"Epoch {epoch + 1}/{num_epochs}",
                        unit="batch",
                    )
                else:
                    iterator = self.data["train_loader"]

                for opt in self.optimizers:
                    opt.zero_grad()
                micro = 0
                for step_in_epoch, batch in enumerate(iterator):
                    with self._autocast():
                        if self.kind == "causal_lm":
                            loss = self._causal_step(batch, vocab)
                        else:
                            loss = self._encoder_decoder_step(batch, vocab)
                    loss_val = loss.item()
                    loss = loss / accum

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    micro += 1

                    if micro >= accum:
                        if grad_clip and grad_clip > 0:
                            if self.scaler is not None:
                                for opt in self.optimizers:
                                    self.scaler.unscale_(opt)
                            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        self._step_optimizers()
                        micro = 0
                        self.global_step += 1

                        if self.dist_env.is_main:
                            self.logger.scalar("train/loss", loss_val, self.global_step)
                            self.logger.flush()
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
                        elif hasattr(iterator, "set_postfix"):
                            iterator.set_postfix(loss=f"{loss_val:6.3f}")

                        if max_steps and self.global_step >= max_steps:
                            stop = True
                            break

                if self.dist_env.is_main:
                    last_ckpt = checkpoint_path(
                        cfg.training.ckpt_dir, cfg.training.ckpt_basename, epoch
                    )
                    save_checkpoint(
                        last_ckpt,
                        epoch=epoch,
                        model=_maybe_unwrap(self.model),
                        optimizers=self.optimizers,
                        global_step=self.global_step,
                    )
                    if tui is not None:
                        tui.event(f"epoch {epoch + 1} done · saved {last_ckpt.name}", "green3")
                        live.update(tui.render())

            if tui is not None:
                tui.event("training complete", "bold green3")
                live.update(tui.render())

        if self.dist_env.is_main:
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


class _NullLogger:
    def scalar(self, *_a, **_k):
        pass

    def flush(self):
        pass

    def close(self):
        pass
