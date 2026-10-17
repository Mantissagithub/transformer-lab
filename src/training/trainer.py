from contextlib import nullcontext
import math
from pathlib import Path
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model.builder import build_causal_lm, build_transformer
from src.registry import DATASET, LOSS, OPTIMIZER, SCHEDULER

from .checkpoint import best_checkpoint_path, checkpoint_path, load_checkpoint, save_checkpoint
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
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
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
        accum = max(1, int(cfg.training.get("gradient_accumulation_steps", 1)))
        if max_steps and max_steps > 0:
            total_steps = int(max_steps)
        elif len_loader is not None:
            total_steps = (len_loader // accum) * cfg.training.num_epochs
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
        self.best_val_loss = math.inf
        self._maybe_resume()

    def _maybe_resume(self) -> None:
        preload = self.cfg.training.get("preload")
        if not preload:
            return
        if str(preload) == "best":
            path = best_checkpoint_path(
                self.cfg.training.ckpt_dir,
                self.cfg.training.ckpt_basename,
            )
        else:
            path = checkpoint_path(
                self.cfg.training.ckpt_dir,
                self.cfg.training.ckpt_basename,
                int(preload),
            )
        state = load_checkpoint(path, _maybe_unwrap(self.model), self.optimizers)
        self.start_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        saved_best = state.get("best_val_loss")
        if saved_best is not None:
            self.best_val_loss = float(saved_best)

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

    def _validation_logits(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "causal_lm":
            ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            return self.model(ids), labels

        src = batch["encoder_input"].to(self.device)
        tgt = batch["decoder_input"].to(self.device)
        src_mask = batch["encoder_mask"].to(self.device)
        tgt_mask = batch["decoder_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        model = _maybe_unwrap(self.model)
        enc_out = model.encode(src, src_mask)
        dec_out = model.decode(enc_out, src_mask, tgt, tgt_mask)
        return model.project(dec_out), labels

    def _validation_loss(self) -> float:
        val_loader = self.data.get("val_loader")
        if val_loader is None:
            raise RuntimeError("validation loader is not configured")
        was_training = self.model.training
        self.model.eval()
        pad_id = int(self.data["pad_token_id"])
        totals_device = torch.device("cpu") if self.device.type == "mps" else self.device
        totals = torch.zeros(2, device=totals_device, dtype=torch.float64)

        with torch.inference_mode():
            for batch in val_loader:
                with self._autocast():
                    logits, labels = self._validation_logits(batch)
                flat_labels = labels.reshape(-1)
                totals[0] += F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    flat_labels,
                    ignore_index=pad_id,
                    reduction="sum",
                ).to(totals.device).double()
                totals[1] += (flat_labels != pad_id).sum().to(totals.device)

        if self.dist_env.is_dist:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if was_training:
            self.model.train()
        if totals[1].item() == 0:
            raise RuntimeError("validation loader produced no scored tokens")
        return float((totals[0] / totals[1]).item())

    def _validate_and_save_best(
        self,
        *,
        epoch: int,
        tui: TrainingTUI | None,
    ) -> Path | None:
        val_loss = self._validation_loss()
        if not self.dist_env.is_main:
            return None

        self.logger.scalar("val/loss", val_loss, self.global_step)
        self.logger.flush()
        improved = val_loss < self.best_val_loss
        message = f"validation loss {val_loss:.4f}"
        if not improved:
            message += f" · best {self.best_val_loss:.4f}"
            if tui is not None:
                tui.event(message)
            else:
                print(message)
            return None

        self.best_val_loss = val_loss
        path = best_checkpoint_path(
            self.cfg.training.ckpt_dir,
            self.cfg.training.ckpt_basename,
        )
        save_checkpoint(
            path,
            epoch=epoch,
            model=_maybe_unwrap(self.model),
            optimizers=self.optimizers,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
        )
        message += f" · saved {path.name}"
        if tui is not None:
            tui.event(message)
        else:
            print(message)
        return path

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
        has_validation = self.data.get("val_loader") is not None

        best_ckpt: Path | None = None
        if use_tui:
            tui = TrainingTUI(
                experiment_name=self.experiment_name,
                model_summary=self._model_summary(),
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                num_epochs=num_epochs,
                device=self.device,
            )
            live_ctx = tui
        else:
            tui = None
            live_ctx = nullcontext()

        stop = False
        with live_ctx:
            if tui is not None:
                tui.event(f"start: {num_epochs} epochs × {steps_per_epoch} steps")
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
                _step_t = time.monotonic()
                _tok_accum = 0
                _toks_per_sec = 0.0
                _grad_norm = 0.0
                checked_midpoint = False
                midpoint = (
                    len_loader // 2
                    if has_validation and len_loader is not None and len_loader > 1
                    else None
                )
                for step_in_epoch, batch in enumerate(iterator):
                    if self.kind == "causal_lm":
                        _tok_accum += batch["input_ids"].numel()
                    else:
                        _tok_accum += batch["encoder_input"].numel()
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
                        if self.scaler is not None:
                            for opt in self.optimizers:
                                self.scaler.unscale_(opt)
                        # always measure gnorm for telemetry; only actually clip
                        # when grad_clip > 0 (max_norm=inf computes but no-ops).
                        max_norm = grad_clip if (grad_clip and grad_clip > 0) else float("inf")
                        _grad_norm = nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm
                        ).item()
                        self._step_optimizers()
                        _now = time.monotonic()
                        _toks_per_sec = _tok_accum / max(_now - _step_t, 1e-9)
                        _tok_accum = 0
                        _step_t = _now
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
                                grad_norm=_grad_norm,
                                tokens_per_sec=_toks_per_sec,
                            )
                        elif hasattr(iterator, "set_postfix"):
                            iterator.set_postfix(loss=f"{loss_val:6.3f}")

                        if (
                            midpoint is not None
                            and not checked_midpoint
                            and step_in_epoch + 1 >= midpoint
                        ):
                            saved = self._validate_and_save_best(epoch=epoch, tui=tui)
                            if saved is not None:
                                best_ckpt = saved
                            checked_midpoint = True

                        if max_steps and self.global_step >= max_steps:
                            stop = True
                            break

                if has_validation:
                    saved = self._validate_and_save_best(epoch=epoch, tui=tui)
                    if saved is not None:
                        best_ckpt = saved
                elif self.dist_env.is_main:
                    save_every_n = cfg.training.get("save_every_n_epochs", 4)
                    is_last_epoch = epoch == num_epochs - 1 or stop
                    if is_last_epoch or (save_every_n > 0 and epoch % save_every_n == 0):
                        best_ckpt = checkpoint_path(
                            cfg.training.ckpt_dir,
                            cfg.training.ckpt_basename,
                            epoch,
                        )
                        save_checkpoint(
                            best_ckpt,
                            epoch=epoch,
                            model=_maybe_unwrap(self.model),
                            optimizers=self.optimizers,
                            global_step=self.global_step,
                        )

            if tui is not None:
                tui.event("training complete")

        if self.dist_env.is_main:
            self.logger.close()
            self._maybe_push_to_hub(best_ckpt, tui)

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
                tui.event(msg)
            else:
                print(msg)
        except Exception as e:  # pragma: no cover - network/auth
            msg = f"hf push failed: {e}"
            if tui is not None:
                tui.event(msg)
            else:
                print(msg)


class _NullLogger:
    def scalar(self, *_a, **_k):
        pass

    def flush(self):
        pass

    def close(self):
        pass
