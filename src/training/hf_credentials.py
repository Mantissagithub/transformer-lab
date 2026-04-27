from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.prompt import Confirm, Prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_hf_credentials(cfg: DictConfig) -> None:
    if not cfg.training.get("hf", {}).get("push", False):
        return

    console = Console()
    hf_cfg = cfg.training.get("hf", {})

    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[yellow]HF push enabled but HF_TOKEN not found in env or .env.[/]")
        token = Prompt.ask("Enter HF write token", password=True).strip()
        if not token:
            raise RuntimeError("HF_TOKEN is required when training.hf.push=true")
        os.environ["HF_TOKEN"] = token
        if Confirm.ask("Save HF_TOKEN to .env for next run?", default=True):
            _append_env(PROJECT_ROOT / ".env", "HF_TOKEN", token)

    repo_id = hf_cfg.get("repo_id")
    if not repo_id:
        username = os.environ.get("HF_USERNAME")
        if not username:
            console.print("[yellow]HF_USERNAME not found in env or .env.[/]")
            username = Prompt.ask("Enter HF username").strip()
            if not username:
                raise RuntimeError("HF_USERNAME is required when training.hf.push=true")
            os.environ["HF_USERNAME"] = username
            if Confirm.ask("Save HF_USERNAME to .env for next run?", default=True):
                _append_env(PROJECT_ROOT / ".env", "HF_USERNAME", username)

        from huggingface_hub import HfApi
        api = HfApi(token=token)
        base = f"{cfg.experiment_name}-{_config_suffix(cfg)}"
        candidate = f"{username}/{base}"
        if not cfg.training.get("preload") and _repo_exists(api, candidate):
            n = 2
            while _repo_exists(api, f"{username}/{base}-{n}"):
                n += 1
            base = f"{base}-{n}"
            candidate = f"{username}/{base}"
            console.print(f"[yellow]Repo exists; using {base} instead.[/]")
        OmegaConf.set_struct(cfg, False)
        cfg.experiment_name = base
        cfg.training.ckpt_basename = base
        OmegaConf.set_struct(cfg, True)

        repo_id = candidate
        OmegaConf.set_struct(cfg, False)
        cfg.training.hf.repo_id = repo_id
        OmegaConf.set_struct(cfg, True)

    from huggingface_hub import HfApi

    try:
        user = HfApi(token=token).whoami()["name"]
    except Exception as exc:
        raise RuntimeError(f"HF token validation failed: {exc}") from exc
    console.print(f"[green]✓ HF authenticated as {user} → {repo_id}[/]")


def _append_env(path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if path.exists():
        lines = [
            line
            for line in path.read_text().splitlines()
            if not line.startswith(f"{key}=")
        ]
    lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def _repo_exists(api, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id, repo_type="model")
        return True
    except Exception:
        return False


def _config_suffix(cfg: DictConfig) -> str:
    t = cfg.training
    bs = int(t.batch_size)
    accum = int(t.get("gradient_accumulation_steps", 1))
    parts = [str(cfg.data.get("name", "")), f"bs{bs * accum}"]
    max_steps = int(t.get("max_steps", 0))
    if max_steps > 0:
        parts.append(f"s{max_steps}")
    else:
        parts.append(f"e{int(t.get('num_epochs', 1))}")
    parts.append(str(t.get("precision", "fp32")))
    return "-".join(p for p in parts if p)
