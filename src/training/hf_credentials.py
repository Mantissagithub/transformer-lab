from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig
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

    repo_id = hf_cfg.get("repo_id") or os.environ.get("HF_REPO_ID")
    if not repo_id:
        console.print("[yellow]No repo_id in cfg.training.hf or HF_REPO_ID env.[/]")
        repo_id = Prompt.ask("HF repo_id (e.g. username/multinews-modern)").strip()
        if not repo_id:
            raise RuntimeError("repo_id is required when training.hf.push=true")
        os.environ["HF_REPO_ID"] = repo_id
        if Confirm.ask("Save HF_REPO_ID to .env for next run?", default=True):
            _append_env(PROJECT_ROOT / ".env", "HF_REPO_ID", repo_id)

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
