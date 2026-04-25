from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def push_checkpoint(
    *,
    ckpt_path: Path,
    repo_id: Optional[str],
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload trained checkpoint",
    extra_files: Optional[Iterable[Path]] = None,
) -> str:
    from huggingface_hub import HfApi, create_repo

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF push requested but HF_TOKEN is not set. "
            "Copy .env.example to .env and fill in HF_TOKEN, "
            "or generate one at https://huggingface.co/settings/tokens"
        )
    repo_id = repo_id or os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError(
            "HF push requested but no repo_id provided. Set training.hf.repo_id "
            "in the config or HF_REPO_ID in .env"
        )

    create_repo(repo_id, token=token, exist_ok=True, private=private)
    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=ckpt_path.name,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    for extra in extra_files or []:
        extra = Path(extra)
        if not extra.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(extra),
            path_in_repo=extra.name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
    return f"https://huggingface.co/{repo_id}"
