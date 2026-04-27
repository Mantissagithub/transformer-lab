import logging
import warnings
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

import src  # noqa: F401  - populates registries

from src.training.hf_credentials import ensure_hf_credentials
from src.training.logging import ensure_logging_backend
from src.training.trainer import Trainer
from src.utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    ensure_hf_credentials(cfg)
    ensure_logging_backend(cfg)
    Trainer(cfg).fit()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    load_dotenv(PROJECT_ROOT / ".env")
    main()
