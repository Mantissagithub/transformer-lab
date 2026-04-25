import warnings
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

import src  # noqa: F401  - populates registries

from src.training.trainer import Trainer
from src.utils.seed import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    Trainer(cfg).fit()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    load_dotenv(PROJECT_ROOT / ".env")
    main()
