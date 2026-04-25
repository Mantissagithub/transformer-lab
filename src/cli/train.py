import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

import src  # noqa: F401  - populates registries

from src.training.trainer import Trainer
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    Trainer(cfg).fit()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
