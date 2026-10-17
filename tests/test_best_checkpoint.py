from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.training.checkpoint import best_checkpoint_path, load_checkpoint
from src.training.trainer import Trainer


class _Logger:
    def __init__(self) -> None:
        self.values = []

    def scalar(self, tag, value, step) -> None:
        self.values.append((tag, value, step))

    def flush(self) -> None:
        pass


class _FixedLogits(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(input_ids.shape[0], -1, -1)


def _trainer(tmp_path: Path) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = OmegaConf.create(
        {
            "training": {
                "ckpt_dir": str(tmp_path),
                "ckpt_basename": "model",
            }
        }
    )
    trainer.dist_env = SimpleNamespace(is_main=True)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters())]
    trainer.logger = _Logger()
    trainer.global_step = 10
    trainer.best_val_loss = float("inf")
    return trainer


def test_best_checkpoint_only_saves_improvements(tmp_path, monkeypatch) -> None:
    trainer = _trainer(tmp_path)
    losses = iter([2.0, 2.5, 1.5])
    monkeypatch.setattr(trainer, "_validation_loss", lambda: next(losses))

    first = trainer._validate_and_save_best(epoch=0, tui=None)
    worse = trainer._validate_and_save_best(epoch=0, tui=None)
    better = trainer._validate_and_save_best(epoch=1, tui=None)

    expected = best_checkpoint_path(str(tmp_path), "model")
    assert first == expected
    assert worse is None
    assert better == expected
    assert list(tmp_path.glob("*.pt")) == [expected]
    assert trainer.best_val_loss == 1.5

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters())
    state = load_checkpoint(expected, restored_model, [restored_optimizer])
    assert state["epoch"] == 1
    assert state["global_step"] == 10
    assert state["best_val_loss"] == 1.5
    assert trainer.logger.values == [
        ("val/loss", 2.0, 10),
        ("val/loss", 2.5, 10),
        ("val/loss", 1.5, 10),
    ]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
    ],
)
def test_validation_loss_is_token_weighted_and_restores_training_mode(device) -> None:
    logits = torch.tensor(
        [[[0.0, 3.0, 1.0], [2.0, 0.0, 1.0], [0.0, 1.0, 3.0]]]
    )
    labels = torch.tensor([[1, 0, 2]])
    trainer = Trainer.__new__(Trainer)
    trainer.kind = "causal_lm"
    trainer.device = torch.device(device)
    trainer.model = _FixedLogits(logits).to(trainer.device).train()
    trainer.data = {
        "val_loader": [{"input_ids": torch.ones_like(labels), "labels": labels}],
        "pad_token_id": 0,
    }
    trainer.autocast_dtype = None
    trainer.dist_env = SimpleNamespace(is_dist=False)

    actual = trainer._validation_loss()
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        labels.reshape(-1),
        ignore_index=0,
        reduction="sum",
    ).item() / 2

    assert actual == pytest.approx(expected, rel=1e-5)
    assert trainer.model.training


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_validation_preserves_float64_accumulation() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.kind = "causal_lm"
    trainer.device = torch.device("mps")
    trainer.model = torch.nn.Embedding(2, 3).to(trainer.device).train()
    with torch.no_grad():
        trainer.model.weight.copy_(torch.tensor([[0.0, 0.0, 1e8], [0.0, 0.0, 0.0]], device="mps"))
    trainer.data = {
        "val_loader": [
            {"input_ids": torch.tensor([[index]]), "labels": torch.tensor([[1]])}
            for index in (0, 1, 1)
        ],
        "pad_token_id": 0,
    }
    trainer.autocast_dtype = None
    trainer.dist_env = SimpleNamespace(is_dist=False)

    # MPS does not support float64:
    # https://docs.pytorch.org/docs/main/notes/mps.html#double-precision-types
    # Use Python double precision to compute the expected sum of MPS batch losses.
    batch_losses = []
    with torch.no_grad():
        for batch in trainer.data["val_loader"]:
            logits, labels = trainer._validation_logits(batch)
            batch_losses.append(torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3), labels.reshape(-1), reduction="sum"
            ).item())

    actual = trainer._validation_loss()
    assert actual == pytest.approx(sum(batch_losses) / 3, rel=0, abs=1e-6)
    assert trainer.model.training
