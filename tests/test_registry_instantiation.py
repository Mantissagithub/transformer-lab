"""Every registered component must instantiate from its YAML defaults without error."""
from pathlib import Path

import pytest
import yaml

from src.registry import (
    ATTENTION,
    CONNECTION,
    FFN,
    LOSS,
    NORM,
    POS,
    SCHEDULER,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
D_MODEL = 64


def _load(group: str, name: str) -> dict:
    cfg = yaml.safe_load((CONFIGS / group / f"{name}.yaml").read_text())
    cfg.pop("name", None)
    return cfg


@pytest.mark.parametrize("name", ATTENTION.names())
def test_attention(name: str) -> None:
    cfg = _load("attention", name)
    cfg.setdefault("d_model", D_MODEL)
    cfg.setdefault("dropout", 0.0)
    ATTENTION.build(name, **cfg)


@pytest.mark.parametrize("name", FFN.names())
def test_ffn(name: str) -> None:
    cfg = _load("feedforward", name)
    cfg.setdefault("d_model", D_MODEL)
    cfg.setdefault("dropout", 0.0)
    FFN.build(name, **cfg)


@pytest.mark.parametrize("name", NORM.names())
def test_norm(name: str) -> None:
    cfg = _load("normalization", name)
    NORM.build(name, d_model=D_MODEL, **cfg)


@pytest.mark.parametrize("name", POS.names())
def test_positional(name: str) -> None:
    cfg = _load("positional", name)
    cfg.setdefault("d_model", D_MODEL)
    cfg.setdefault("max_len", 32)
    if name == "alibi":
        cfg.setdefault("n_heads", 4)
    POS.build(name, **cfg)


@pytest.mark.parametrize("name", CONNECTION.names())
def test_connection(name: str) -> None:
    cfg = _load("connection", name)
    cfg.setdefault("d_model", D_MODEL)
    CONNECTION.build(name, **cfg)


@pytest.mark.parametrize("name", LOSS.names())
def test_loss(name: str) -> None:
    cfg = _load("loss", name)
    LOSS.build(name, **cfg)


@pytest.mark.parametrize("name", SCHEDULER.names())
def test_scheduler_yaml_loads(name: str) -> None:
    _load("scheduler", name)
