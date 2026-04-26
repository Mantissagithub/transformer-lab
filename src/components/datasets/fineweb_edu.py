from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from datasets import load_dataset

from src.registry import DATASET

from .base import get_or_build_bpe_tokenizer


def _stream(hf_path: str, hf_config: str | None, split: str):
    if hf_config:
        return load_dataset(hf_path, hf_config, split=split, streaming=True)
    return load_dataset(hf_path, split=split, streaming=True)


def _text_iter(stream, limit: int) -> Iterator[str]:
    for ex in islice(stream, limit):
        yield ex["text"]


class _PackedCausalLM(IterableDataset):
    """Packs a streaming text dataset into fixed (seq_len+1) chunks for next-token training."""

    def __init__(
        self,
        hf_path: str,
        hf_config: str | None,
        split: str,
        tokenizer,
        seq_len: int,
        eos_id: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.hf_path = hf_path
        self.hf_config = hf_config
        self.split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        global_id = self.rank * num_workers + worker_id
        global_size = self.world_size * num_workers

        stream = _stream(self.hf_path, self.hf_config, self.split)
        buf: list[int] = []
        chunk = self.seq_len + 1
        for idx, ex in enumerate(stream):
            if idx % global_size != global_id:
                continue
            ids = self.tokenizer.encode(ex["text"]).ids
            buf.extend(ids)
            buf.append(self.eos_id)
            while len(buf) >= chunk:
                window = buf[:chunk]
                buf = buf[chunk:]
                t = torch.tensor(window, dtype=torch.int64)
                yield {"input_ids": t[:-1], "labels": t[1:]}


@DATASET.register("fineweb_edu")
def build_fineweb_edu(
    name: str = "fineweb_edu",
    hf_path: str = "HuggingFaceFW/fineweb-edu",
    hf_config: str | None = "sample-10BT",
    train_split: str = "train",
    val_split: str | None = None,
    tokenizer_dir: str = "tokenizers",
    tokenizer_basename: str = "fineweb_edu_bpe",
    vocab_size: int = 32000,
    seq_len: int = 2048,
    batch_size: int = 8,
    val_batch_size: int = 1,
    num_workers: int = 0,
    tokenizer_train_samples: int = 200_000,
    val_samples: int = 256,
    rank: int = 0,
    world_size: int = 1,
    **_: object,
) -> dict:
    tokenizer_path = Path(tokenizer_dir) / f"{tokenizer_basename}.json"
    if not tokenizer_path.exists():
        train_stream = _stream(hf_path, hf_config, train_split)
        tokenizer = get_or_build_bpe_tokenizer(
            tokenizer_path,
            _text_iter(train_stream, tokenizer_train_samples),
            vocab_size,
        )
    else:
        tokenizer = get_or_build_bpe_tokenizer(tokenizer_path, iter(()), vocab_size)

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    train_ds = _PackedCausalLM(
        hf_path=hf_path,
        hf_config=hf_config,
        split=train_split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        eos_id=eos_id,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)

    if val_split is None:
        val_loader: Iterable | None = None
    else:
        val_ds = _PackedCausalLM(
            hf_path=hf_path,
            hf_config=hf_config,
            split=val_split,
            tokenizer=tokenizer,
            seq_len=seq_len,
            eos_id=eos_id,
            rank=rank,
            world_size=world_size,
        )
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=0)

    vocab = tokenizer.get_vocab_size()
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "src_tokenizer": tokenizer,
        "tgt_tokenizer": tokenizer,
        "src_vocab_size": vocab,
        "tgt_vocab_size": vocab,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
    }
