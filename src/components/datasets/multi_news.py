from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

from src.registry import DATASET
from src.utils.masks import causal_mask

from .base import get_or_build_bpe_tokenizer, multi_field_iter


class MultiNewsDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len_src: int, seq_len_tgt: int) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt
        self.sos = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        src_text = item["document"].replace("|||||", "\n\n")
        tgt_text = item["summary"]
        enc_tokens = self.tokenizer.encode(src_text).ids[: self.seq_len_src - 2]
        dec_tokens = self.tokenizer.encode(tgt_text).ids[: self.seq_len_tgt - 2]
        enc_pad = self.seq_len_src - len(enc_tokens) - 2
        dec_pad = self.seq_len_tgt - len(dec_tokens) - 1

        encoder_input = torch.cat([
            self.sos,
            torch.tensor(enc_tokens, dtype=torch.int64),
            self.eos,
            self.pad.repeat(enc_pad),
        ])
        decoder_input = torch.cat([
            self.sos,
            torch.tensor(dec_tokens, dtype=torch.int64),
            self.pad.repeat(dec_pad),
        ])
        label = torch.cat([
            torch.tensor(dec_tokens, dtype=torch.int64),
            self.eos,
            self.pad.repeat(dec_pad),
        ])

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad).unsqueeze(0).int().unsqueeze(0)
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def _maybe_limit(ds, limit: int):
    if limit is None or limit <= 0:
        return ds
    return ds.select(range(min(limit, len(ds))))


@DATASET.register("multi_news")
def build_multi_news(
    name: str = "multi_news",
    hf_path: str = "alexfabbri/multi_news",
    tokenizer_dir: str = "tokenizers",
    tokenizer_basename: str = "multi_news_bpe",
    vocab_size: int = 32000,
    max_src_len: int = 2048,
    max_tgt_len: int = 512,
    batch_size: int = 8,
    val_batch_size: int = 1,
    num_workers: int = 0,
    limit: int = 0,
    **_: object,
) -> dict:
    raw = load_dataset(hf_path)
    train_raw = _maybe_limit(raw["train"], limit)
    val_raw = _maybe_limit(raw["validation"], limit)

    tokenizer_path = Path(tokenizer_dir) / f"{tokenizer_basename}.json"
    tokenizer = get_or_build_bpe_tokenizer(
        tokenizer_path,
        multi_field_iter(train_raw, ("document", "summary")),
        vocab_size,
    )

    train_ds = MultiNewsDataset(train_raw, tokenizer, max_src_len, max_tgt_len)
    val_ds = MultiNewsDataset(val_raw, tokenizer, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    vocab = tokenizer.get_vocab_size()
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "src_tokenizer": tokenizer,
        "tgt_tokenizer": tokenizer,
        "src_vocab_size": vocab,
        "tgt_vocab_size": vocab,
        "pad_token_id": tokenizer.token_to_id("[PAD]"),
    }
