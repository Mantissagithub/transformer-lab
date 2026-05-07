from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

from src.registry import DATASET
from src.utils.masks import causal_mask

from .base import field_iter, get_or_build_tokenizer


class MeetingSummarizationDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, seq_len_src: int, seq_len_tgt: int) -> None:
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt
        self.src_sos = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.src_eos = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.src_pad = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
        self.tgt_sos = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.tgt_eos = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.tgt_pad = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        src_text = item["transcript"]
        tgt_text = item["summary"]
        enc_tokens = self.tokenizer_src.encode(src_text).ids[: self.seq_len_src - 2]
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids[: self.seq_len_tgt - 2]
        enc_pad = self.seq_len_src - len(enc_tokens) - 2
        dec_pad = self.seq_len_tgt - len(dec_tokens) - 1

        encoder_input = torch.cat([
            self.src_sos,
            torch.tensor(enc_tokens, dtype=torch.int64),
            self.src_eos,
            self.src_pad.repeat(enc_pad),
        ])
        decoder_input = torch.cat([
            self.tgt_sos,
            torch.tensor(dec_tokens, dtype=torch.int64),
            self.tgt_pad.repeat(dec_pad),
        ])
        label = torch.cat([
            torch.tensor(dec_tokens, dtype=torch.int64),
            self.tgt_eos,
            self.tgt_pad.repeat(dec_pad),
        ])

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.tgt_pad).unsqueeze(0).int().unsqueeze(0)
            & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class MeetingCausalDataset(Dataset):
    """decoder-only / instruction-style packing for meeting summarization.

    each example is laid out as ``[BOS] transcript [SEP] summary [EOS]``,
    truncated/padded to ``seq_len`` (= max_src_len + max_tgt_len). the
    transcript prefix is masked out of the loss by overwriting those label
    positions with ``pad_id`` — cross-entropy uses ``ignore_index=pad_id``,
    so loss is only scored on the summary continuation. used by the self-
    attention-only variants (csa, hca, mla) which can't drive an encoder-
    decoder topology.
    """

    def __init__(self, dataset, tokenizer, seq_len_src: int, seq_len_tgt: int) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len_src = seq_len_src
        self.seq_len_tgt = seq_len_tgt
        self.seq_len = seq_len_src + seq_len_tgt
        self.bos_id = tokenizer.token_to_id("[SOS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        if self.sep_id is None:
            # older meetingbank tokenizers (built before [SEP] was in the
            # special-tokens list) won't have a [SEP] id; fall back to [EOS]
            # as the boundary marker.
            self.sep_id = self.eos_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        # reserve room for [BOS], [SEP], [EOS], plus the summary itself.
        src_room = self.seq_len_src - 2  # [BOS] ... [SEP]
        tgt_room = self.seq_len_tgt - 1  # ... [EOS]
        src_ids = self.tokenizer.encode(item["transcript"]).ids[:src_room]
        tgt_ids = self.tokenizer.encode(item["summary"]).ids[:tgt_room]

        prefix = [self.bos_id, *src_ids, self.sep_id]
        completion = [*tgt_ids, self.eos_id]
        full = prefix + completion
        # pad to seq_len + 1 so we can split into input_ids / labels by shifting.
        pad_room = (self.seq_len + 1) - len(full)
        if pad_room > 0:
            full = full + [self.pad_id] * pad_room
        else:
            full = full[: self.seq_len + 1]

        full_t = torch.tensor(full, dtype=torch.int64)
        input_ids = full_t[:-1].clone()
        labels = full_t[1:].clone()

        # mask the prefix region in labels: cross-entropy ignores pad_id, so
        # loss is computed only over the summary continuation. the label at
        # position i predicts input_ids[i+1], so the prefix-mask runs up to
        # len(prefix) - 1 (inclusive).
        prefix_label_end = min(len(prefix) - 1, labels.shape[0])
        labels[:prefix_label_end] = self.pad_id

        return {"input_ids": input_ids, "labels": labels}


def _maybe_limit(ds, limit: int):
    if limit is None or limit <= 0:
        return ds
    return ds.select(range(min(limit, len(ds))))


@DATASET.register("meetingbank")
def build_meetingbank(
    name: str = "meetingbank",
    hf_path: str = "huuuyeah/meetingbank",
    tokenizer_dir: str = "tokenizers",
    tokenizer_basename: str = "meetingbank",
    vocab_size: int = 32000,
    max_src_len: int = 512,
    max_tgt_len: int = 256,
    batch_size: int = 8,
    val_batch_size: int = 1,
    num_workers: int = 0,
    limit: int = 0,
    mode: str = "encoder_decoder",
    **_: object,
) -> dict:
    if mode not in ("encoder_decoder", "causal"):
        raise ValueError(f"meetingbank: unknown mode {mode!r}")

    raw = load_dataset(hf_path)
    train_raw = _maybe_limit(raw["train"], limit)
    val_raw = _maybe_limit(raw["validation"], limit)

    tokenizer_dir = Path(tokenizer_dir)
    src_tok = get_or_build_tokenizer(
        tokenizer_dir / f"{tokenizer_basename}_transcript.json",
        field_iter(train_raw, "transcript"),
        vocab_size,
    )
    if mode == "encoder_decoder":
        tgt_tok = get_or_build_tokenizer(
            tokenizer_dir / f"{tokenizer_basename}_summary.json",
            field_iter(train_raw, "summary"),
            vocab_size,
        )
        train_ds = MeetingSummarizationDataset(train_raw, src_tok, tgt_tok, max_src_len, max_tgt_len)
        val_ds = MeetingSummarizationDataset(val_raw, src_tok, tgt_tok, max_src_len, max_tgt_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "src_tokenizer": src_tok,
            "tgt_tokenizer": tgt_tok,
            "src_vocab_size": src_tok.get_vocab_size(),
            "tgt_vocab_size": tgt_tok.get_vocab_size(),
            "pad_token_id": src_tok.token_to_id("[PAD]"),
        }

    # causal mode: a single tokenizer over the concatenated transcript+summary
    # corpus so the model sees a unified vocabulary.
    def _both_fields(rows):
        for row in rows:
            yield row["transcript"]
            yield row["summary"]

    tok = get_or_build_tokenizer(
        tokenizer_dir / f"{tokenizer_basename}_causal.json",
        _both_fields(train_raw),
        vocab_size,
    )
    train_ds = MeetingCausalDataset(train_raw, tok, max_src_len, max_tgt_len)
    val_ds = MeetingCausalDataset(val_raw, tok, max_src_len, max_tgt_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "src_tokenizer": tok,
        "tgt_tokenizer": tok,
        "src_vocab_size": tok.get_vocab_size(),
        "tgt_vocab_size": tok.get_vocab_size(),
        "pad_token_id": tok.token_to_id("[PAD]"),
        "eos_token_id": tok.token_to_id("[EOS]"),
    }
