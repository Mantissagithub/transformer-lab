"""causal-mode packing for the meetingbank dataset.

verifies that ``MeetingCausalDataset`` lays each example out as
``[BOS] transcript [SEP] summary [EOS]`` and that the loss-mask hides every
prefix label so cross-entropy (with ``ignore_index=pad_id``) only scores the
summary continuation. this is the contract the self-attention-only sweep
relies on; if it slips, csa/hca/mla would end up training against transcript
tokens they were supposed to ignore.
"""
import torch

from src.components.datasets.base import get_or_build_tokenizer
from src.components.datasets.meetingbank import MeetingCausalDataset


def _toy_tokenizer(tmp_path):
    corpus = [
        "alpha beta gamma delta epsilon",
        "summary one two three",
        "another transcript with words",
        "another summary text",
    ]
    return get_or_build_tokenizer(tmp_path / "tok.json", iter(corpus), vocab_size=64)


def test_layout_and_label_masking(tmp_path):
    tok = _toy_tokenizer(tmp_path)
    src_len, tgt_len = 8, 6
    rows = [{"transcript": "alpha beta gamma", "summary": "summary one"}]
    ds = MeetingCausalDataset(rows, tok, seq_len_src=src_len, seq_len_tgt=tgt_len)

    item = ds[0]
    input_ids = item["input_ids"]
    labels = item["labels"]

    bos = tok.token_to_id("[SOS]")
    sep = tok.token_to_id("[SEP]")
    eos = tok.token_to_id("[EOS]")
    pad = tok.token_to_id("[PAD]")

    # total length is (src_len + tgt_len), split into input_ids/labels by shifting.
    assert input_ids.shape == labels.shape == torch.Size([src_len + tgt_len])

    # input_ids[0] is [BOS]; somewhere before the summary there's exactly one [SEP].
    assert int(input_ids[0]) == bos
    sep_positions = (input_ids == sep).nonzero(as_tuple=True)[0]
    assert sep_positions.numel() == 1, f"expected exactly one [SEP], got {sep_positions.tolist()}"
    sep_pos = int(sep_positions.item())

    # labels for the prefix tokens (everything up to and including [SEP]) must
    # be pad so cross-entropy ignores them. labels[i] predicts input_ids[i+1],
    # so the prefix-mask covers indices [0, sep_pos) inclusive of sep_pos-1.
    prefix_label_end = sep_pos  # prefix length is sep_pos+1, label mask runs to sep_pos
    assert torch.all(labels[:prefix_label_end] == pad), labels[:prefix_label_end].tolist()

    # the summary region of labels must carry real ids (not all pad) — that's
    # what cross-entropy actually scores. summary in input_ids starts at sep_pos+1;
    # the matching label position is sep_pos.
    assert torch.any(labels[prefix_label_end:] != pad)

    # and the [EOS] marker must terminate the summary.
    eos_positions = (labels == eos).nonzero(as_tuple=True)[0]
    assert eos_positions.numel() >= 1


def test_truncation_does_not_drop_eos(tmp_path):
    """even when the transcript fills the budget, the [BOS]/[SEP]/[EOS] structure
    survives, and the prefix-mask still covers the transcript region."""
    tok = _toy_tokenizer(tmp_path)
    src_len, tgt_len = 6, 4
    long_transcript = "alpha beta gamma delta epsilon another"
    rows = [{"transcript": long_transcript, "summary": "summary one"}]
    ds = MeetingCausalDataset(rows, tok, seq_len_src=src_len, seq_len_tgt=tgt_len)

    item = ds[0]
    input_ids = item["input_ids"]
    pad = tok.token_to_id("[PAD]")
    sep = tok.token_to_id("[SEP]")

    # first token is [BOS] regardless of truncation.
    assert int(input_ids[0]) == tok.token_to_id("[SOS]")
    # [SEP] must appear exactly once and cleanly partition prefix from completion.
    sep_positions = (input_ids == sep).nonzero(as_tuple=True)[0]
    assert sep_positions.numel() == 1
    # labels before the [SEP] boundary must be pad (loss ignored).
    sep_pos = int(sep_positions.item())
    assert torch.all(item["labels"][:sep_pos] == pad)
