#!/usr/bin/env bash
# train every registered attention variant on meetingbank, routing each one
# to the topology it was designed for:
#
#   * cross-attn-capable variants -> encoder-decoder mode (existing residual
#     experiment config, default meetingbank dataset).
#   * self-attention-only variants (csa, hca, mla) -> causal-lm mode
#     (meeting_summarization_causal experiment config, meetingbank dataset
#     with mode=causal).
#
# per-variant attention overrides are inlined here because the small "test"
# yamls for csa / hca / mla cap max_seq_len at 32-64; meeting summarization
# packs ~768-token sequences, so those caps have to be lifted. variant configs
# live in configs/attention/.
#
# usage:
#   ./scripts/sweep_meeting_summarization.sh                 # all variants
#   ./scripts/sweep_meeting_summarization.sh mha gqa mla     # subset
#
# outputs land in outputs/<experiment_name>/<timestamp>/ and checkpoints in
# weights/<ckpt_basename>_epoch_*.pt with a per-variant suffix.

set -euo pipefail

ENC_DEC=(mha gqa gqa_rope mqa sliding_window sliding_gqa)
CAUSAL=(csa hca mla)
SEQ_LEN=768

# per-variant attention overrides for causal mode (must accommodate SEQ_LEN).
# these mirror the small csa/hca/mla.yaml shapes but with max_seq_len lifted.
declare -A CAUSAL_OVERRIDES
CAUSAL_OVERRIDES[csa]="attention.max_seq_len=${SEQ_LEN} attention.n_win=128 attention.k=64"
CAUSAL_OVERRIDES[hca]="attention.m=8"
CAUSAL_OVERRIDES[mla]="attention.max_seq_len=${SEQ_LEN}"

requested=("$@")
if [ ${#requested[@]} -eq 0 ]; then
  requested=("${ENC_DEC[@]}" "${CAUSAL[@]}")
fi

contains() { local x="$1"; shift; for i in "$@"; do [ "$i" = "$x" ] && return 0; done; return 1; }

for ATTN in "${requested[@]}"; do
  if contains "$ATTN" "${ENC_DEC[@]}"; then
    EXPERIMENT=meeting_summarization_residual
    EXTRA=""
    NAME_PREFIX=meeting_residual
  elif contains "$ATTN" "${CAUSAL[@]}"; then
    EXPERIMENT=meeting_summarization_causal
    EXTRA="${CAUSAL_OVERRIDES[$ATTN]:-}"
    NAME_PREFIX=meeting_causal
  else
    echo "unknown attention variant: $ATTN" >&2
    exit 1
  fi

  echo "==> training $ATTN  (experiment=$EXPERIMENT)"
  # shellcheck disable=SC2086
  python -m src.cli.train \
    experiment="$EXPERIMENT" \
    attention="$ATTN" \
    experiment_name="${NAME_PREFIX}_${ATTN}" \
    training.ckpt_basename="${NAME_PREFIX}_model_${ATTN}" \
    $EXTRA
done
