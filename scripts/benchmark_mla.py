"""Compare absorption off/on for one MLA checkpoint on identical synthetic tokens.

Measures batch-one prefill and cached decode separately, without dataset evaluation.
Decode consumes a fixed continuation, not free-running generated text.
"""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

import benchmark_hf_collection as benchmark


def measure(model, tokens, prompt_length, device):
    benchmark._sync(device)
    start = time.perf_counter()
    logits, cache = model(tokens[:, :prompt_length], return_kvs=True)
    benchmark._sync(device)
    prefill_ms = (time.perf_counter() - start) * 1000
    outputs = [logits[:, -1]]
    start = time.perf_counter()
    for token in tokens[:, prompt_length:].split(1, dim=1):
        logits, cache = model(token, past_kvs=cache, return_kvs=True)
        outputs.append(logits[:, -1])
    benchmark._sync(device)
    decode_ms = (time.perf_counter() - start) * 1000
    return prefill_ms, decode_ms, torch.cat(outputs).float().cpu()


def plot(result, path):
    plt = benchmark._setup_mpl()
    if plt is None:
        raise RuntimeError("Plotting requires the benchmark extra (matplotlib).")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.04))
    for ax, title, metric, unit in [
        (axes[0], "Prefill", "median_prefill_ms", "ms / prompt"),
        (axes[1], "Decode", "median_decode_ms_per_token", "ms / token"),
    ]:
        bars = [(mode, data[metric], benchmark.VARIANT_COLORS["mla"]
                 if mode == "mla" else "#DD8452") for mode, data in result["modes"].items()]
        benchmark._panel_hbar(ax, bars, f"{unit} (lower is better)", lower_better=True)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/mla/standard_vs_absorbed.json")
    args = parser.parse_args()
    if min(args.prompt_length, args.decode_steps, args.repeats) < 1:
        parser.error("prompt length, decode steps and repeats must be positive")
    device = benchmark._resolve_device(args.device)
    token = benchmark._hf_token(Path.cwd(), required=False)
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = benchmark._download_artifacts(benchmark.HfApi(token=token), args.repo_id, Path(tmp), token)
        cfg = benchmark.OmegaConf.load(artifacts.config_path)
        if cfg.attention.name != "mla" or cfg.model.kind != "causal_lm":
            parser.error("requires a causal MLA checkpoint")
        model = benchmark.build_causal_lm(cfg).eval().to(device)
        state = torch.load(artifacts.checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
    if args.prompt_length + args.decode_steps > min(layer.attn.max_seq_len for layer in model.layers):
        parser.error("prompt length + decode steps exceeds the checkpoint's maximum sequence length")
    tokens = torch.randint(int(cfg.model.tgt_vocab_size), (1, args.prompt_length + args.decode_steps),
                           generator=torch.Generator().manual_seed(42)).to(device)

    def run(enabled):
        for layer in model.layers:
            layer.attn.weight_absorption = enabled
        return measure(model, tokens, args.prompt_length, device)

    for _ in range(2):
        for enabled in (False, True):
            run(enabled)
    trials = {False: [], True: []}
    max_error, close, same_top1 = 0.0, True, True
    for repeat in range(args.repeats):
        outputs = {}
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            prefill_ms, decode_ms, outputs[enabled] = run(enabled)
            trials[enabled].append({"prefill_ms": prefill_ms, "decode_ms": decode_ms})
        a, b = outputs[False], outputs[True]
        max_error = max(max_error, (a - b).abs().max().item())
        close &= torch.allclose(a, b, atol=1e-5, rtol=1e-4)
        same_top1 &= torch.equal(a.argmax(-1), b.argmax(-1))
    result = {**vars(args), "device": str(device), "torch": torch.__version__, "precision": "fp32",
              "config_revision": artifacts.revision, "checkpoint_file": artifacts.checkpoint_file,
              "workload": "batch=1; seeded synthetic tokens; fixed teacher-forced cached decode",
              "seed": 42, "warmup_runs_per_mode": 2, "atol": 1e-5, "rtol": 1e-4,
              "logit_positions": "last prompt position and every decode step", "max_logit_error": max_error,
              "logits_close": close, "same_top1": same_top1, "modes": {}}
    for enabled, samples in trials.items():
        result["modes"]["absorbed" if enabled else "mla"] = {
            "median_prefill_ms": statistics.median(s["prefill_ms"] for s in samples),
            "median_decode_ms_per_token": statistics.median(s["decode_ms"] for s in samples) / args.decode_steps,
            "trials": samples}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    assets = path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    plot(result, assets / path.with_suffix(".png").name)
    print(json.dumps(result, indent=2))
    if not close:
        raise SystemExit("Absorbed logits exceeded atol=1e-5, rtol=1e-4")


if __name__ == "__main__":
    main()
