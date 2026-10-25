from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import src  # noqa: F401 - registers repo components
from src.model.builder import build_causal_lm, build_transformer
from src.registry import DATASET


COLLECTION_SLUG = "Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0"
DEFAULT_DATASET = "meetingbank"
DEFAULT_MEETINGBANK_HF_PATH = "huuuyeah/meetingbank"
DEFAULT_OUTPUT_ROOT = Path("benchmarks") / "attention"
RESULT_FIELDS = [
    "repo_id",
    "status",
    "eval_loss",
    "perplexity",
    "token_accuracy",
    "top_5_accuracy",
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu",
    "bertscore_f1",
    "non_pad_tokens",
    "parameters",
    "checkpoint_size_mb",
    "device",
    "dtype",
    "eval_tokens_per_sec",
    "avg_forward_latency_ms",
    "generation_tokens_per_sec",
    "peak_cuda_memory_mb",
    "wall_time_sec",
    "checkpoint_file",
    "config_revision",
    "error",
]


@dataclass
class CoreMetricAccumulator:
    ignore_index: int
    loss_sum: float = 0.0
    token_count: int = 0
    correct_1: int = 0
    correct_5: int = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        vocab = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab)
        flat_labels = labels.reshape(-1)
        mask = flat_labels != self.ignore_index
        count = int(mask.sum().item())
        if count == 0:
            return

        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self.ignore_index,
            reduction="sum",
        )
        self.loss_sum += float(loss.item())
        self.token_count += count

        scored_logits = flat_logits[mask]
        scored_labels = flat_labels[mask]
        self.correct_1 += int((scored_logits.argmax(dim=-1) == scored_labels).sum().item())

        k = min(5, vocab)
        top_k = scored_logits.topk(k=k, dim=-1).indices
        self.correct_5 += int((top_k == scored_labels.unsqueeze(-1)).any(dim=-1).sum().item())

    def compute(self) -> dict[str, float | int | None]:
        if self.token_count == 0:
            return {
                "eval_loss": None,
                "perplexity": None,
                "token_accuracy": None,
                "top_5_accuracy": None,
                "non_pad_tokens": 0,
            }
        loss = self.loss_sum / self.token_count
        perplexity = math.inf if loss > 88.0 else math.exp(loss)
        return {
            "eval_loss": loss,
            "perplexity": perplexity,
            "token_accuracy": self.correct_1 / self.token_count,
            "top_5_accuracy": self.correct_5 / self.token_count,
            "non_pad_tokens": self.token_count,
        }


@dataclass
class RepoArtifacts:
    repo_id: str
    config_path: Path
    checkpoint_path: Path
    checkpoint_file: str
    revision: str | None
    tokenizer_dir: Path
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    repo_id: str
    status: str
    checkpoint_file: str | None = None
    config_revision: str | None = None
    eval_loss: float | None = None
    perplexity: float | None = None
    token_accuracy: float | None = None
    top_5_accuracy: float | None = None
    rouge1: float | None = None
    rouge2: float | None = None
    rougeL: float | None = None
    bleu: float | None = None
    bertscore_f1: float | None = None
    non_pad_tokens: int = 0
    parameters: int | None = None
    checkpoint_size_mb: float | None = None
    device: str | None = None
    dtype: str | None = None
    eval_tokens_per_sec: float | None = None
    avg_forward_latency_ms: float | None = None
    generation_tokens_per_sec: float | None = None
    peak_cuda_memory_mb: float | None = None
    wall_time_sec: float | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        _render_only(output_root, args)
        return

    output_dir = _run_output_dir(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = _hf_token(Path.cwd(), required=False)
    api = HfApi(token=token)
    repo_ids = args.repo_id or _collection_repo_ids(api, args.collection_slug)
    if not repo_ids:
        raise RuntimeError(f"no model repos found in {args.collection_slug}")

    settings = vars(args).copy()
    settings["repo_count"] = len(repo_ids)
    settings["started_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    results: list[BenchmarkResult] = []
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w") as jsonl:
        for repo_id in tqdm(repo_ids, desc="benchmark models", unit="model"):
            result = _benchmark_repo(api, repo_id, args, token)
            results.append(result)
            jsonl.write(json.dumps(_json_ready(asdict(result)), sort_keys=True) + "\n")
            jsonl.flush()
            _clear_cuda()

    _write_summary_csv(output_dir / "summary.csv", results)
    _write_readme(output_dir / "README.md", results, settings)
    _publish_consolidated_outputs(output_root, output_dir)
    print(f"wrote benchmark results to {output_dir}")


def _render_only(output_root: Path, args: argparse.Namespace) -> None:
    """rebuild summary.csv + charts + README from the existing results.jsonl,
    without downloading or evaluating any model checkpoints."""
    results = _load_results(output_root / "results.jsonl")
    if not results:
        raise RuntimeError(f"no results.jsonl in {output_root}; run a benchmark first")
    settings_path = output_root / "settings.json"
    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}
    settings.setdefault("collection_slug", args.collection_slug)
    settings.setdefault("dataset", args.dataset)
    settings.setdefault("split", args.split)
    settings.setdefault("generation_samples", args.generation_samples)
    settings.setdefault("precision", args.precision)
    _write_summary_csv(output_root / "summary.csv", results)
    _write_readme(output_root / "README.md", results, settings)
    print(f"re-rendered report in {output_root}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark custom transformer-lab checkpoints from a Hugging Face collection."
    )
    parser.add_argument("--collection-slug", default=COLLECTION_SLUG)
    parser.add_argument("--repo-id", action="append", default=[], help="Benchmark a specific repo; repeatable.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=[DEFAULT_DATASET])
    parser.add_argument("--dataset-hf-path", default=DEFAULT_MEETINGBANK_HF_PATH)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means full validation loader.")
    parser.add_argument("--generation-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument(
        "--bertscore",
        default="auto",
        choices=["auto", "off", "on"],
        help="auto attempts BERTScore when installed; off records null; on raises metric warnings on failure.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 uses model/data max_tgt_len.")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip evaluation; rebuild charts + README from the existing results.jsonl.",
    )
    return parser.parse_args()


def _benchmark_repo(api: HfApi, repo_id: str, args: argparse.Namespace, token: str | None) -> BenchmarkResult:
    start = time.monotonic()
    result = BenchmarkResult(repo_id=repo_id, status="failed")
    try:
        with tempfile.TemporaryDirectory(prefix="transformer-bench-") as tmp:
            artifacts = _download_artifacts(api, repo_id, Path(tmp), token)
            result.warnings.extend(artifacts.warnings)
            result.checkpoint_file = artifacts.checkpoint_file
            result.config_revision = artifacts.revision
            result.checkpoint_size_mb = artifacts.checkpoint_path.stat().st_size / 1_000_000

            cfg = OmegaConf.load(artifacts.config_path)
            _prepare_cfg(cfg, artifacts.tokenizer_dir, args)
            model_kind = str(cfg.model.get("kind", "encoder_decoder"))
            data = _build_data(cfg, args)
            _set_vocab_sizes(cfg, data)

            model = build_causal_lm(cfg) if model_kind == "causal_lm" else build_transformer(cfg)
            state = torch.load(artifacts.checkpoint_path, map_location="cpu")
            state_dict = state["model_state_dict"] if "model_state_dict" in state else state
            model.load_state_dict(state_dict)

            device = _resolve_device(args.device)
            model = model.to(device)
            model.eval()
            result.parameters = sum(p.numel() for p in model.parameters())
            result.device = str(device)
            result.dtype = args.precision

            core, perf = _evaluate_core(model, data, cfg, model_kind, device, args)
            for key, value in core.items():
                setattr(result, key, value)
            result.eval_tokens_per_sec = perf["eval_tokens_per_sec"]
            result.avg_forward_latency_ms = perf["avg_forward_latency_ms"]
            result.peak_cuda_memory_mb = perf["peak_cuda_memory_mb"]

            if not args.skip_generation and args.generation_samples > 0:
                gen_metrics, gen_perf, warnings = _evaluate_generation(
                    model, data, cfg, model_kind, device, args
                )
                result.warnings.extend(warnings)
                for key, value in gen_metrics.items():
                    setattr(result, key, value)
                result.generation_tokens_per_sec = gen_perf["generation_tokens_per_sec"]

            result.status = "ok"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.wall_time_sec = time.monotonic() - start
    return result


def _download_artifacts(api: HfApi, repo_id: str, tmp_root: Path, token: str | None) -> RepoArtifacts:
    files = set(_hf_call(f"list files for {repo_id}", api.list_repo_files, repo_id=repo_id, repo_type="model"))
    checkpoint_file = _checkpoint_file(files)
    if checkpoint_file is None:
        raise RuntimeError("missing checkpoint .pt file")
    if "config.json" not in files:
        raise RuntimeError("missing config.json")

    download_dir = tmp_root / "hf"
    download_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(
        _hf_call(
            f"download config.json from {repo_id}",
            hf_hub_download,
            repo_id=repo_id,
            filename="config.json",
            repo_type="model",
            token=token,
            local_dir=download_dir,
        )
    )
    checkpoint_path = Path(
        _hf_call(
            f"download {checkpoint_file} from {repo_id}",
            hf_hub_download,
            repo_id=repo_id,
            filename=checkpoint_file,
            repo_type="model",
            token=token,
            local_dir=download_dir,
        )
    )
    tokenizer_dir = tmp_root / "tokenizers"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    _download_tokenizer_if_present(
        repo_id, files, "transcript_tokenizer.json", tokenizer_dir / "meetingbank_transcript.json", token
    )
    _download_tokenizer_if_present(
        repo_id, files, "summary_tokenizer.json", tokenizer_dir / "meetingbank_summary.json", token
    )
    warnings: list[str] = []
    if "causal_tokenizer.json" in files:
        _download_tokenizer_if_present(
            repo_id, files, "causal_tokenizer.json", tokenizer_dir / "meetingbank_causal.json", token
        )
    elif "tokenizer.json" in files:
        _download_tokenizer_if_present(
            repo_id, files, "tokenizer.json", tokenizer_dir / "meetingbank_causal.json", token
        )

    # Older published causal checkpoints uploaded transcript tokenizer.json
    # instead of the unified causal tokenizer. Fall back to the local repo
    # causal tokenizer so benchmarking can reflect the actual checkpoint.
    config = OmegaConf.load(config_path)
    model_kind = str(config.model.get("kind", "encoder_decoder"))
    local_causal = Path("tokenizers") / "meetingbank_causal.json"
    if model_kind == "causal_lm" and not (tokenizer_dir / "meetingbank_causal.json").exists() and local_causal.exists():
        (tokenizer_dir / "meetingbank_causal.json").write_bytes(local_causal.read_bytes())
        warnings.append(f"used local causal tokenizer fallback for {repo_id}")
    elif model_kind == "causal_lm" and "causal_tokenizer.json" not in files and local_causal.exists():
        (tokenizer_dir / "meetingbank_causal.json").write_bytes(local_causal.read_bytes())
        warnings.append(f"replaced published tokenizer.json with local causal tokenizer fallback for {repo_id}")

    info = _hf_call(f"fetch model info for {repo_id}", api.model_info, repo_id=repo_id)
    return RepoArtifacts(
        repo_id=repo_id,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_file=checkpoint_file,
        revision=getattr(info, "sha", None),
        tokenizer_dir=tokenizer_dir,
        warnings=warnings,
    )


def _download_tokenizer_if_present(
    repo_id: str, files: set[str], filename: str, target: Path, token: str | None
) -> None:
    if filename not in files:
        return
    source = Path(
        _hf_call(
            f"download {filename} from {repo_id}",
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=target.parent,
        )
    )
    target.write_bytes(source.read_bytes())


def _hf_call(label: str, fn, *args, attempts: int = 3, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = 2 ** (attempt - 1)
            print(f"{label} failed ({type(exc).__name__}: {exc}); retrying in {wait}s")
            time.sleep(wait)


def _checkpoint_file(files: Iterable[str]) -> str | None:
    checkpoints = sorted(path for path in files if path.endswith(".pt"))
    if not checkpoints:
        return None
    if len(checkpoints) == 1:
        return checkpoints[0]

    def rank(path: str) -> tuple[int, str]:
        stem = Path(path).stem
        digits = ""
        for char in reversed(stem):
            if not char.isdigit():
                break
            digits = char + digits
        return (int(digits) if digits else -1, path)

    return sorted(checkpoints, key=rank)[-1]


def _prepare_cfg(cfg: DictConfig, tokenizer_dir: Path, args: argparse.Namespace) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.data.name = DEFAULT_DATASET
    cfg.data.hf_path = args.dataset_hf_path
    cfg.data.tokenizer_dir = str(tokenizer_dir)
    cfg.data.tokenizer_basename = "meetingbank"
    cfg.data.batch_size = args.batch_size
    cfg.data.val_batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.data.mode = "causal" if str(cfg.model.get("kind", "encoder_decoder")) == "causal_lm" else "encoder_decoder"
    OmegaConf.set_struct(cfg, True)


def _build_data(cfg: DictConfig, args: argparse.Namespace) -> dict[str, Any]:
    kwargs = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(kwargs, dict)
    name = str(kwargs.pop("name", DEFAULT_DATASET))
    kwargs["batch_size"] = args.batch_size
    kwargs["val_batch_size"] = args.batch_size
    kwargs["num_workers"] = args.num_workers
    return DATASET.build(name, **kwargs)


def _set_vocab_sizes(cfg: DictConfig, data: dict[str, Any]) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.model.src_vocab_size = data["src_vocab_size"]
    cfg.model.tgt_vocab_size = data["tgt_vocab_size"]
    OmegaConf.set_struct(cfg, True)


def _evaluate_core(
    model: torch.nn.Module,
    data: dict[str, Any],
    cfg: DictConfig,
    model_kind: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | None], dict[str, float | None]]:
    loader = data["val_loader"]
    pad_id = int(data["pad_token_id"])
    metrics = CoreMetricAccumulator(ignore_index=pad_id)
    latencies: list[float] = []
    start = time.monotonic()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="core eval", unit="batch", leave=False)):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                break
            _sync(device)
            batch_start = time.monotonic()
            with _autocast(device, args.precision):
                logits, labels = _forward_batch(model, batch, cfg, model_kind, device)
            _sync(device)
            latencies.append(time.monotonic() - batch_start)
            metrics.update(logits.float(), labels)

    elapsed = max(time.monotonic() - start, 1e-9)
    computed = metrics.compute()
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
    perf = {
        "eval_tokens_per_sec": computed["non_pad_tokens"] / elapsed if computed["non_pad_tokens"] else None,
        "avg_forward_latency_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else None,
        "peak_cuda_memory_mb": peak_mb,
    }
    return computed, perf


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    cfg: DictConfig,
    model_kind: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_kind == "causal_lm":
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        return model(input_ids), labels

    src_ids = batch["encoder_input"].to(device)
    tgt_ids = batch["decoder_input"].to(device)
    src_mask = batch["encoder_mask"].to(device)
    tgt_mask = batch["decoder_mask"].to(device)
    labels = batch["label"].to(device)
    enc_out = model.encode(src_ids, src_mask)
    dec_out = model.decode(enc_out, src_mask, tgt_ids, tgt_mask)
    return model.project(dec_out), labels


def _evaluate_generation(
    model: torch.nn.Module,
    data: dict[str, Any],
    cfg: DictConfig,
    model_kind: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float | None], dict[str, float | None], list[str]]:
    raw = load_dataset(args.dataset_hf_path, split=args.split)
    sample_count = min(args.generation_samples, len(raw))
    predictions: list[str] = []
    references: list[str] = []
    generated_tokens = 0
    start = time.monotonic()

    with torch.inference_mode():
        for row in tqdm(raw.select(range(sample_count)), desc="generation", unit="sample", leave=False):
            text, count = _generate_summary(model, data, cfg, model_kind, row["transcript"], device, args)
            predictions.append(text)
            references.append(row["summary"])
            generated_tokens += count

    elapsed = max(time.monotonic() - start, 1e-9)
    metrics, warnings = _text_metrics(predictions, references, args.bertscore)
    perf = {
        "generation_tokens_per_sec": generated_tokens / elapsed if generated_tokens else None,
    }
    return metrics, perf, warnings


def _generate_summary(
    model: torch.nn.Module,
    data: dict[str, Any],
    cfg: DictConfig,
    model_kind: str,
    transcript: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[str, int]:
    if model_kind == "causal_lm":
        tokenizer = data["tgt_tokenizer"]
        max_src_len = int(cfg.model.get("max_src_len", cfg.data.get("max_src_len", 512)))
        max_tgt_len = _max_new_tokens(cfg, args)
        bos_id = tokenizer.token_to_id("[SOS]")
        sep_id = tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("[EOS]")
        eos_id = tokenizer.token_to_id("[EOS]")
        src_ids = tokenizer.encode(transcript).ids[: max_src_len - 2]
        prompt = torch.tensor([[bos_id, *src_ids, sep_id]], dtype=torch.long, device=device)
        with _autocast(device, args.precision):
            output = model.generate(prompt, max_new_tokens=max_tgt_len, temperature=0.0, eos_id=eos_id)
        new_ids = output[0, prompt.shape[1] :].detach().cpu().tolist()
        return tokenizer.decode(_strip_after_eos(new_ids, eos_id), skip_special_tokens=True), len(new_ids)

    src_tokenizer = data["src_tokenizer"]
    tgt_tokenizer = data["tgt_tokenizer"]
    max_src_len = int(cfg.model.get("max_src_len", cfg.data.get("max_src_len", 512)))
    max_tgt_len = _max_new_tokens(cfg, args)
    src_sos = src_tokenizer.token_to_id("[SOS]")
    src_eos = src_tokenizer.token_to_id("[EOS]")
    src_pad = src_tokenizer.token_to_id("[PAD]")
    tgt_sos = tgt_tokenizer.token_to_id("[SOS]")
    tgt_eos = tgt_tokenizer.token_to_id("[EOS]")
    src_ids = src_tokenizer.encode(transcript).ids[: max_src_len - 2]
    padded_src = [src_sos, *src_ids, src_eos]
    padded_src.extend([src_pad] * (max_src_len - len(padded_src)))

    src_tensor = torch.tensor([padded_src], dtype=torch.long, device=device)
    src_mask = (src_tensor != src_pad).unsqueeze(0).unsqueeze(0).int()
    generated = [tgt_sos]
    with _autocast(device, args.precision):
        enc_out = model.encode(src_tensor, src_mask)
        for _ in range(max_tgt_len):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_mask = _decoder_mask(tgt_tensor, tgt_tokenizer.token_to_id("[PAD]"))
            dec_out = model.decode(enc_out, src_mask, tgt_tensor, tgt_mask)
            next_id = int(model.project(dec_out)[:, -1, :].argmax(dim=-1).item())
            generated.append(next_id)
            if next_id == tgt_eos:
                break
    new_ids = generated[1:]
    return tgt_tokenizer.decode(_strip_after_eos(new_ids, tgt_eos), skip_special_tokens=True), len(new_ids)


def _decoder_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    seq_len = tgt.shape[1]
    causal = torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.int, device=tgt.device))
    return (tgt != pad_id).unsqueeze(0).int().unsqueeze(0) & causal


def _strip_after_eos(ids: list[int], eos_id: int | None) -> list[int]:
    if eos_id is None:
        return ids
    if eos_id in ids:
        return ids[: ids.index(eos_id)]
    return ids


def _text_metrics(
    predictions: list[str],
    references: list[str],
    bertscore_mode: str,
) -> tuple[dict[str, float | None], list[str]]:
    metrics: dict[str, float | None] = {
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "bleu": None,
        "bertscore_f1": None,
    }
    warnings: list[str] = []
    if not predictions:
        return metrics, warnings

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure
        for key, value in totals.items():
            metrics[key] = value / len(predictions)
    except Exception as exc:
        warnings.append(f"rouge unavailable: {type(exc).__name__}: {exc}")

    try:
        import sacrebleu

        metrics["bleu"] = sacrebleu.corpus_bleu(predictions, [references]).score
    except Exception as exc:
        warnings.append(f"bleu unavailable: {type(exc).__name__}: {exc}")

    if bertscore_mode != "off":
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(predictions, references, lang="en", verbose=False)
            metrics["bertscore_f1"] = float(f1.mean().item())
        except Exception as exc:
            warnings.append(f"bertscore unavailable: {type(exc).__name__}: {exc}")
            if bertscore_mode == "on":
                metrics["bertscore_f1"] = None

    return metrics, warnings


def _max_new_tokens(cfg: DictConfig, args: argparse.Namespace) -> int:
    if args.max_new_tokens > 0:
        return args.max_new_tokens
    return int(cfg.model.get("max_tgt_len", cfg.data.get("max_tgt_len", 256)))


def _collection_repo_ids(api: HfApi, collection_slug: str) -> list[str]:
    collection = api.get_collection(collection_slug)
    return [item.item_id for item in collection.items if item.item_type == "model"]


def _hf_token(root: Path, required: bool) -> str | None:
    token = os.environ.get("HF_TOKEN")
    env_path = root / ".env"
    if not token and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    if required and not token:
        raise RuntimeError("HF_TOKEN is required in the environment or .env")
    return token


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is false")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps requested but torch.backends.mps.is_available() is false")
    return torch.device(name)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type == "cpu":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "runs" / stamp


def _publish_consolidated_outputs(root: Path, run_dir: Path) -> None:
    run_settings = json.loads((run_dir / "settings.json").read_text())
    run_results = _load_results(run_dir / "results.jsonl")

    consolidated_results = _load_results(root / "results.jsonl")
    merged_results = _merge_results(consolidated_results, run_results)

    if not consolidated_results or not run_settings.get("repo_id"):
        settings = run_settings
    else:
        settings = json.loads((root / "settings.json").read_text())
        settings["last_partial_update_at"] = run_settings.get("started_at")
        settings["last_partial_repo_ids"] = list(run_settings.get("repo_id", []))

    (root / "results.jsonl").write_text(
        "".join(json.dumps(_json_ready(asdict(result)), sort_keys=True) + "\n" for result in merged_results)
    )
    (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    _write_summary_csv(root / "summary.csv", merged_results)
    _write_readme(root / "README.md", merged_results, settings)


def _load_results(path: Path) -> list[BenchmarkResult]:
    if not path.exists():
        return []
    return [
        BenchmarkResult(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _merge_results(existing: list[BenchmarkResult], incoming: list[BenchmarkResult]) -> list[BenchmarkResult]:
    merged: dict[str, BenchmarkResult] = {result.repo_id: result for result in existing}
    for result in incoming:
        merged[result.repo_id] = result
    return list(merged.values())


def _write_summary_csv(path: Path, results: list[BenchmarkResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: _csv_value(row.get(field)) for field in RESULT_FIELDS})


# ---------------------------------------------------------------------------
# academic report rendering: per-variant colours, loss curves, metric charts
# ---------------------------------------------------------------------------

# fixed colourblind-friendly palette so a variant keeps the same colour in every
# figure (loss overlay, bar charts, scatter).
VARIANT_COLORS: dict[str, str] = {
    "mha":            "#4C72B0",
    "mqa":            "#DD8452",
    "gqa":            "#55A868",
    "gqa_rope":       "#C44E52",
    "sliding_window": "#8172B3",
    "sliding_gqa":    "#937860",
    "gemma3_hybrid":  "#DA8BC3",
    "mla":            "#419D95",
    "csa":            "#CCB974",
    "hca":            "#64B5CD",
    "msa":            "#E63946",
    "kda":            "#7B2CBF",
}
_FALLBACK_COLOR = "#444444"

# canonical collection URL + inline HF logo used in the report's Models section.
COLLECTION_URL = "https://huggingface.co/collections/Pradheep1647/transformer-lab-6a07fe3185f5728e217997e0"
HF_LOGO = (
    '<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" '
    'width="16" alt="🤗"/>'
)


def _variant_key(repo_id: str) -> str:
    """short variant name, e.g. 'msa' from 'owner/run_msa-meetingbank-...'."""
    name = repo_id.split("/", 1)[-1]
    if name.startswith("run_"):
        name = name[len("run_"):]
    if "-meetingbank-" in name:
        name = name.split("-meetingbank-", 1)[0]
    if name == "meeting_summarization_kda":
        return "kda"
    return name


def _variant_color(repo_id: str) -> str:
    return VARIANT_COLORS.get(_variant_key(repo_id), _FALLBACK_COLOR)


def _num(value: Any) -> float | None:
    """coerce a metric to a finite float, else None (handles None / 'inf')."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _download_loss_curve(repo_id: str, token: str | None):
    """fetch loss_curve.csv from the model repo -> (steps, losses) or None."""
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename="loss_curve.csv", repo_type="model", token=token
        )
        # dedupe by step (last write wins, like tensorboard) so concatenated or
        # out-of-order runs collapse into one clean monotonic curve.
        by_step: dict[int, float] = {}
        with open(local, newline="") as f:
            for row in csv.DictReader(f):
                loss = _num(row.get("loss"))
                if loss is not None:
                    by_step[int(float(row["step"]))] = loss
        if not by_step:
            return None
        steps = sorted(by_step)
        return steps, [by_step[s] for s in steps]
    except Exception:
        return None


def _model_kind(repo_id: str, token: str | None) -> str:
    """'causal_lm' (decoder-only) or 'encoder_decoder' (transformer), from the
    repo's config.json; best-effort, defaults to encoder_decoder."""
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename="config.json", repo_type="model", token=token
        )
        cfg = json.loads(Path(local).read_text())
        return str(cfg.get("model", {}).get("kind", "encoder_decoder"))
    except Exception:
        return "encoder_decoder"


def _ema(values: list[float], alpha: float = 0.04) -> list[float]:
    out: list[float] = []
    acc = values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def _downsample(xs: list, ys: list, max_points: int = 800):
    if len(xs) <= max_points:
        return xs, ys
    stride = len(xs) // max_points
    return xs[::stride], ys[::stride]


def _setup_mpl():
    """academic figure style; returns pyplot or None if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    return plt


def _panel_hbar(ax, data, xlabel, lower_better) -> None:
    """one horizontal-bar panel; best value at the top, value labels at bar ends."""
    data = [(n, v, c) for n, v, c in data if v is not None]
    data.sort(key=lambda t: t[1], reverse=not lower_better)
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    cols = [d[2] for d in data]
    y = list(range(len(names)))
    ax.barh(y, vals, color=cols, edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    vmax = max(vals) if vals else 1.0
    for yy, v in zip(y, vals):
        ax.text(v + vmax * 0.012, yy, f"{v:.3g}", va="center", ha="left", fontsize=8.5)
    ax.margins(x=0.16)


def _render_charts(
    assets_dir: Path,
    results: list[BenchmarkResult],
    token: str | None,
    kinds: dict[str, str],
) -> dict[str, str]:
    """render the report figures into assets_dir; returns {chart: relpath}.
    empty dict means rendering is unavailable -> caller falls back to a table."""
    plt = _setup_mpl()
    if plt is None:
        return {}
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return {}
    assets_dir.mkdir(parents=True, exist_ok=True)
    charts: dict[str, str] = {}

    def k(r): return _variant_key(r.repo_id)
    def c(r): return _variant_color(r.repo_id)
    # decoder-only (causal-LM) drawn solid; encoder-decoder transformers dashed.
    def style(r): return "-" if kinds.get(r.repo_id) == "causal_lm" else "--"

    # 1) training-loss overlay (pulled from each repo's loss_curve.csv)
    try:
        from matplotlib.lines import Line2D
        curves = []
        for r in ok:
            lc = _download_loss_curve(r.repo_id, token)
            if lc is None:
                continue
            steps, losses = _downsample(lc[0], lc[1])
            curves.append((k(r), c(r), steps, _ema(losses), style(r)))
        if curves:
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            for name, col, steps, losses, ls in sorted(curves, key=lambda t: t[0]):
                ax.plot(steps, losses, color=col, lw=1.7, ls=ls, label=name)
            ax.set_xlabel("training step")
            ax.set_ylabel("loss (EMA-smoothed)")
            ax.set_title("Training loss")
            variant_legend = ax.legend(ncol=2, loc="upper right")
            ax.add_artist(variant_legend)
            ax.legend(
                handles=[
                    Line2D([0], [0], color="#444444", ls="-", label="decoder-only (causal-LM)"),
                    Line2D([0], [0], color="#444444", ls="--", label="encoder–decoder (transformer)"),
                ],
                loc="lower left",
                fontsize=8.5,
                title="model type",
            )
            fig.savefig(assets_dir / "loss_curves.png", bbox_inches="tight")
            plt.close(fig)
            charts["loss_curves"] = "assets/loss_curves.png"
    except Exception:
        pass

    # 2+3) quality charts, split by family — perplexity/accuracy and ROUGE-L/BLEU
    # are not comparable across encoder-decoder vs decoder-only, so each family
    # gets its own figures.
    enc = [r for r in ok if kinds.get(r.repo_id) != "causal_lm"]
    dec = [r for r in ok if kinds.get(r.repo_id) == "causal_lm"]
    charts.update(_quality_charts(plt, assets_dir, enc, "encdec"))
    charts.update(_quality_charts(plt, assets_dir, dec, "causal"))

    # 4) throughput: eval vs generation tok/s (speed is comparable -> combined)
    try:
        import numpy as np
        names = [k(r) for r in ok]
        ev = [_num(r.eval_tokens_per_sec) or 0.0 for r in ok]
        gen = [_num(r.generation_tokens_per_sec) or 0.0 for r in ok]
        x = np.arange(len(names))
        w = 0.4
        fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * len(names)), 5))
        ax.bar(x - w / 2, ev, w, label="eval tok/s", color="#4C72B0", edgecolor="white")
        ax.bar(x + w / 2, gen, w, label="generation tok/s", color="#DD8452", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("tokens / second")
        ax.set_title("Throughput")
        ax.grid(axis="x", visible=False)
        ax.legend()
        fig.savefig(assets_dir / "throughput.png", bbox_inches="tight")
        plt.close(fig)
        charts["throughput"] = "assets/throughput.png"
    except Exception:
        pass

    # 5) quality vs efficiency scatter (combined; marker shape encodes family)
    try:
        from matplotlib.lines import Line2D
        pts = [(r, _num(r.generation_tokens_per_sec), _num(r.rougeL)) for r in ok]
        pts = [p for p in pts if p[1] is not None and p[2] is not None]
        if pts:
            fig, ax = plt.subplots(figsize=(8, 6))
            for r, xv, yv in pts:
                marker = "o" if kinds.get(r.repo_id) == "causal_lm" else "s"
                ax.scatter(xv, yv, s=120, color=c(r), marker=marker,
                           edgecolor="white", linewidth=1.2, zorder=3)
                ax.annotate(k(r), (xv, yv), textcoords="offset points",
                            xytext=(7, 4), fontsize=9)
            ax.set_xlabel("generation speed (tok/s)  →")
            ax.set_ylabel("ROUGE-L  →")
            ax.set_title("Quality vs. efficiency")
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444",
                       markersize=9, label="decoder-only"),
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#444444",
                       markersize=9, label="encoder–decoder"),
            ], loc="best", title="model type")
            fig.savefig(assets_dir / "tradeoff.png", bbox_inches="tight")
            plt.close(fig)
            charts["tradeoff"] = "assets/tradeoff.png"
    except Exception:
        pass

    return charts


def _quality_charts(plt, assets_dir: Path, subset: list, suffix: str) -> dict[str, str]:
    """per-family eval-quality (PPL + token acc) and generation-quality
    (ROUGE-L + BLEU) figures. keyed '<metric>_<suffix>'."""
    out: dict[str, str] = {}
    if not subset:
        return out

    def k(r): return _variant_key(r.repo_id)
    def c(r): return _variant_color(r.repo_id)
    h = 0.62 * len(subset) + 1.8

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, h))
        _panel_hbar(axes[0], [(k(r), _num(r.perplexity), c(r)) for r in subset],
                    "perplexity  (lower is better)", lower_better=True)
        axes[0].set_title("Eval perplexity")
        _panel_hbar(axes[1], [(k(r), _num(r.token_accuracy), c(r)) for r in subset],
                    "token accuracy  (higher is better)", lower_better=False)
        axes[1].set_title("Token accuracy")
        fig.savefig(assets_dir / f"eval_quality_{suffix}.png", bbox_inches="tight")
        plt.close(fig)
        out[f"eval_quality_{suffix}"] = f"assets/eval_quality_{suffix}.png"
    except Exception:
        pass

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, h))
        _panel_hbar(axes[0], [(k(r), _num(r.rougeL), c(r)) for r in subset],
                    "ROUGE-L  (higher is better)", lower_better=False)
        axes[0].set_title("ROUGE-L")
        _panel_hbar(axes[1], [(k(r), _num(r.bleu), c(r)) for r in subset],
                    "BLEU  (higher is better)", lower_better=False)
        axes[1].set_title("BLEU")
        fig.savefig(assets_dir / f"generation_quality_{suffix}.png", bbox_inches="tight")
        plt.close(fig)
        out[f"generation_quality_{suffix}"] = f"assets/generation_quality_{suffix}.png"
    except Exception:
        pass

    return out


def _highlights(ok: list[BenchmarkResult]) -> list[str]:
    def best(metric: str, lower: bool):
        cand = [(_num(getattr(r, metric)), _variant_key(r.repo_id)) for r in ok]
        cand = [c for c in cand if c[0] is not None]
        if not cand:
            return None
        return (min if lower else max)(cand)
    out = []
    ppl = best("perplexity", True)
    if ppl:
        out.append(f"Lowest perplexity — **{ppl[1]}** ({ppl[0]:.3g}).")
    rl = best("rougeL", False)
    if rl:
        out.append(f"Best generation overlap (ROUGE-L) — **{rl[1]}** ({rl[0]:.3g}).")
    gs = best("generation_tokens_per_sec", False)
    if gs:
        out.append(f"Fastest generation — **{gs[1]}** ({gs[0]:.0f} tok/s).")
    return out


def _metrics_table_lines(results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "| Attention Variant | Repo | Status | Loss | PPL | Tok Acc | ROUGE-L | BLEU | Tok/s | Gen tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda r: (r.status != "ok", _sort_value(r.perplexity))):
        lines.append(
            "| "
            + " | ".join(
                [
                    _attention_variant_label(result.repo_id),
                    f"[`{result.repo_id}`](https://huggingface.co/{result.repo_id})",
                    result.status,
                    _fmt(result.eval_loss),
                    _fmt(result.perplexity),
                    _fmt(result.token_accuracy),
                    _fmt(result.rougeL),
                    _fmt(result.bleu),
                    _fmt(result.eval_tokens_per_sec),
                    _fmt(result.generation_tokens_per_sec),
                ]
            )
            + " |"
        )
    return lines


def _warning_lines(results: list[BenchmarkResult]) -> list[str]:
    warnings = sorted(set(warning for result in results for warning in result.warnings))
    if not warnings:
        return []
    lines = ["## Warnings", ""]
    if any("unavailable" in warning for warning in warnings):
        lines.extend([
            "Some optional generation metrics were skipped because their packages are not installed.",
            "Install them with `uv sync --extra benchmark`, then rerun the benchmark to populate the blank metric columns.",
            "",
        ])
    lines.extend(f"- {warning}" for warning in warnings)
    return lines


def _write_readme_table(path: Path, results: list[BenchmarkResult], settings: dict[str, Any]) -> None:
    """plain table README — fallback when chart rendering is unavailable."""
    rows = [
        "# Transformer Lab Benchmark",
        "",
        f"- Collection: `{settings.get('collection_slug', '')}`",
        f"- Dataset: `{settings.get('dataset', 'meetingbank')}` / `{settings.get('split', 'validation')}`",
        f"- Generation samples: `{settings.get('generation_samples', '')}`",
        f"- Precision: `{settings.get('precision', 'fp32')}`",
        "- Archived runs: `runs/<timestamp>/` under this folder.",
        "",
    ]
    rows.extend(_metrics_table_lines(results))
    warn = _warning_lines(results)
    if warn:
        rows.append("")
        rows.extend(warn)
    path.write_text("\n".join(rows) + "\n")


def _write_readme(path: Path, results: list[BenchmarkResult], settings: dict[str, Any]) -> None:
    assets_dir = path.parent / "assets"
    token = _hf_token(Path.cwd(), required=False)
    kinds = {r.repo_id: _model_kind(r.repo_id, token) for r in results}
    try:
        charts = _render_charts(assets_dir, results, token, kinds)
    except Exception:
        charts = {}

    if not charts:
        _write_readme_table(path, results, settings)
        return

    ok = [r for r in results if r.status == "ok"]
    dec_only = sorted({_variant_key(r.repo_id) for r in ok if kinds.get(r.repo_id) == "causal_lm"})
    enc_dec = sorted({_variant_key(r.repo_id) for r in ok if kinds.get(r.repo_id) != "causal_lm"})
    enc_results = [r for r in results if kinds.get(r.repo_id) != "causal_lm"]
    dec_results = [r for r in results if kinds.get(r.repo_id) == "causal_lm"]
    enc_ok = [r for r in ok if kinds.get(r.repo_id) != "causal_lm"]
    dec_ok = [r for r in ok if kinds.get(r.repo_id) == "causal_lm"]
    precisions = sorted({str(result.dtype) for result in ok if result.dtype})
    precision_label = " / ".join(precisions) or str(settings.get("precision", "fp32"))

    lines = [
        "# Transformer Lab — Attention Benchmark",
        "",
        "Attention variants trained on MeetingBank, grouped into the two model families they were "
        "trained in — **encoder–decoder transformers** and **decoder-only (causal-LM)** models. "
        "Because the families optimise different objectives, the quality metrics are benchmarked "
        "**separately per family**; only the cross-cutting views (training loss, throughput, and "
        "the speed/quality overview) place them on a shared axis.",
        "",
        f"**Setup** — dataset `{settings.get('dataset', 'meetingbank')}` / "
        f"`{settings.get('split', 'validation')}` · generation samples "
        f"`{settings.get('generation_samples', '?')}` · precision "
        f"`{precision_label}` by checkpoint · "
        f"[model collection]({COLLECTION_URL}).",
        "",
    ]

    def chart_block(level: int, title: str, chart_key: str, takeaway: str) -> None:
        if chart_key not in charts:
            return
        lines.extend(["#" * level + f" {title}", "", f"![{title}]({charts[chart_key]})", "", takeaway, ""])

    # --- cross-family views (shared, comparable axes) ---
    loss_takeaway = (
        "Training loss vs. step, pulled from each model's `loss_curve.csv` on the Hub and "
        "exponentially smoothed. **Solid** lines are decoder-only / causal-LM models"
        + (f" ({', '.join(dec_only)})" if dec_only else "")
        + "; **dashed** lines are encoder–decoder transformers"
        + (f" ({', '.join(enc_dec)})" if enc_dec else "")
        + ". Shown together for reference, but the two families optimise different objectives so "
        "absolute loss is not directly comparable."
    )
    chart_block(2, "Training loss", "loss_curves", loss_takeaway)
    chart_block(2, "Throughput", "throughput",
                "Evaluation and autoregressive-generation speed in tokens per second — the metric "
                "that is directly comparable across both families.")
    if "tradeoff" in charts:
        lines.extend([
            "## Quality vs. efficiency",
            "",
            "> Cross-family overview only. ROUGE-L against generation speed for all variants "
            "(circles = decoder-only, squares = encoder–decoder). Quality is *not* comparable "
            "across families — see each family's own section below for the head-to-head numbers.",
            "",
            f"![Quality vs. efficiency]({charts['tradeoff']})",
            "",
        ])

    # --- per-family quality ---
    def family_section(title: str, suffix: str, subset_ok: list, table_results: list, lead=None) -> None:
        lines.extend([f"## {title}", ""])
        if lead:
            lines.extend([lead, ""])
        hi = _highlights(subset_ok)
        if hi:
            lines.append("**Highlights**")
            lines.append("")
            lines.extend(f"- {h}" for h in hi)
            lines.append("")
        chart_block(3, "Evaluation quality", f"eval_quality_{suffix}",
                    "Teacher-forced perplexity (lower is better) and next-token accuracy on the "
                    "held-out validation split.")
        chart_block(3, "Generation quality", f"generation_quality_{suffix}",
                    "Summary-generation overlap against reference summaries (ROUGE-L and BLEU).")
        lines.extend(["<details>", f"<summary>Raw metrics — {title}</summary>", ""])
        lines.extend(_metrics_table_lines(table_results))
        lines.extend(["", "</details>", ""])

    causal_lead = (
        "> **Why this family scores much lower on generation.** Encoder–decoder models get "
        "cross-attention that aligns the decoder directly to the source transcript, making the "
        "copy-heavy summarization task far easier; decoder-only models must learn that purely "
        "in-context. These models also compress or sparsify the KV cache, discarding source "
        "detail that copying needs, and they show the classic teacher-forcing gap — msa reaches "
        "the lowest perplexity of any model yet near-zero BLEU, because low next-token loss does "
        "not survive free-running generation. Compare these variants against each other, not "
        "against the encoder–decoder family above."
    )
    family_section("Encoder–decoder transformers", "encdec", enc_ok, enc_results)
    family_section("Decoder-only (causal-LM)", "causal", dec_ok, dec_results, lead=causal_lead)

    # --- model index, grouped by family ---
    lines.extend([
        "## Models",
        "",
        f"All checkpoints are published in the {HF_LOGO} "
        f"[transformer-lab collection]({COLLECTION_URL}).",
        "",
    ])
    for group_title, group in (("Encoder–decoder", enc_results), ("Decoder-only", dec_results)):
        if not group:
            continue
        lines.extend([f"**{group_title}**", ""])
        for result in sorted(group, key=lambda r: (r.status != "ok", _sort_value(r.perplexity))):
            lines.append(
                f"- {HF_LOGO} **{_variant_key(result.repo_id)}** "
                f"[`{result.repo_id}`](https://huggingface.co/{result.repo_id}) ({result.status})"
            )
        lines.append("")

    warn = _warning_lines(results)
    if warn:
        lines.extend(warn)
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n")


def _attention_variant_label(repo_id: str) -> str:
    variant = repo_id.split("/", 1)[1]
    if variant.startswith("run_"):
        variant = variant[len("run_") :]
    if "-meetingbank-" in variant:
        variant = variant.split("-meetingbank-", 1)[0]

    labels = {
        "mha": "mha (multi-head attention)",
        "mqa": "mqa (multi-query attention)",
        "gqa": "gqa (grouped-query attention)",
        "gqa_rope": "gqa_rope (grouped-query attention + RoPE)",
        "sliding_gqa": "sliding_gqa (sliding-window grouped-query attention)",
        "csa": "csa (compressed sparse attention)",
        "hca": "hca (heavily compressed attention)",
        "mla": "mla (multi-head latent attention)",
        "msa": "msa (minimax sparse attention)",
        "meeting_summarization_kda": "kda (Kimi delta attention)",
    }
    return labels.get(variant, variant.replace("_", " "))


def _sort_value(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return math.inf
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4g}"
    return str(value)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()
