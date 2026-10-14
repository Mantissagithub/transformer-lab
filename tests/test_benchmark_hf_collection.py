import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_hf_collection.py"
SPEC = importlib.util.spec_from_file_location("benchmark_hf_collection", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

BenchmarkResult = benchmark.BenchmarkResult
CoreMetricAccumulator = benchmark.CoreMetricAccumulator
_checkpoint_file = benchmark._checkpoint_file
_merge_results = benchmark._merge_results
_publish_consolidated_outputs = benchmark._publish_consolidated_outputs
_write_summary_csv = benchmark._write_summary_csv


def test_core_metric_accumulator_masks_pad_and_computes_accuracy():
    pad = 0
    logits = torch.tensor(
        [
            [
                [0.0, 4.0, 1.0, 2.0, 3.0, -1.0],
                [5.0, 1.0, 0.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            ]
        ]
    )
    labels = torch.tensor([[1, 5, pad]])

    metrics = CoreMetricAccumulator(ignore_index=pad)
    metrics.update(logits, labels)
    out = metrics.compute()

    expected_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=pad,
        reduction="sum",
    ).item() / 2
    assert out["non_pad_tokens"] == 2
    assert math.isclose(out["eval_loss"], expected_loss)
    assert math.isclose(out["perplexity"], math.exp(expected_loss))
    assert out["token_accuracy"] == 0.5
    assert out["top_5_accuracy"] == 1.0


def test_checkpoint_file_prefers_highest_numeric_suffix():
    files = {
        "README.md",
        "meeting_model04.pt",
        "meeting_model12.pt",
        "meeting_model02.pt",
    }
    assert _checkpoint_file(files) == "meeting_model12.pt"


def test_write_summary_csv_keeps_failed_rows(tmp_path):
    path = tmp_path / "summary.csv"
    results = [
        BenchmarkResult(repo_id="owner/good", status="ok", eval_loss=1.2, perplexity=3.3),
        BenchmarkResult(repo_id="owner/bad", status="failed", error="boom"),
    ]
    _write_summary_csv(path, results)

    with path.open() as f:
        rows = list(csv.DictReader(f))

    assert [row["repo_id"] for row in rows] == ["owner/good", "owner/bad"]
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "failed"
    assert rows[1]["error"] == "boom"


def test_publish_consolidated_outputs_copies_latest_run_files(tmp_path):
    root = tmp_path / "benchmarks" / "attention"
    run_dir = root / "runs" / "20260518_000000"
    run_dir.mkdir(parents=True)

    result = BenchmarkResult(repo_id="owner/a", status="ok", eval_loss=1.2)
    (run_dir / "results.jsonl").write_text(json.dumps(benchmark._json_ready(benchmark.asdict(result))) + "\n")
    (run_dir / "settings.json").write_text(
        json.dumps(
            {
                "collection_slug": "owner/collection",
                "dataset": "meetingbank",
                "split": "validation",
                "generation_samples": 128,
                "precision": "fp32",
                "repo_id": [],
                "started_at": "2026-05-18T00:00:00+00:00",
            }
        )
        + "\n"
    )
    (run_dir / "README.md").write_text("run readme\n")
    (run_dir / "summary.csv").write_text("repo_id,status\nowner/a,ok\n")

    _publish_consolidated_outputs(root, run_dir)

    assert "owner/a" in (root / "results.jsonl").read_text()
    assert "owner/a" in (root / "summary.csv").read_text()
    assert "owner/a" in (root / "README.md").read_text()
    assert json.loads((root / "settings.json").read_text())["collection_slug"] == "owner/collection"


def test_merge_results_replaces_matching_repo_and_keeps_others():
    existing = [
        BenchmarkResult(repo_id="owner/a", status="ok", eval_loss=1.0),
        BenchmarkResult(repo_id="owner/b", status="ok", eval_loss=2.0),
    ]
    incoming = [
        BenchmarkResult(repo_id="owner/b", status="ok", eval_loss=3.0),
        BenchmarkResult(repo_id="owner/c", status="failed"),
    ]

    merged = {result.repo_id: result for result in _merge_results(existing, incoming)}

    assert set(merged) == {"owner/a", "owner/b", "owner/c"}
    assert merged["owner/a"].eval_loss == 1.0
    assert merged["owner/b"].eval_loss == 3.0
    assert merged["owner/c"].status == "failed"
