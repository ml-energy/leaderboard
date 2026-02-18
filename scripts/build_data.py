"""Build leaderboard data by composing mlenergy-data APIs.

All leaderboard-specific formatting (display names, JSON structure, sort order)
lives here.  The toolkit provides run table loading and output-length extraction.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlenergy_data.records import DiffusionRun, DiffusionRuns, LLMRun, LLMRuns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leaderboard-specific constants
# ---------------------------------------------------------------------------

LLM_TASKS = {"gpqa", "lm-arena-chat", "sourcegraph-fim"}
MLLM_TASKS = {"image-chat", "video-chat"}
DIFFUSION_TASKS = {"text-to-image", "text-to-video"}

TASK_DISPLAY_NAMES = {
    "gpqa": "GPQA Diamond",
    "lm-arena-chat": "LLM Chat (LM Arena)",
    "sourcegraph-fim": "Fill-in-the-Middle (Sourcegraph)",
    "image-chat": "Image Chat",
    "video-chat": "Video Chat",
    "text-to-image": "Text to Image",
    "text-to-video": "Text to Video",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _compute_output_length_distribution(
    lengths: np.ndarray,
    *,
    bins: int | None = None,
    bin_edges: np.ndarray | None = None,
) -> dict[str, list[float]]:
    if bin_edges is not None:
        counts, edges = np.histogram(lengths, bins=bin_edges)
    else:
        counts, edges = np.histogram(lengths, bins=(bins or 50))
    return {
        "bins": edges.tolist(),
        "counts": counts.tolist(),
    }


# ---------------------------------------------------------------------------
# Sort helpers for deterministic output
# ---------------------------------------------------------------------------


def _sort_llm_runs(runs: LLMRuns) -> list[LLMRun]:
    return sorted(
        runs,
        key=lambda r: (
            r.model_id, r.task, r.gpu_model, r.num_gpus,
            r.max_num_seqs,
            r.num_request_repeats or 1,
            r.seed or 0,
        ),
    )


def _sort_diffusion_runs(runs: DiffusionRuns) -> list[DiffusionRun]:
    return sorted(
        runs,
        key=lambda r: (
            r.model_id, r.task, r.gpu_model, r.num_gpus, r.batch_size,
            r.height,
            r.width,
            r.inference_steps or 0,
            r.ulysses_degree or 0,
            r.ring_degree or 0,
            r.use_torch_compile or False,
        ),
    )


# ---------------------------------------------------------------------------
# index.json
# ---------------------------------------------------------------------------


def _build_index_payload(
    llm: LLMRuns,
    diff: DiffusionRuns,
) -> dict[str, Any]:
    llm_mllm_tasks = sorted({r.task for r in llm})
    diffusion_tasks = sorted({r.task for r in diff})
    all_tasks = sorted(set(llm_mllm_tasks + diffusion_tasks))

    architectures = {
        "diffusion": sorted(t for t in diffusion_tasks if t in DIFFUSION_TASKS),
        "llm": sorted(t for t in llm_mllm_tasks if t in LLM_TASKS),
        "mllm": sorted(t for t in llm_mllm_tasks if t in MLLM_TASKS),
    }

    models: dict[str, dict[str, Any]] = {}
    for model_id, group in llm.group_by("model_id").items():
        first = group[0]
        models[str(model_id)] = {
            "activated_params_billions": float(first.activated_params_billions),
            "architecture": first.architecture,
            "nickname": first.nickname,
            "total_params_billions": float(first.total_params_billions),
            "weight_precision": first.weight_precision,
        }
    for model_id, group in diff.group_by("model_id").items():
        mid = str(model_id)
        if mid not in models:
            first = group[0]
            models[mid] = {
                "activated_params_billions": float(first.activated_params_billions),
                "nickname": first.nickname,
                "total_params_billions": float(first.total_params_billions),
                "weight_precision": first.weight_precision,
            }

    return {
        "architectures": architectures,
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "models": dict(sorted(models.items())),
        "tasks": all_tasks,
    }


# ---------------------------------------------------------------------------
# Task JSONs
# ---------------------------------------------------------------------------


def _build_llm_task_payload(task: str, runs: list[LLMRun]) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for r in runs:
        configs.append({
            "activated_params_billions": float(r.activated_params_billions),
            "architecture": r.architecture,
            "avg_batch_size": r.avg_batch_size,
            "avg_output_len": r.avg_output_len,
            "avg_power_watts": float(r.avg_power_watts),
            "data_parallel": int(r.data_parallel),
            "energy_per_request_joules": r.energy_per_request_joules,
            "energy_per_token_joules": float(r.energy_per_token_joules),
            "expert_parallel": int(r.expert_parallel),
            "gpu_model": r.gpu_model,
            "max_num_seqs": int(r.max_num_seqs),
            "median_itl_ms": float(r.median_itl_ms),
            "model_id": r.model_id,
            "nickname": r.nickname,
            "num_gpus": int(r.num_gpus),
            "output_throughput_tokens_per_sec": float(r.output_throughput_tokens_per_sec),
            "p90_itl_ms": float(r.p90_itl_ms),
            "p95_itl_ms": float(r.p95_itl_ms),
            "p99_itl_ms": float(r.p99_itl_ms),
            "tensor_parallel": int(r.tensor_parallel),
            "total_params_billions": float(r.total_params_billions),
            "weight_precision": r.weight_precision,
        })
    return {
        "configurations": configs,
        "task": task,
        "task_display_name": TASK_DISPLAY_NAMES.get(task, task),
    }


def _build_diffusion_task_payload(task: str, runs: list[DiffusionRun]) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for r in runs:
        rec: dict[str, Any] = {
            "activated_params_billions": float(r.activated_params_billions),
            "avg_power_watts": float(r.avg_power_watts),
            "batch_latency_s": float(r.batch_latency_s),
            "batch_size": int(r.batch_size),
            "gpu_model": r.gpu_model,
            "inference_steps": int(r.inference_steps),
            "model_id": r.model_id,
            "nickname": r.nickname,
            "num_gpus": int(r.num_gpus),
            "ring_degree": int(r.ring_degree),
            "total_params_billions": float(r.total_params_billions),
            "ulysses_degree": int(r.ulysses_degree),
            "weight_precision": r.weight_precision,
        }
        if task == "text-to-image":
            rec["energy_per_image_joules"] = float(r.energy_per_generation_joules)
            rec["throughput_images_per_sec"] = float(r.throughput_generations_per_sec)
            rec["image_height"] = int(r.height)
            rec["image_width"] = int(r.width)
        else:
            rec["energy_per_video_joules"] = float(r.energy_per_generation_joules)
            rec["throughput_videos_per_sec"] = float(r.throughput_generations_per_sec)
            rec["video_height"] = int(r.height)
            rec["video_width"] = int(r.width)
            rec["fps"] = int(r.fps)
            rec["num_frames"] = int(r.num_frames)
        configs.append(rec)
    return {
        "architecture": "diffusion",
        "configurations": configs,
        "task": task,
        "task_display_name": TASK_DISPLAY_NAMES.get(task, task),
    }


# ---------------------------------------------------------------------------
# Model detail JSONs
# ---------------------------------------------------------------------------


def _build_llm_model_payload(
    model_id: str,
    task: str,
    runs: list[LLMRun],
) -> dict[str, Any]:
    all_lengths_list: list[int] = []
    per_run_lengths: list[np.ndarray] = []
    for r in runs:
        lengths = np.array(r.output_lengths(), dtype=int)
        all_lengths_list.extend(lengths)
        per_run_lengths.append(lengths)

    all_lengths = np.array(all_lengths_list, dtype=int)
    if len(all_lengths) == 0:
        raise ValueError(f"No output length samples for {model_id} / {task}")
    _, bin_edges = np.histogram(all_lengths, bins=50)
    agg_dist = _compute_output_length_distribution(all_lengths, bin_edges=bin_edges)

    first = runs[0]
    configs: list[dict[str, Any]] = []
    for i, r in enumerate(runs):
        cfg_dist = _compute_output_length_distribution(
            per_run_lengths[i], bin_edges=bin_edges
        )

        dp = int(r.data_parallel)
        ep = int(r.expert_parallel)
        configs.append({
            "avg_batch_size": r.avg_batch_size,
            "avg_output_len": r.avg_output_len,
            "avg_power_watts": float(r.avg_power_watts),
            "energy_per_request_joules": r.energy_per_request_joules,
            "energy_per_token_joules": float(r.energy_per_token_joules),
            "gpu_model": r.gpu_model,
            "max_num_seqs": int(r.max_num_seqs),
            "median_itl_ms": float(r.median_itl_ms),
            "num_gpus": int(r.num_gpus),
            "output_length_distribution": cfg_dist,
            "output_throughput_tokens_per_sec": float(r.output_throughput_tokens_per_sec),
            "p90_itl_ms": float(r.p90_itl_ms),
            "p95_itl_ms": float(r.p95_itl_ms),
            "p99_itl_ms": float(r.p99_itl_ms),
            "parallelization": {
                "data_parallel": dp,
                "expert_parallel": ep,
                "notes": "DP for attention, EP for MLP experts" if dp > 1 and ep > 1 else "",
                "tensor_parallel": int(r.tensor_parallel),
            },
        })

    return {
        "activated_params_billions": float(first.activated_params_billions),
        "architecture": first.architecture,
        "configurations": configs,
        "model_id": model_id,
        "output_length_distribution": agg_dist,
        "task": task,
        "total_params_billions": float(first.total_params_billions),
        "weight_precision": first.weight_precision,
    }


def _build_diffusion_model_payload(
    model_id: str,
    task: str,
    runs: list[DiffusionRun],
) -> dict[str, Any]:
    is_video = task == "text-to-video"
    first = runs[0]

    configs: list[dict[str, Any]] = []
    for r in runs:
        rec: dict[str, Any] = {
            "avg_power_watts": float(r.avg_power_watts),
            "batch_latency_s": float(r.batch_latency_s),
            "batch_size": int(r.batch_size),
            "gpu_model": r.gpu_model,
            "inference_steps": int(r.inference_steps),
            "num_gpus": int(r.num_gpus),
            "parallelization": {
                "ring_degree": int(r.ring_degree),
                "ulysses_degree": int(r.ulysses_degree),
            },
        }
        if is_video:
            rec["energy_per_video_joules"] = float(r.energy_per_generation_joules)
            rec["throughput_videos_per_sec"] = float(r.throughput_generations_per_sec)
            rec["video_height"] = int(r.height)
            rec["video_width"] = int(r.width)
            rec["fps"] = int(r.fps)
            rec["num_frames"] = int(r.num_frames)
        else:
            rec["energy_per_image_joules"] = float(r.energy_per_generation_joules)
            rec["throughput_images_per_sec"] = float(r.throughput_generations_per_sec)
            rec["image_height"] = int(r.height)
            rec["image_width"] = int(r.width)
        configs.append(rec)

    return {
        "activated_params_billions": float(first.activated_params_billions),
        "configurations": configs,
        "model_id": model_id,
        "nickname": first.nickname,
        "task": task,
        "total_params_billions": float(first.total_params_billions),
        "weight_precision": first.weight_precision,
    }


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------


def _build_and_write_model_json(
    args: tuple[str, str, str, list[dict[str, Any]], str, int, int],
) -> tuple[str, str]:
    model_id, task, domain, rows, out_dir, idx, total = args
    logger.info("[%d/%d] Generating %s / %s", idx, total, model_id, task)

    if domain == "llm":
        llm_runs = [LLMRun(**r) for r in rows]
        payload = _build_llm_model_payload(model_id, task, llm_runs)
    else:
        diff_runs = [DiffusionRun(**r) for r in rows]
        payload = _build_diffusion_model_payload(model_id, task, diff_runs)

    out = Path(out_dir) / "models" / f"{model_id.replace('/', '__')}__{task}.json"
    _json_dump(out, payload)
    logger.info("[%d/%d] Generated %s", idx, total, out)
    return (model_id, task)


# ---------------------------------------------------------------------------
# Size validation
# ---------------------------------------------------------------------------


def _validate_size(output_dir: Path, max_size_mb: int) -> None:
    total_size = 0
    for p in output_dir.rglob("*.json"):
        total_size += p.stat().st_size
    size_mb = total_size / (1024 * 1024)
    logger.info("Total data size: %.2f MB", size_mb)
    if size_mb > max_size_mb:
        raise ValueError(f"Data size exceeds {max_size_mb} MB")
    logger.info("Size check passed (under %d MB limit)", max_size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leaderboard data from benchmark results")
    parser.add_argument(
        "--mlenergy-data-dir",
        type=str,
        default=None,
        help="Path to compiled ML.ENERGY Benchmark data directory. If omitted, loads from Hugging Face Hub.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="public/data",
        help="Directory to write output JSON files (default: public/data)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=10,
        help="Maximum allowed total size of generated JSON files in MB (default: 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all stable runs via toolkit
    if args.mlenergy_data_dir:
        logger.info("Loading runs from %s", args.mlenergy_data_dir)
        llm = LLMRuns.from_directory(args.mlenergy_data_dir)
        diff = DiffusionRuns.from_directory(args.mlenergy_data_dir)
    else:
        logger.info("Loading runs from Hugging Face Hub")
        llm = LLMRuns.from_hf()
        diff = DiffusionRuns.from_hf()
        llm.download_raw_files()
        diff.download_raw_files()
    if not llm and not diff:
        raise ValueError("No benchmark runs found")

    # Sort in leaderboard-canonical order
    sorted_llm = _sort_llm_runs(llm)
    sorted_diff = _sort_diffusion_runs(diff)
    logger.info("Total: %d LLM + %d diffusion stable runs", len(sorted_llm), len(sorted_diff))

    # Generate index.json
    logger.info("Generating index.json")
    _json_dump(output_dir / "index.json", _build_index_payload(llm, diff))

    # Generate per-task JSONs
    logger.info("Generating task JSONs")
    for task, group in llm.group_by("task").items():
        task_runs = [r for r in sorted_llm if r.task == task]
        payload = _build_llm_task_payload(str(task), task_runs)
        _json_dump(output_dir / "tasks" / f"{task}.json", payload)
    for task, group in diff.group_by("task").items():
        task_runs = [r for r in sorted_diff if r.task == task]
        payload = _build_diffusion_task_payload(str(task), task_runs)
        _json_dump(output_dir / "tasks" / f"{task}.json", payload)

    # Generate per-model detail JSONs (parallelized)
    logger.info("Generating model detail JSONs")
    import dataclasses

    pool_args: list[tuple[str, str, str, list[dict[str, Any]], str, int, int]] = []
    idx = 0

    for (model_id, task), group in llm.group_by("model_id", "task").items():
        idx += 1
        run_dicts = [dataclasses.asdict(r) for r in sorted_llm
                     if r.model_id == model_id and r.task == task]
        pool_args.append((
            str(model_id), str(task), "llm", run_dicts,
            str(output_dir), idx, 0,
        ))
    for (model_id, task), group in diff.group_by("model_id", "task").items():
        idx += 1
        run_dicts = [dataclasses.asdict(r) for r in sorted_diff
                     if r.model_id == model_id and r.task == task]
        pool_args.append((
            str(model_id), str(task), "diffusion", run_dicts,
            str(output_dir), idx, 0,
        ))

    total = len(pool_args)
    pool_args = [(mid, t, d, rows, od, i, total) for mid, t, d, rows, od, i, _ in pool_args]

    workers = min(max(1, multiprocessing.cpu_count()), max(1, total))
    logger.info("Using %d workers for %d model JSONs", workers, total)
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(_build_and_write_model_json, pool_args)

    # Validate size
    _validate_size(output_dir, int(args.max_size_mb))
    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
