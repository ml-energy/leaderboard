#!/usr/bin/env python3
"""Build leaderboard data from benchmark results.

This script scans the results directory, extracts metrics, compresses timelines,
and generates optimized JSON files for the leaderboard website.
"""

import argparse
import json
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ============================================================================
# RDP Timeline Compression
# ============================================================================


def perpendicular_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Calculate perpendicular distance from point to line segment.

    Args:
        point: (x, y) point to measure distance from
        line_start: (x, y) start of line segment
        line_end: (x, y) end of line segment

    Returns:
        Perpendicular distance in Y-axis units (e.g., Watts for power)
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    if x2 == x1:
        return abs(x0 - x1)

    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return numerator / denominator if denominator > 0 else 0


def rdp_compress(
    timestamps: List[float],
    values: List[float],
    epsilon: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Compress timeline using Ramer-Douglas-Peucker algorithm.

    Recursively removes points that are within epsilon distance from the
    line connecting their neighbors. Preserves start and end points.

    Args:
        timestamps: List of timestamps (X-axis)
        values: List of values (Y-axis, e.g., power in Watts)
        epsilon: Maximum perpendicular distance threshold
                 For power: 0.5W gives ~95% compression with no visual difference
                 For batch size: 1.0 (step function, compresses extremely well)

    Returns:
        (compressed_timestamps, compressed_values)
    """
    if len(timestamps) <= 2:
        return timestamps, values

    points = list(zip(timestamps, values))
    line_start = points[0]
    line_end = points[-1]

    max_distance = 0
    max_index = 0

    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], line_start, line_end)
        if distance > max_distance:
            max_distance = distance
            max_index = i

    if max_distance > epsilon:
        left_t, left_v = rdp_compress(
            timestamps[: max_index + 1],
            values[: max_index + 1],
            epsilon,
        )
        right_t, right_v = rdp_compress(
            timestamps[max_index:],
            values[max_index:],
            epsilon,
        )

        return left_t + right_t[1:], left_v + right_v[1:]
    else:
        return [timestamps[0], timestamps[-1]], [values[0], values[-1]]


def compress_timeline(timeline_data: dict, epsilon: float = 0.5) -> dict:
    """Compress a timeline dict with timestamps and values.

    Args:
        timeline_data: Dict with 'timestamps' and 'values' keys
        epsilon: RDP epsilon parameter

    Returns:
        Compressed timeline dict with same structure
    """
    timestamps = timeline_data["timestamps"]
    values = timeline_data["values"]

    compressed_t, compressed_v = rdp_compress(timestamps, values, epsilon)

    return {"timestamps": compressed_t, "values": compressed_v}


# ============================================================================
# Data Structures
# ============================================================================


# Global cache for loaded JSON files to avoid redundant disk reads
_FILE_CACHE: Dict[Path, dict] = {}


def load_json_cached(file_path: Path) -> dict:
    """Load JSON file with caching to avoid redundant reads."""
    if file_path not in _FILE_CACHE:
        with open(file_path) as f:
            _FILE_CACHE[file_path] = json.load(f)
    return _FILE_CACHE[file_path]


@dataclass
class BenchmarkRun:
    """Metadata for a single benchmark run."""

    model_id: str
    gpu_model: str
    num_gpus: int
    task: str
    max_num_seqs: Optional[int]
    max_num_batched_tokens: Optional[int]
    num_request_repeats: int
    seed: int
    results_path: Path
    prometheus_path: Path
    config_dir: Path


# ============================================================================
# Directory Scanning and Parsing
# ============================================================================


def parse_dir_params(dirname: str) -> dict:
    """Parse parameters from directory name.

    Directory names follow pattern: key1+value1+key2+value2+...

    Args:
        dirname: Directory name like "num_gpus+1+max_num_seqs+128+seed+48105"

    Returns:
        Dict mapping parameter names to values
    """
    parts = dirname.split("+")
    params = {}

    for i in range(0, len(parts) - 1, 2):
        key = parts[i]
        value = parts[i + 1]

        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                if value == "None":
                    params[key] = None
                else:
                    params[key] = value

    return params


def scan_results_directory(results_dir: str) -> List[BenchmarkRun]:
    """Scan results directory and collect metadata for all runs.

    Args:
        results_dir: Path to results directory (e.g., "results/")

    Returns:
        List of BenchmarkRun objects
    """
    results_root = Path(results_dir)
    runs = []

    for results_json_path in results_root.rglob("results.json"):
        parts = results_json_path.relative_to(results_root).parts

        if parts[0] in ("llm", "mllm"):
            if len(parts) < 8:
                continue
            task = parts[1]
            org = parts[3]
            model_name = parts[4]
            gpu_model = parts[5]
            params_dir = parts[6]
        else:
            if len(parts) < 5:
                continue
            org = parts[0]
            model_name = parts[1]
            gpu_model = parts[2]
            params_dir = parts[3]

            config_base = Path("configs/vllm")
            task = None
            for task_candidate in [
                "lm-arena-chat",
                "gpqa",
                "sourcegraph-fim",
                "image-chat",
                "video-chat",
                "audio-chat",
            ]:
                config_path = (
                    config_base / task_candidate / f"{org}/{model_name}" / gpu_model / "monolithic.config.yaml"
                )
                if config_path.exists():
                    task = task_candidate
                    break

            if task is None:
                print(f"Warning: Could not determine task for {org}/{model_name} on {gpu_model}")
                continue

        model_id = f"{org}/{model_name}"
        params = parse_dir_params(params_dir)
        config_base = Path("configs/vllm")

        # Validate required parameters exist
        try:
            num_gpus = params["num_gpus"]
        except KeyError:
            raise ValueError(f"Missing required parameter 'num_gpus' in {results_json_path}")

        run = BenchmarkRun(
            model_id=model_id,
            gpu_model=gpu_model,
            num_gpus=num_gpus,
            task=task,
            max_num_seqs=params.get("max_num_seqs"),  # Optional
            max_num_batched_tokens=params.get("max_num_batched_tokens"),  # Optional
            num_request_repeats=params.get("num_request_repeats", 1),  # Has reasonable default
            seed=params.get("seed", 42),  # Has reasonable default
            results_path=results_json_path,
            prometheus_path=results_json_path.parent / "prometheus.json",
            config_dir=config_base / task / model_id / gpu_model,
        )

        runs.append(run)

    print(f"Found {len(runs)} benchmark runs")
    return runs


# ============================================================================
# Metrics Extraction
# ============================================================================


def extract_metrics(run: BenchmarkRun) -> Dict:
    """Extract all metrics from a benchmark run.

    Args:
        run: BenchmarkRun object with paths to results files

    Returns:
        Dict with all extracted metrics
    """
    results = load_json_cached(run.results_path)
    prometheus = load_json_cached(run.prometheus_path)

    # Validate required fields exist in results.json
    required_fields = [
        "steady_state_energy_per_token", "steady_state_energy", "output_throughput",
        "request_throughput", "total_token_throughput", "total_input_tokens",
        "total_output_tokens", "completed", "duration", "steady_state_duration", "num_prompts"
    ]
    for field in required_fields:
        if field not in results:
            raise ValueError(f"Missing required field '{field}' in {run.results_path}")

    metrics = {
        "energy_per_token_joules": results["steady_state_energy_per_token"],
        "total_energy_joules": results["steady_state_energy"],
        "output_throughput_tokens_per_sec": results["output_throughput"],
        "request_throughput_req_per_sec": results["request_throughput"],
        "total_token_throughput": results["total_token_throughput"],
        "total_input_tokens": results["total_input_tokens"],
        "total_output_tokens": results["total_output_tokens"],
        "avg_output_len": results["total_output_tokens"] / results["completed"],
        "duration_seconds": results["duration"],
        "steady_state_duration_seconds": results["steady_state_duration"],
        "num_requests": results["num_prompts"],
        "completed_requests": results["completed"],
    }

    metrics["energy_per_request_joules"] = (
        metrics["energy_per_token_joules"] * metrics["avg_output_len"]
    )

    # Validate Prometheus data exists (critical for ITL and batch metrics)
    if "steady_state_stats" not in prometheus:
        raise ValueError(
            f"Missing 'steady_state_stats' in prometheus.json for {run.results_path}\n"
            f"This indicates incomplete benchmark data. Re-run this benchmark."
        )

    prom_stats = prometheus["steady_state_stats"]

    # Extract ITL percentiles - these are CRITICAL metrics
    required_itl_percentiles = [50, 90, 95, 99]
    for p in required_itl_percentiles:
        key = f"vllm:inter_token_latency_seconds_p{p}"
        if key not in prom_stats:
            raise ValueError(
                f"Missing required metric '{key}' in prometheus steady_state_stats\n"
                f"File: {run.prometheus_path}\n"
                f"This indicates incomplete vLLM metrics. Re-run this benchmark."
            )
        metrics[f"p{p}_itl_ms"] = prom_stats[key] * 1000

    metrics["median_itl_ms"] = metrics["p50_itl_ms"]

    # Validate batch size metric exists (critical for detecting unstable runs)
    if "vllm:num_requests_running" not in prom_stats:
        raise ValueError(
            f"Missing required metric 'vllm:num_requests_running' in prometheus\n"
            f"File: {run.prometheus_path}\n"
            f"This indicates incomplete vLLM metrics. Re-run this benchmark."
        )
    metrics["avg_batch_size"] = prom_stats["vllm:num_requests_running"]

    return metrics


def parse_prometheus_histogram(metrics_text: str, metric_name: str) -> Dict:
    """Parse histogram from Prometheus text format.

    Returns dict with buckets, sum, count.
    """
    buckets = {}

    bucket_pattern = rf"^{re.escape(metric_name)}_bucket\{{([^}}]*)\}}\s+([\d.eE+-]+)"

    for match in re.finditer(bucket_pattern, metrics_text, re.MULTILINE):
        labels = match.group(1)
        count = float(match.group(2))

        le_match = re.search(r'le="([^"]+)"', labels)
        if le_match:
            le_str = le_match.group(1)
            if le_str == "+Inf":
                le = float("inf")
            else:
                le = float(le_str)
            buckets[le] = count

    return {"buckets": buckets}


def calculate_histogram_percentile(histogram_data: Dict, percentile: float) -> Optional[float]:
    """Calculate percentile from histogram buckets using linear interpolation."""
    buckets = histogram_data.get("buckets", {})
    if not buckets:
        return None

    sorted_buckets = sorted(buckets.items())

    total_count = sorted_buckets[-1][1]
    if total_count == 0:
        return None

    target_count = total_count * (percentile / 100.0)

    prev_upper = 0.0
    prev_count = 0.0

    for upper_bound, cumulative_count in sorted_buckets:
        if cumulative_count >= target_count:
            if prev_count == cumulative_count:
                return prev_upper

            bucket_width = upper_bound - prev_upper
            count_in_bucket = cumulative_count - prev_count
            fraction = (target_count - prev_count) / count_in_bucket

            return prev_upper + fraction * bucket_width

        prev_upper = upper_bound
        prev_count = cumulative_count

    return None


def generate_cdf_points(histogram_data: Dict, num_points: int = 100) -> List[List[float]]:
    """Generate points for CDF plot from histogram.

    Returns list of [x, y] points where y is cumulative probability (0-1).
    """
    buckets = histogram_data.get("buckets", {})
    if not buckets:
        return []

    sorted_buckets = sorted(buckets.items())
    total_count = sorted_buckets[-1][1]

    if total_count == 0:
        return []

    percentiles = np.linspace(0, 100, num_points)
    points = []

    for p in percentiles:
        value = calculate_histogram_percentile(histogram_data, p)
        if value is not None:
            points.append([value, p / 100.0])

    return points


# ============================================================================
# Timeline Extraction
# ============================================================================


def aggregate_device_timelines(
    device_data: Dict[str, List],
    start_time: float,
    end_time: float,
) -> Dict[str, List]:
    """Aggregate timelines across multiple devices and filter to steady state.

    Args:
        device_data: Dict mapping device_id to [[timestamp, value], ...]
        start_time: Steady state start timestamp
        end_time: Steady state end timestamp

    Returns:
        Dict with 'timestamps' and 'values' (relative to start_time, summed across devices)
    """
    all_points = []
    for device_id, points in device_data.items():
        for timestamp, value in points:
            if start_time <= timestamp <= end_time:
                all_points.append((timestamp, value))

    if not all_points:
        return {"timestamps": [], "values": []}

    all_points.sort(key=lambda x: x[0])

    timestamps = [t - start_time for t, v in all_points]
    values = [v for t, v in all_points]

    return {"timestamps": timestamps, "values": values}


def parse_prometheus_gauge(metrics_text: str, metric_name: str) -> Optional[float]:
    """Parse single gauge value from Prometheus text.

    Returns the first matching value, or None if not found.
    """
    pattern = rf"^{re.escape(metric_name)}\{{[^}}]*\}}\s+([\d.eE+-]+)"
    match = re.search(pattern, metrics_text, re.MULTILINE)

    if match:
        return float(match.group(1))
    return None


def extract_batch_size_timeline(
    prometheus: Dict,
    start_time: float,
    end_time: float,
) -> Dict[str, List]:
    """Extract batch size timeline from Prometheus metrics.

    Batch size is vllm:num_requests_running gauge metric.

    Args:
        prometheus: Prometheus JSON data
        start_time: Steady state start
        end_time: Steady state end

    Returns:
        Timeline dict with timestamps and values
    """
    timeline = prometheus.get("timeline", [])

    timestamps = []
    values = []

    for snapshot in timeline:
        timestamp = snapshot["timestamp"]
        if not (start_time <= timestamp <= end_time):
            continue

        metrics_text = snapshot["metrics"]

        batch_size = parse_prometheus_gauge(metrics_text, "vllm:num_requests_running")

        if batch_size is not None:
            timestamps.append(timestamp - start_time)
            values.append(batch_size)

    return {"timestamps": timestamps, "values": values}


def extract_timelines(run: BenchmarkRun) -> Dict[str, Dict]:
    """Extract all timelines from results.

    Args:
        run: BenchmarkRun object

    Returns:
        Dict with 'power_instant', 'power_average', 'temperature', 'batch_size'
        Each timeline is a dict with 'timestamps' and 'values' (compressed with RDP)
    """
    results = load_json_cached(run.results_path)
    prometheus = load_json_cached(run.prometheus_path)

    # Validate timeline structure exists
    if "timeline" not in results:
        raise ValueError(
            f"Missing 'timeline' in results.json for {run.results_path}\n"
            f"This indicates incomplete benchmark data. Re-run this benchmark."
        )

    timeline_data = results["timeline"]

    # Validate steady state timestamps exist
    if "steady_state_start_time" not in timeline_data:
        raise ValueError(
            f"Missing 'steady_state_start_time' in timeline for {run.results_path}\n"
            f"This indicates incomplete benchmark data. Re-run this benchmark."
        )
    if "steady_state_end_time" not in timeline_data:
        raise ValueError(
            f"Missing 'steady_state_end_time' in timeline for {run.results_path}\n"
            f"This indicates incomplete benchmark data. Re-run this benchmark."
        )

    steady_start = timeline_data["steady_state_start_time"]
    steady_end = timeline_data["steady_state_end_time"]

    timelines = {}

    # Validate power data exists (critical for energy measurements)
    if "power" not in timeline_data:
        raise ValueError(
            f"Missing 'power' data in timeline for {run.results_path}\n"
            f"This indicates Zeus monitoring failed. Re-run this benchmark."
        )

    power_data = timeline_data["power"]

    # At least one of device_instant or device_average must exist
    if "device_instant" not in power_data and "device_average" not in power_data:
        raise ValueError(
            f"Missing both 'device_instant' and 'device_average' in power timeline\n"
            f"File: {run.results_path}\n"
            f"This indicates Zeus monitoring failed. Re-run this benchmark."
        )

    if "device_instant" in power_data:
        device_instant = power_data["device_instant"]
        aggregated = aggregate_device_timelines(device_instant, steady_start, steady_end)
        if not aggregated["timestamps"]:
            raise ValueError(
                f"Empty power_instant timeline after filtering to steady state\n"
                f"File: {run.results_path}\n"
                f"Steady state: {steady_start} to {steady_end}\n"
                f"This indicates timing/monitoring issues. Re-run this benchmark."
            )
        timelines["power_instant"] = compress_timeline(aggregated, epsilon=0.5)

    if "device_average" in power_data:
        device_average = power_data["device_average"]
        aggregated = aggregate_device_timelines(device_average, steady_start, steady_end)
        if not aggregated["timestamps"]:
            raise ValueError(
                f"Empty power_average timeline after filtering to steady state\n"
                f"File: {run.results_path}\n"
                f"Steady state: {steady_start} to {steady_end}\n"
                f"This indicates timing/monitoring issues. Re-run this benchmark."
            )
        timelines["power_average"] = compress_timeline(aggregated, epsilon=0.5)

    # Validate batch size timeline exists (critical for understanding behavior)
    batch_size_timeline = extract_batch_size_timeline(prometheus, steady_start, steady_end)
    if not batch_size_timeline or not batch_size_timeline["timestamps"]:
        raise ValueError(
            f"Empty or missing batch_size timeline from Prometheus\n"
            f"File: {run.prometheus_path}\n"
            f"This indicates vLLM metrics were not collected. Re-run this benchmark."
        )
    timelines["batch_size"] = compress_timeline(batch_size_timeline, epsilon=1.0)

    return timelines


# ============================================================================
# Parallelization Config Parsing
# ============================================================================


def parse_parallelization(run: BenchmarkRun) -> Dict:
    """Parse parallelization configuration from monolithic.config.yaml.

    Args:
        run: BenchmarkRun object with config_dir path

    Returns:
        Dict with tensor_parallel, expert_parallel, data_parallel, and notes
    """
    config_path = run.config_dir / "monolithic.config.yaml"

    if not config_path.exists():
        print(f"Warning: Config not found: {config_path}")
        return {
            "tensor_parallel": 1,
            "expert_parallel": 0,
            "data_parallel": 0,
            "notes": "Config file not found",
        }

    with open(config_path) as f:
        config = yaml.safe_load(f)

    result = {
        "tensor_parallel": 1,
        "expert_parallel": 0,
        "data_parallel": 0,
        "notes": "",
    }

    if "tensor-parallel-size" in config:
        result["tensor_parallel"] = config["tensor-parallel-size"]

    if config.get("enable-expert-parallel"):
        if "expert-parallel-size" in config:
            result["expert_parallel"] = config["expert-parallel-size"]
        else:
            result["expert_parallel"] = run.num_gpus

    if "data-parallel-size" in config:
        result["data_parallel"] = config["data-parallel-size"]

    if "Qwen3-Coder-480B" in run.model_id:
        result["notes"] = "EP for experts, DP for attention layers"

    return result


def infer_weight_precision(model_id: str, model_info: dict) -> str:
    """Infer weight precision from model name or metadata.

    Logic:
    - If specified in model_info, use that
    - If "FP8" in model name, use "fp8"
    - If DeepSeek-R1 or DeepSeek-V3, use "fp8" (most params are FP8)
    - Otherwise, default to "bfloat16"

    Args:
        model_id: Model identifier like "Qwen/Qwen3-8B"
        model_info: Loaded model_info dict

    Returns:
        Weight precision string: "fp8", "bfloat16", "float16", etc.
    """
    if "weight_precision" in model_info:
        return model_info["weight_precision"]

    if "FP8" in model_id:
        return "fp8"

    if "DeepSeek-R1" in model_id or "DeepSeek-V3" in model_id:
        return "fp8"

    return "bfloat16"


def get_model_params(model_id: str, task: str) -> Dict[str, float]:
    """Get total and activated parameters for a model from model_info.json.

    Args:
        model_id: Model identifier like "Qwen/Qwen3-8B"
        task: Task name like "lm-arena-chat"

    Returns:
        Dict with total_params_billions, activated_params_billions, is_moe, and weight_precision
    """
    model_info_path = Path("configs/vllm") / task / model_id / "model_info.json"

    if model_info_path.exists():
        try:
            with open(model_info_path) as f:
                model_info = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Malformed JSON in model_info.json\n"
                f"File: {model_info_path}\n"
                f"Error: {e}\n"
                f"Fix the model_info.json file."
            )

        # Validate required fields in model_info.json
        if "total_parameters_billion" not in model_info:
            raise ValueError(
                f"Missing 'total_parameters_billion' in model_info.json\n"
                f"File: {model_info_path}\n"
                f"This field is required. Add it to the model_info.json file."
            )

        total_params = model_info["total_parameters_billion"]

        # active_parameters_billion is optional - defaults to total if not MoE
        active_params = model_info.get("active_parameters_billion", total_params)
        weight_precision = infer_weight_precision(model_id, model_info)

        return {
            "total_params_billions": float(total_params),
            "activated_params_billions": float(active_params),
            "is_moe": total_params != active_params,
            "weight_precision": weight_precision,
        }

    match = re.search(r"(\d+)B", model_id)
    if match:
        params = float(match.group(1))
        weight_precision = infer_weight_precision(model_id, {})
        print(f"Warning: model_info.json not found for {model_id}, inferred {params}B from name")
        return {
            "total_params_billions": params,
            "activated_params_billions": params,
            "is_moe": False,
            "weight_precision": weight_precision,
        }

    weight_precision = infer_weight_precision(model_id, {})
    print(f"Warning: Could not determine parameters for {model_id}")
    return {
        "total_params_billions": 0,
        "activated_params_billions": 0,
        "is_moe": False,
        "weight_precision": weight_precision,
    }


# ============================================================================
# Distribution Aggregation
# ============================================================================


def aggregate_output_length_distributions(runs: List[BenchmarkRun]) -> Dict:
    """Aggregate output length across all runs."""
    all_lengths = []

    for run in runs:
        results = load_json_cached(run.results_path)

        # Validate results structure
        if "results" not in results:
            raise ValueError(
                f"Missing 'results' field in results.json\n"
                f"File: {run.results_path}\n"
                f"This indicates incomplete benchmark data. Re-run this benchmark."
            )

        for req in results["results"]:
            # Validate required fields in each request result
            if "success" not in req:
                raise ValueError(
                    f"Missing 'success' field in request result\n"
                    f"File: {run.results_path}\n"
                    f"This indicates malformed results. Re-run this benchmark."
                )
            if req["success"]:
                if "output_len" not in req:
                    raise ValueError(
                        f"Missing 'output_len' in successful request result\n"
                        f"File: {run.results_path}\n"
                        f"This indicates incomplete results. Re-run this benchmark."
                    )
                all_lengths.append(req["output_len"])

    if not all_lengths:
        raise ValueError(
            f"No successful requests found across all runs\n"
            f"This indicates all benchmarks failed. Re-run benchmarks."
        )

    counts, bin_edges = np.histogram(all_lengths, bins=50)

    return {"bins": bin_edges.tolist(), "counts": counts.tolist()}


def is_unstable_run(
    run: BenchmarkRun,
    min_steady_duration: float = 10.0,
    batch_utilization_threshold: float = 0.9,
) -> tuple[bool, str]:
    """Check if a benchmark run is unstable (OOM or server issues).

    A run is considered unstable if:
    1. Steady state duration is too short (< min_steady_duration seconds), OR
    2. Batch size at end of steady state is < batch_utilization_threshold * max_num_seqs

    Args:
        run: BenchmarkRun to check
        min_steady_duration: Minimum acceptable steady state duration in seconds
        batch_utilization_threshold: Minimum ratio of actual batch size to max_num_seqs

    Returns:
        (is_unstable, reason) tuple
    """
    try:
        results = load_json_cached(run.results_path)

        # Check steady state duration
        steady_duration = results.get("steady_state_duration")
        if steady_duration is None or steady_duration < min_steady_duration:
            return (True, f"Steady state duration too short: {steady_duration}s < {min_steady_duration}s")

        # Check batch utilization if max_num_seqs is configured
        if run.max_num_seqs is not None:
            prometheus = load_json_cached(run.prometheus_path)
            prom_stats = prometheus.get("steady_state_stats", {})
            avg_batch_size = prom_stats.get("vllm:num_requests_running")

            if avg_batch_size is not None:
                utilization = avg_batch_size / run.max_num_seqs
                if utilization < batch_utilization_threshold:
                    return (True, f"Low batch utilization: {avg_batch_size}/{run.max_num_seqs} = {utilization:.1%} < {batch_utilization_threshold:.0%}")

        return (False, "")

    except Exception as e:
        # If we can't load the files, consider it unstable
        return (True, f"Failed to load results: {e}")


def filter_unstable_runs(runs: List[BenchmarkRun]) -> List[BenchmarkRun]:
    """Filter out unstable runs and cascade exclusions.

    When a run with max_num_seqs=X is unstable, also exclude all runs
    with max_num_seqs > X for the same model/task/GPU combination.

    Args:
        runs: List of all benchmark runs

    Returns:
        Filtered list with unstable runs removed
    """
    # Group by (model_id, task, gpu_model, num_gpus)
    grouped = defaultdict(list)
    for run in runs:
        key = (run.model_id, run.task, run.gpu_model, run.num_gpus)
        grouped[key].append(run)

    filtered_runs = []
    total_excluded = 0

    for key, group_runs in grouped.items():
        model_id, task, gpu_model, num_gpus = key

        # Find the minimum unstable max_num_seqs
        min_unstable_seqs = float('inf')
        unstable_reasons = []

        for run in group_runs:
            is_unstable, reason = is_unstable_run(run)
            if is_unstable:
                unstable_reasons.append((run, reason))
                if run.max_num_seqs is not None:
                    min_unstable_seqs = min(min_unstable_seqs, run.max_num_seqs)

        # Filter runs
        for run in group_runs:
            # Exclude if this run is unstable
            is_unstable, reason = is_unstable_run(run)
            if is_unstable:
                print(f"  Excluding unstable run: {model_id}/{task}/{gpu_model} max_num_seqs={run.max_num_seqs}")
                print(f"    Reason: {reason}", flush=True)
                total_excluded += 1
                continue

            # Exclude if max_num_seqs >= minimum unstable value
            if run.max_num_seqs is not None and run.max_num_seqs >= min_unstable_seqs:
                print(f"  Excluding run due to cascade: {model_id}/{task}/{gpu_model} max_num_seqs={run.max_num_seqs}")
                print(f"    (>= unstable threshold {min_unstable_seqs})", flush=True)
                total_excluded += 1
                continue

            filtered_runs.append(run)

    print(f"\nFiltered out {total_excluded} unstable runs", flush=True)
    print(f"Remaining: {len(filtered_runs)} valid runs", flush=True)

    return filtered_runs


def group_runs_by_model_task(runs: List[BenchmarkRun]) -> Dict[tuple, List[BenchmarkRun]]:
    """Group benchmark runs by (model_id, task).

    Returns:
        Dict mapping (model_id, task) to list of runs
    """
    grouped = defaultdict(list)

    for run in runs:
        key = (run.model_id, run.task)
        grouped[key].append(run)

    return dict(grouped)


# ============================================================================
# JSON Generation
# ============================================================================


def generate_index_json(runs: List[BenchmarkRun], output_dir: Path) -> None:
    """Generate index.json with task list and model metadata."""
    from datetime import datetime

    tasks = sorted(set(run.task for run in runs))

    models = {}
    for run in runs:
        if run.model_id not in models:
            params = get_model_params(run.model_id, run.task)
            nickname = run.model_id.split("/")[-1]
            models[run.model_id] = {
                "nickname": nickname,
                "total_params_billions": params["total_params_billions"],
                "activated_params_billions": params["activated_params_billions"],
                "is_moe": params["is_moe"],
                "weight_precision": params["weight_precision"],
            }

    index_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "tasks": tasks,
        "models": models,
    }

    output_path = output_dir / "index.json"
    with open(output_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Generated {output_path}")


def generate_task_json(task: str, runs: List[BenchmarkRun], output_dir: Path) -> None:
    """Generate task-specific JSON with all configurations."""
    task_runs = [run for run in runs if run.task == task]

    if not task_runs:
        return

    task_display_names = {
        "lm-arena-chat": "LLM Chat (LM Arena)",
        "gpqa": "GPQA Diamond",
        "sourcegraph-fim": "Fill-in-the-Middle (Sourcegraph)",
        "image-chat": "Image Chat",
        "video-chat": "Video Chat",
        "audio-chat": "Audio Chat",
    }

    configurations = []
    for run in task_runs:
        metrics = extract_metrics(run)
        params = get_model_params(run.model_id, task)

        config = {
            "model_id": run.model_id,
            "nickname": run.model_id.split("/")[-1],
            "gpu_model": run.gpu_model,
            "num_gpus": run.num_gpus,
            "total_params_billions": params["total_params_billions"],
            "activated_params_billions": params["activated_params_billions"],
            "max_num_seqs": run.max_num_seqs,
            "max_num_batched_tokens": run.max_num_batched_tokens,
            "energy_per_token_joules": metrics["energy_per_token_joules"],
            "energy_per_request_joules": metrics["energy_per_request_joules"],
            "median_itl_ms": metrics["median_itl_ms"],
            "output_throughput_tokens_per_sec": metrics["output_throughput_tokens_per_sec"],
            # These fields are now guaranteed to exist due to strict validation in extract_metrics()
            "p95_itl_ms": metrics["p95_itl_ms"],
            "avg_batch_size": metrics["avg_batch_size"],
        }

        configurations.append(config)

    task_data = {
        "task": task,
        "task_display_name": task_display_names.get(task, task),
        "configurations": configurations,
    }

    output_path = output_dir / "tasks" / f"{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(task_data, f, indent=2)

    print(f"Generated {output_path}")


def generate_model_json(
    model_id: str,
    task: str,
    runs: List[BenchmarkRun],
    output_dir: Path,
) -> None:
    """Generate model detail JSON with aggregated distributions and all configs."""
    params = get_model_params(model_id, task)

    output_length_dist = aggregate_output_length_distributions(runs)

    configurations = []
    for run in runs:
        metrics = extract_metrics(run)
        timelines = extract_timelines(run)
        parallelization = parse_parallelization(run)

        config = {
            "gpu_model": run.gpu_model,
            "num_gpus": run.num_gpus,
            "max_num_seqs": run.max_num_seqs,
            "max_num_batched_tokens": run.max_num_batched_tokens,
            "parallelization": parallelization,
            "energy_per_token_joules": metrics["energy_per_token_joules"],
            "energy_per_request_joules": metrics["energy_per_request_joules"],
            "median_itl_ms": metrics["median_itl_ms"],
            # These fields are now guaranteed to exist due to strict validation in extract_metrics()
            "p95_itl_ms": metrics["p95_itl_ms"],
            "output_throughput_tokens_per_sec": metrics["output_throughput_tokens_per_sec"],
            "avg_batch_size": metrics["avg_batch_size"],
            "timelines": timelines,
        }

        configurations.append(config)

    model_data = {
        "model_id": model_id,
        "task": task,
        "total_params_billions": params["total_params_billions"],
        "activated_params_billions": params["activated_params_billions"],
        "is_moe": params["is_moe"],
        "weight_precision": params["weight_precision"],
        "output_length_distribution": output_length_dist,
        "configurations": configurations,
    }

    output_path = output_dir / "models" / f"{model_id.replace('/', '__')}__{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"Generated {output_path}")


def validate_size(output_dir: Path, max_size_mb: int = 900) -> None:
    """Validate total size of generated data."""
    total_size = 0
    for file_path in output_dir.rglob("*.json"):
        total_size += file_path.stat().st_size

    size_mb = total_size / (1024 * 1024)
    print(f"\nTotal data size: {size_mb:.2f} MB")

    if size_mb > max_size_mb:
        print(f"ERROR: Data size exceeds {max_size_mb} MB limit!")
        sys.exit(1)

    print(f"Size check passed (under {max_size_mb} MB limit)")


# ============================================================================
# Main Function
# ============================================================================


def _generate_model_json_worker(args):
    """Wrapper for generate_model_json to use with multiprocessing.Pool."""
    model_id, task, runs, output_dir, idx, total = args
    print(f"  [{idx}/{total}] Generating {model_id} / {task}...", flush=True)
    generate_model_json(model_id, task, runs, output_dir)
    return model_id, task


def main():
    parser = argparse.ArgumentParser(
        description="Build leaderboard data from benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        action="append",
        dest="results_dirs",
        help="Path to results directory (can be specified multiple times for incremental aggregation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="leaderboard/public/data",
        help="Path to output directory",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=900,
        help="Maximum allowed size in MB",
    )

    args = parser.parse_args()

    if not args.results_dirs:
        print("Error: At least one --results-dir must be specified")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {len(args.results_dirs)} results directories...", flush=True)
    all_runs = []
    for results_dir in args.results_dirs:
        print(f"  Scanning {results_dir}...", flush=True)
        runs = scan_results_directory(results_dir)
        all_runs.extend(runs)
        print(f"    Found {len(runs)} runs", flush=True)

    runs = all_runs
    print(f"Total: {len(runs)} benchmark runs from all directories", flush=True)

    if not runs:
        print("No benchmark runs found!")
        sys.exit(1)

    # Filter out unstable runs (OOM, server crashes, etc.)
    print("\nFiltering unstable runs...", flush=True)
    runs = filter_unstable_runs(runs)

    if not runs:
        print("No valid runs remaining after filtering!")
        sys.exit(1)

    print("\nGenerating index.json...", flush=True)
    generate_index_json(runs, output_dir)

    print("\nGenerating task JSONs...", flush=True)
    tasks = sorted(set(run.task for run in runs))
    for task in tasks:
        print(f"  Generating {task}...", flush=True)
        generate_task_json(task, runs, output_dir)

    print("\nGenerating model detail JSONs...")
    grouped = group_runs_by_model_task(runs)
    total_models = len(grouped)

    # Prepare arguments for multiprocessing
    pool_args = [
        (model_id, task, model_runs, output_dir, idx, total_models)
        for idx, ((model_id, task), model_runs) in enumerate(grouped.items(), 1)
    ]

    # Use multiprocessing.Pool to generate model files in parallel
    num_workers = min(multiprocessing.cpu_count(), total_models)
    print(f"Using {num_workers} workers for parallel processing...", flush=True)

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(_generate_model_json_worker, pool_args)

    print("\nValidating data size...", flush=True)
    validate_size(output_dir, args.max_size_mb)

    print("\n✅ Data generation complete!", flush=True)


if __name__ == "__main__":
    main()
