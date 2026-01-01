"""Build leaderboard data from benchmark results.

This script scans the results directory, extracts metrics, compresses timelines,
and generates optimized JSON files for the leaderboard website.
"""

import argparse
import json
import multiprocessing
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml


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
    config_path: Path


@dataclass
class DiffusionBenchmarkRun:
    """Metadata for a single diffusion model benchmark run."""

    model_id: str
    gpu_model: str
    num_gpus: int
    task: str  # "text-to-image" or "text-to-video"
    batch_size: int
    height: int
    width: int
    inference_steps: int
    ulysses_degree: int
    ring_degree: int
    use_torch_compile: bool
    seed: int
    results_path: Path
    # Video-specific fields (None for text-to-image)
    num_frames: Optional[int] = None
    fps: Optional[int] = None


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


def scan_results_directory(results_dir: str, config_dir: Path) -> List[BenchmarkRun]:
    """Scan results directory and collect metadata for all runs.

    Args:
        results_dir: Path to results directory (e.g., "results/")
        config_dir: Path to config directory (e.g., Path("configs/vllm"))

    Returns:
        List of BenchmarkRun objects
    """
    results_root = Path(results_dir)
    runs = []

    for results_json_path in results_root.rglob("results.json"):
        parts = results_json_path.relative_to(results_root).parts

        # Skip diffusion results - those are handled by scan_diffusion_results_directory
        if parts[0] == "diffusion":
            continue

        if parts[0] in ("llm", "mllm"):
            if len(parts) < 8:
                continue
            task = parts[1]
            org = parts[3]
            model_name = parts[4]
            gpu_model = parts[5]
            params_dir = parts[6]
            config_path = config_dir / task / f"{org}/{model_name}" / gpu_model / "monolithic.config.yaml"
        else:
            if len(parts) < 5:
                continue
            org = parts[0]
            model_name = parts[1]
            gpu_model = parts[2]
            params_dir = parts[3]

            task = None
            config_path = None
            for task_candidate in [
                "lm-arena-chat",
                "gpqa",
                "sourcegraph-fim",
                "image-chat",
                "video-chat",
                "audio-chat",
            ]:
                config_path = (
                    config_dir / task_candidate / f"{org}/{model_name}" / gpu_model / "monolithic.config.yaml"
                )
                if config_path.exists():
                    task = task_candidate
                    break

            if task is None or config_path is None:
                print(f"Warning: Could not determine task for {org}/{model_name} on {gpu_model}")
                continue

        model_id = f"{org}/{model_name}"
        params = parse_dir_params(params_dir)

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
            config_path=config_path,
        )

        runs.append(run)

    print(f"Found {len(runs)} benchmark runs")
    return runs


def parse_diffusion_config_dir(dirname: str) -> dict:
    """Parse parameters from diffusion config directory name.

    Directory names follow pattern: batch-{B}+size-{H}x{W}+steps-{S}+seed-{SEED}+uly-{U}+ring-{R}+tc-{TC}

    Args:
        dirname: Directory name like "batch-1+size-1024x1024+steps-20+seed-48105+uly-1+ring-1+tc-False"

    Returns:
        Dict with parsed parameters
    """
    params = {}
    parts = dirname.split("+")

    for part in parts:
        if part.startswith("batch-"):
            params["batch_size"] = int(part[6:])
        elif part.startswith("size-"):
            size_str = part[5:]
            h, w = size_str.split("x")
            params["height"] = int(h)
            params["width"] = int(w)
        elif part.startswith("steps-"):
            params["inference_steps"] = int(part[6:])
        elif part.startswith("seed-"):
            params["seed"] = int(part[5:])
        elif part.startswith("uly-"):
            params["ulysses_degree"] = int(part[4:])
        elif part.startswith("ring-"):
            params["ring_degree"] = int(part[5:])
        elif part.startswith("tc-"):
            params["use_torch_compile"] = part[3:].lower() == "true"

    return params


def scan_diffusion_results_directory(results_dir: str) -> List[DiffusionBenchmarkRun]:
    """Scan diffusion results directory and collect metadata for all runs.

    Args:
        results_dir: Path to results directory containing diffusion results

    Returns:
        List of DiffusionBenchmarkRun objects
    """
    results_root = Path(results_dir)
    runs = []

    for results_json_path in results_root.rglob("results.json"):
        parts = results_json_path.relative_to(results_root).parts

        # Check if this is a diffusion result: diffusion/{task}/{org}/{model}/{config}/results.json
        if len(parts) < 6 or parts[0] != "diffusion":
            continue

        task = parts[1]  # "text-to-image" or "text-to-video"
        if task not in ("text-to-image", "text-to-video"):
            continue

        org = parts[2]
        model_name = parts[3]
        config_dir = parts[4]

        model_id = f"{org}/{model_name}"
        config_params = parse_diffusion_config_dir(config_dir)

        # Validate required diffusion config parameters
        required_diffusion_params = ["ulysses_degree", "ring_degree", "batch_size", "height", "width", "inference_steps", "seed"]
        for param in required_diffusion_params:
            if param not in config_params:
                raise ValueError(
                    f"Missing required parameter '{param}' in diffusion config directory name\n"
                    f"Directory: {config_dir}\n"
                    f"Expected format: batch-{{B}}+size-{{H}}x{{W}}+steps-{{S}}+seed-{{SEED}}+uly-{{U}}+ring-{{R}}+tc-{{TC}}"
                )

        # Derive num_gpus from parallelism degrees
        ulysses_degree = config_params["ulysses_degree"]
        ring_degree = config_params["ring_degree"]
        num_gpus = ulysses_degree * ring_degree

        # Derive GPU model from results path (e.g., .../h100/run/...)
        # Look for gpu model in parent directories
        gpu_model = None
        for parent in results_json_path.parents:
            if parent.name.lower() in ("h100", "b200", "a100"):
                gpu_model = parent.name.upper()
                break

        if gpu_model is None:
            raise ValueError(
                f"Could not determine GPU model from results path\n"
                f"Path: {results_json_path}\n"
                f"Expected one of the parent directories to be 'h100', 'b200', or 'a100'"
            )

        # For video tasks, read num_frames and fps from results.json
        num_frames = None
        fps = None
        if task == "text-to-video":
            results_data = load_json_cached(results_json_path)
            num_frames = results_data["num_frames"]
            fps = results_data["fps"]

        run = DiffusionBenchmarkRun(
            model_id=model_id,
            gpu_model=gpu_model,
            num_gpus=num_gpus,
            task=task,
            batch_size=config_params["batch_size"],
            height=config_params["height"],
            width=config_params["width"],
            inference_steps=config_params["inference_steps"],
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            use_torch_compile=config_params.get("use_torch_compile", False),
            seed=config_params["seed"],
            results_path=results_json_path,
            num_frames=num_frames,
            fps=fps,
        )

        runs.append(run)

    print(f"Found {len(runs)} diffusion benchmark runs")
    return runs


# ============================================================================
# Metrics Extraction
# ============================================================================


def smooth_chunked_itl(itl_list: List[float]) -> List[float]:
    """Smooth chunked ITL values by spreading latency across chunk tokens.

    When tokens arrive in chunks due to CPU contention, the benchmark records
    the full latency for the first token and zeros for the rest. This function
    spreads that latency evenly across all tokens in each chunk.

    A chunk is defined as: non-zero value followed by consecutive zeros.
    Properly streamed tokens (all non-zero) are left unchanged.

    Args:
        itl_list: List of inter-token latencies in seconds

    Returns:
        Smoothed ITL list with chunk latencies spread evenly
    """
    if not itl_list:
        return []

    # Step 1: Filter leading zeros
    first_nonzero_idx = 0
    for i, val in enumerate(itl_list):
        if val > 0:
            first_nonzero_idx = i
            break
    else:
        return []  # All zeros

    filtered = itl_list[first_nonzero_idx:]

    # Step 2: Smooth each chunk
    result = []
    i = 0
    while i < len(filtered):
        if filtered[i] <= 0:
            # Should not happen after filtering, but handle gracefully
            result.append(filtered[i])
            i += 1
            continue

        # Found a non-zero value - find the extent of this chunk
        chunk_latency = filtered[i]
        j = i + 1
        while j < len(filtered) and filtered[j] == 0:
            j += 1

        chunk_size = j - i

        if chunk_size == 1:
            # Single token, not a chunk - keep as-is
            result.append(chunk_latency)
        else:
            # Spread latency across chunk
            avg_latency = chunk_latency / chunk_size
            result.extend([avg_latency] * chunk_size)

        i = j

    return result


def extract_client_itl_percentiles(results: Dict) -> Dict[str, float]:
    """Extract ITL percentiles from client-side measurements.

    Client-side ITL is more accurate than Prometheus histogram buckets
    which have coarse granularity (~50ms minimum bucket).

    For each request, we smooth chunked ITL values by spreading the latency
    of each chunk (non-zero followed by zeros) evenly across all tokens in
    that chunk. This handles CPU contention causing tokens to arrive in batches.

    Args:
        results: Loaded results.json dict

    Returns:
        Dict with p50_itl_ms, p90_itl_ms, p95_itl_ms, p99_itl_ms
    """
    all_itl_values = []

    for req in results.get("results", []):
        if not req.get("success", False):
            continue

        itl_list = req.get("itl", [])
        if not itl_list:
            continue

        filtered_itl = smooth_chunked_itl(itl_list)
        all_itl_values.extend(filtered_itl)

    if not all_itl_values:
        raise ValueError("No valid ITL values found in results")

    itl_array = np.array(all_itl_values)

    return {
        "p50_itl_ms": float(np.percentile(itl_array, 50)) * 1000,
        "p90_itl_ms": float(np.percentile(itl_array, 90)) * 1000,
        "p95_itl_ms": float(np.percentile(itl_array, 95)) * 1000,
        "p99_itl_ms": float(np.percentile(itl_array, 99)) * 1000,
    }


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

    steady_state_tokens = results["steady_state_energy"] / results["steady_state_energy_per_token"]
    steady_state_throughput = steady_state_tokens / results["steady_state_duration"]

    metrics = {
        "energy_per_token_joules": results["steady_state_energy_per_token"],
        "total_energy_joules": results["steady_state_energy"],
        "output_throughput_tokens_per_sec": steady_state_throughput,
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

    # Calculate average power during steady state
    metrics["avg_power_watts"] = results["steady_state_energy"] / results["steady_state_duration"]

    # Extract ITL percentiles from client-side measurements (more accurate than Prometheus)
    itl_metrics = extract_client_itl_percentiles(results)
    metrics.update(itl_metrics)
    metrics["median_itl_ms"] = metrics["p50_itl_ms"]

    # Validate Prometheus data exists (needed for batch size metric)
    if "steady_state_stats" not in prometheus:
        raise ValueError(
            f"Missing 'steady_state_stats' in prometheus.json for {run.results_path}\n"
            f"This indicates incomplete benchmark data. Re-run this benchmark."
        )

    prom_stats = prometheus["steady_state_stats"]

    # Validate batch size metric exists (critical for detecting unstable runs)
    if "vllm:num_requests_running" not in prom_stats:
        raise ValueError(
            f"Missing required metric 'vllm:num_requests_running' in prometheus\n"
            f"File: {run.prometheus_path}\n"
            f"This indicates incomplete vLLM metrics. Re-run this benchmark."
        )
    metrics["avg_batch_size"] = prom_stats["vllm:num_requests_running"]

    return metrics


def extract_diffusion_metrics(run: DiffusionBenchmarkRun) -> Dict:
    """Extract all metrics from a diffusion benchmark run.

    Args:
        run: DiffusionBenchmarkRun object with path to results file

    Returns:
        Dict with all extracted metrics
    """
    results = load_json_cached(run.results_path)

    # Validate required fields exist
    if "iteration_energy_measurements" not in results:
        raise ValueError(f"Missing 'iteration_energy_measurements' in {run.results_path}")

    iterations = results["iteration_energy_measurements"]
    if not iterations:
        raise ValueError(f"Empty 'iteration_energy_measurements' in {run.results_path}")

    # Compute average metrics across iterations
    # Sum GPU energy across all GPUs for each iteration
    gpu_energies = []
    latencies = []
    for iter_data in iterations:
        # Sum energy from all GPUs
        total_gpu_energy = sum(iter_data["gpu_energy"].values())
        gpu_energies.append(total_gpu_energy)
        latencies.append(iter_data["time"])

    avg_gpu_energy = float(np.mean(gpu_energies))
    avg_batch_latency = float(np.mean(latencies))

    # Energy per output = total batch energy / batch size
    energy_per_output = avg_gpu_energy / run.batch_size

    # Throughput = batch_size / batch_latency
    throughput = run.batch_size / avg_batch_latency

    metrics = {
        "batch_latency_s": avg_batch_latency,
        "avg_gpu_energy_joules": avg_gpu_energy,
        "num_iterations": len(iterations),
        "avg_power_watts": avg_gpu_energy / avg_batch_latency,
    }

    # Task-specific metrics
    if run.task == "text-to-image":
        metrics["energy_per_image_joules"] = energy_per_output
        metrics["throughput_images_per_sec"] = throughput
    else:  # text-to-video
        metrics["energy_per_video_joules"] = energy_per_output
        metrics["throughput_videos_per_sec"] = throughput

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
# Parallelization Config Parsing
# ============================================================================


def parse_parallelization(run: BenchmarkRun) -> Dict:
    """Parse parallelization configuration from monolithic.config.yaml.

    Args:
        run: BenchmarkRun object with config_dir path

    Returns:
        Dict with tensor_parallel, expert_parallel, data_parallel, and notes
    """
    config_path = run.config_path

    if not config_path.exists():
        raise ValueError(
            f"Config file not found: {config_path}\n"
            f"Model: {run.model_id}\n"
            f"Task: {run.task}\n"
            f"GPU: {run.gpu_model}\n"
            f"Please create the config file."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    result = {
        "tensor_parallel": 1,
        "expert_parallel": 1,
        "data_parallel": 1,
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
        # 0 means auto-expand to num_gpus
        result["data_parallel"] = config["data-parallel-size"] or run.num_gpus

    # Add notes for hybrid DP+EP configurations
    if result["data_parallel"] > 1 and result["expert_parallel"] > 1:
        result["notes"] = "DP for attention, EP for MLP experts"

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


def get_diffusion_model_params(model_id: str, task: str, config_dir: Path) -> dict:
    """Get model parameters for a diffusion model from model_info.json.

    Args:
        model_id: Model identifier like "stabilityai/stable-diffusion-3.5-large"
        task: Task name like "text-to-image"
        config_dir: Configuration directory containing model_info.json files

    Returns:
        Dict with nickname, total_params_billions, activated_params_billions, weight_precision
    """
    model_info_path = config_dir / task / model_id / "model_info.json"

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

        # Validate required fields
        if "total_parameters_billion" not in model_info:
            raise ValueError(
                f"Missing 'total_parameters_billion' in model_info.json\n"
                f"File: {model_info_path}\n"
                f"This field is required. Add it to the model_info.json file."
            )

        total_params = model_info["total_parameters_billion"]
        # Diffusion models are dense, so activated = total
        active_params = model_info.get("active_parameters_billion", total_params)
        nickname = model_info.get("nickname", model_id.split("/")[-1])
        weight_precision = model_info.get("weight_precision", "bfloat16")

        return {
            "nickname": nickname,
            "total_params_billions": float(total_params),
            "activated_params_billions": float(active_params),
            "weight_precision": weight_precision,
        }

    # If model_info.json is not found, error out
    raise ValueError(
        f"model_info.json not found for {model_id} in task {task}\n"
        f"Looked in: {model_info_path}\n"
        f"Please create the model_info.json file with the following fields:\n"
        f"  - nickname: Display name for the model\n"
        f"  - total_parameters_billion: Total parameter count in billions\n"
        f"  - weight_precision: Weight precision (e.g., 'bfloat16')\n"
    )


def get_model_params(model_id: str, task: str, config_dir: Path) -> dict:
    """Get total and activated parameters for a model from model_info.json.

    Args:
        model_id: Model identifier like "Qwen/Qwen3-8B"
        task: Task name like "lm-arena-chat"
        config_dir: Base configuration directory (e.g., "configs/vllm")

    Returns:
        Dict with total_params_billions, activated_params_billions, architecture, and weight_precision
    """
    model_info_path = config_dir / task / model_id / "model_info.json"

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

        # Use nickname from model_info if available, otherwise derive from model_id
        nickname = model_info.get("nickname", model_id.split("/")[-1])

        # Determine architecture: use explicit field if available, otherwise derive from params
        if "architecture" in model_info:
            architecture = model_info["architecture"]
        else:
            architecture = "MoE" if total_params != active_params else "Dense"

        return {
            "nickname": nickname,
            "total_params_billions": float(total_params),
            "activated_params_billions": float(active_params),
            "architecture": architecture,
            "weight_precision": weight_precision,
        }

    # If model_info.json is not found, error out
    raise ValueError(
        f"model_info.json not found for {model_id} in task {task}\n"
        f"Looked in: {model_info_path}\n"
        f"Please create the model_info.json file with the following fields:\n"
        f"  - total_parameters_billion: Total parameter count in billions\n"
        f"  - active_parameters_billion: Active parameter count (optional, defaults to total)\n"
        f"  - weight_precision: Weight precision (optional, will be inferred if not provided)\n"
    )


# ============================================================================
# Distribution Aggregation
# ============================================================================


def get_output_lengths_from_run(run: BenchmarkRun) -> List[int]:
    """Extract output lengths from a single benchmark run."""
    results = load_json_cached(run.results_path)

    # Validate results structure
    if "results" not in results:
        raise ValueError(
            f"Missing 'results' field in results.json\n"
            f"File: {run.results_path}\n"
            f"This indicates incomplete benchmark data. Re-run this benchmark."
        )

    lengths = []
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
            lengths.append(req["output_len"])

    return lengths


def compute_output_length_distribution(
    lengths: List[int],
    bin_edges: np.ndarray,
) -> Dict:
    """Compute histogram of output lengths using fixed bin edges."""
    counts, _ = np.histogram(lengths, bins=bin_edges)
    return {"bins": bin_edges.tolist(), "counts": counts.tolist()}


def aggregate_output_length_distributions(runs: List[BenchmarkRun]) -> Dict:
    """Aggregate output length across all runs."""
    all_lengths = []

    for run in runs:
        all_lengths.extend(get_output_lengths_from_run(run))

    if not all_lengths:
        raise ValueError(
            f"No successful requests found across all runs\n"
            f"This indicates all benchmarks failed. Re-run benchmarks."
        )

    counts, bin_edges = np.histogram(all_lengths, bins=50)

    return {"bins": bin_edges.tolist(), "counts": counts.tolist()}


def is_unstable_run(
    run: BenchmarkRun,
    min_steady_duration: float = 20.0,
    batch_utilization_threshold: float = 0.85,
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


def _check_unstable_worker(run: BenchmarkRun) -> tuple[BenchmarkRun, bool, str]:
    """Worker function for parallel unstable run checking."""
    is_unstable, reason = is_unstable_run(run)
    return (run, is_unstable, reason)


def filter_unstable_runs(runs: List[BenchmarkRun]) -> List[BenchmarkRun]:
    """Filter out unstable runs and cascade exclusions.

    When a run with max_num_seqs=X is unstable, also exclude all runs
    with max_num_seqs > X for the same model/task/GPU combination.

    Args:
        runs: List of all benchmark runs

    Returns:
        Filtered list with unstable runs removed
    """
    # Check all runs in parallel and cache results
    print(f"  Checking {len(runs)} runs for stability...", flush=True)
    num_workers = min(multiprocessing.cpu_count(), len(runs))

    with multiprocessing.Pool(processes=num_workers) as pool:
        check_results = pool.map(_check_unstable_worker, runs)

    # Build lookup dict for stability results
    stability_cache: Dict[Path, tuple[bool, str]] = {}
    for run, is_unstable, reason in check_results:
        stability_cache[run.results_path] = (is_unstable, reason)

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

        for run in group_runs:
            is_unstable, reason = stability_cache[run.results_path]
            if is_unstable and run.max_num_seqs is not None:
                min_unstable_seqs = min(min_unstable_seqs, run.max_num_seqs)

        # Filter runs
        for run in group_runs:
            is_unstable, reason = stability_cache[run.results_path]

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


def generate_index_json(
    runs: List[BenchmarkRun],
    diffusion_runs: List[DiffusionBenchmarkRun],
    output_dir: Path,
    config_dir: Path,
    diffusion_config_dir: Optional[Path] = None,
) -> None:
    """Generate index.json with task list, architecture groupings, and model metadata."""
    from datetime import datetime

    # Collect all tasks
    llm_mllm_tasks = sorted(set(run.task for run in runs))
    diffusion_tasks = sorted(set(run.task for run in diffusion_runs))
    all_tasks = sorted(set(llm_mllm_tasks + diffusion_tasks))

    # Group tasks by architecture
    llm_tasks = [t for t in llm_mllm_tasks if t in ("gpqa", "lm-arena-chat", "sourcegraph-fim")]
    mllm_tasks = [t for t in llm_mllm_tasks if t in ("image-chat", "video-chat", "audio-chat")]

    architectures = {
        "llm": llm_tasks,
        "mllm": mllm_tasks,
        "diffusion": diffusion_tasks,
    }

    # Collect model metadata
    models = {}
    for run in runs:
        if run.model_id not in models:
            params = get_model_params(run.model_id, run.task, config_dir)
            models[run.model_id] = {
                "nickname": params["nickname"],
                "total_params_billions": params["total_params_billions"],
                "activated_params_billions": params["activated_params_billions"],
                "architecture": params["architecture"],
                "weight_precision": params["weight_precision"],
            }

    for run in diffusion_runs:
        if run.model_id not in models:
            params = get_diffusion_model_params(run.model_id, run.task, diffusion_config_dir)
            models[run.model_id] = {
                "nickname": params["nickname"],
                "total_params_billions": params["total_params_billions"],
                "activated_params_billions": params["activated_params_billions"],
                "weight_precision": params["weight_precision"],
            }

    index_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "tasks": all_tasks,
        "architectures": architectures,
        "models": models,
    }

    output_path = output_dir / "index.json"
    with open(output_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Generated {output_path}")


def generate_task_json(task: str, runs: List[BenchmarkRun], output_dir: Path, config_dir: Path) -> None:
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
        params = get_model_params(run.model_id, task, config_dir)
        parallelization = parse_parallelization(run)

        config = {
            "model_id": run.model_id,
            "nickname": params["nickname"],
            "gpu_model": run.gpu_model,
            "num_gpus": run.num_gpus,
            "total_params_billions": params["total_params_billions"],
            "activated_params_billions": params["activated_params_billions"],
            "architecture": params["architecture"],
            "weight_precision": params["weight_precision"],
            "max_num_seqs": run.max_num_seqs,
            "max_num_batched_tokens": run.max_num_batched_tokens,
            # Parallelization fields flattened for table display
            "tensor_parallel": parallelization["tensor_parallel"],
            "expert_parallel": parallelization["expert_parallel"],
            "data_parallel": parallelization["data_parallel"],
            "energy_per_token_joules": metrics["energy_per_token_joules"],
            "energy_per_request_joules": metrics["energy_per_request_joules"],
            "avg_output_len": metrics["avg_output_len"],
            "avg_power_watts": metrics["avg_power_watts"],
            "median_itl_ms": metrics["median_itl_ms"],
            "output_throughput_tokens_per_sec": metrics["output_throughput_tokens_per_sec"],
            # These fields are now guaranteed to exist due to strict validation in extract_metrics()
            "p90_itl_ms": metrics["p90_itl_ms"],
            "p95_itl_ms": metrics["p95_itl_ms"],
            "p99_itl_ms": metrics["p99_itl_ms"],
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


def generate_diffusion_task_json(
    task: str,
    runs: List[DiffusionBenchmarkRun],
    output_dir: Path,
    config_dir: Optional[Path] = None,
) -> None:
    """Generate task-specific JSON for diffusion tasks."""
    task_runs = [run for run in runs if run.task == task]

    if not task_runs:
        return

    task_display_names = {
        "text-to-image": "Text to Image",
        "text-to-video": "Text to Video",
    }

    configurations = []
    for run in task_runs:
        metrics = extract_diffusion_metrics(run)
        params = get_diffusion_model_params(run.model_id, task, config_dir)

        config = {
            "model_id": run.model_id,
            "nickname": params["nickname"],
            "gpu_model": run.gpu_model,
            "num_gpus": run.num_gpus,
            "total_params_billions": params["total_params_billions"],
            "activated_params_billions": params["activated_params_billions"],
            "weight_precision": params["weight_precision"],
            "batch_size": run.batch_size,
            "batch_latency_s": metrics["batch_latency_s"],
            "avg_power_watts": metrics["avg_power_watts"],
            "ulysses_degree": run.ulysses_degree,
            "ring_degree": run.ring_degree,
            "inference_steps": run.inference_steps,
        }

        # Add task-specific fields
        if task == "text-to-image":
            config["energy_per_image_joules"] = metrics["energy_per_image_joules"]
            config["throughput_images_per_sec"] = metrics["throughput_images_per_sec"]
            config["image_height"] = run.height
            config["image_width"] = run.width
        else:  # text-to-video
            config["energy_per_video_joules"] = metrics["energy_per_video_joules"]
            config["throughput_videos_per_sec"] = metrics["throughput_videos_per_sec"]
            config["video_height"] = run.height
            config["video_width"] = run.width
            config["num_frames"] = run.num_frames
            config["fps"] = run.fps

        configurations.append(config)

    task_data = {
        "task": task,
        "task_display_name": task_display_names.get(task, task),
        "architecture": "diffusion",
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
    config_dir: Path,
) -> Path:
    """Generate model detail JSON with aggregated distributions and all configs."""
    params = get_model_params(model_id, task, config_dir)

    # First pass: collect all output lengths to determine fixed bin edges
    all_lengths = []
    run_lengths: Dict[Path, List[int]] = {}
    for run in runs:
        lengths = get_output_lengths_from_run(run)
        run_lengths[run.results_path] = lengths
        all_lengths.extend(lengths)

    if not all_lengths:
        raise ValueError(
            f"No successful requests found across all runs for {model_id}\n"
            f"This indicates all benchmarks failed. Re-run benchmarks."
        )

    # Compute fixed bin edges from all data (50 bins)
    _, bin_edges = np.histogram(all_lengths, bins=50)

    # Aggregate distribution (for backwards compatibility)
    agg_counts, _ = np.histogram(all_lengths, bins=bin_edges)
    output_length_dist = {"bins": bin_edges.tolist(), "counts": agg_counts.tolist()}

    configurations = []
    for run in runs:
        metrics = extract_metrics(run)
        parallelization = parse_parallelization(run)

        # Per-config output length distribution using fixed bin edges
        config_lengths = run_lengths[run.results_path]
        config_output_dist = compute_output_length_distribution(config_lengths, bin_edges)

        config = {
            "gpu_model": run.gpu_model,
            "num_gpus": run.num_gpus,
            "max_num_seqs": run.max_num_seqs,
            "max_num_batched_tokens": run.max_num_batched_tokens,
            "parallelization": parallelization,
            "energy_per_token_joules": metrics["energy_per_token_joules"],
            "energy_per_request_joules": metrics["energy_per_request_joules"],
            "avg_output_len": metrics["avg_output_len"],
            "avg_power_watts": metrics["avg_power_watts"],
            "median_itl_ms": metrics["median_itl_ms"],
            "p90_itl_ms": metrics["p90_itl_ms"],
            "p95_itl_ms": metrics["p95_itl_ms"],
            "p99_itl_ms": metrics["p99_itl_ms"],
            "output_throughput_tokens_per_sec": metrics["output_throughput_tokens_per_sec"],
            "avg_batch_size": metrics["avg_batch_size"],
            "output_length_distribution": config_output_dist,
        }

        configurations.append(config)

    model_data = {
        "model_id": model_id,
        "task": task,
        "total_params_billions": params["total_params_billions"],
        "activated_params_billions": params["activated_params_billions"],
        "architecture": params["architecture"],
        "weight_precision": params["weight_precision"],
        "output_length_distribution": output_length_dist,
        "configurations": configurations,
    }

    output_path = output_dir / "models" / f"{model_id.replace('/', '__')}__{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model_data, f, indent=2)

    return output_path


def generate_diffusion_model_json(
    model_id: str,
    task: str,
    runs: List[DiffusionBenchmarkRun],
    output_dir: Path,
    config_dir: Optional[Path] = None,
) -> Path:
    """Generate model detail JSON for diffusion models."""
    params = get_diffusion_model_params(model_id, task, config_dir)

    configurations = []
    for run in runs:
        metrics = extract_diffusion_metrics(run)

        config = {
            "gpu_model": run.gpu_model,
            "num_gpus": run.num_gpus,
            "batch_size": run.batch_size,
            "parallelization": {
                "ulysses_degree": run.ulysses_degree,
                "ring_degree": run.ring_degree,
            },
            "batch_latency_s": metrics["batch_latency_s"],
            "avg_power_watts": metrics["avg_power_watts"],
            "inference_steps": run.inference_steps,
        }

        # Add task-specific fields
        if task == "text-to-image":
            config["energy_per_image_joules"] = metrics["energy_per_image_joules"]
            config["throughput_images_per_sec"] = metrics["throughput_images_per_sec"]
            config["image_height"] = run.height
            config["image_width"] = run.width
        else:  # text-to-video
            config["energy_per_video_joules"] = metrics["energy_per_video_joules"]
            config["throughput_videos_per_sec"] = metrics["throughput_videos_per_sec"]
            config["video_height"] = run.height
            config["video_width"] = run.width
            config["num_frames"] = run.num_frames
            config["fps"] = run.fps

        configurations.append(config)

    model_data = {
        "model_id": model_id,
        "task": task,
        "nickname": params["nickname"],
        "total_params_billions": params["total_params_billions"],
        "activated_params_billions": params["activated_params_billions"],
        "weight_precision": params["weight_precision"],
        "configurations": configurations,
    }

    output_path = output_dir / "models" / f"{model_id.replace('/', '__')}__{task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model_data, f, indent=2)

    return output_path


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
    model_id, task, runs, output_dir, config_dir, idx, total = args
    print(f"  [{idx}/{total}] Generating {model_id} / {task}...", flush=True)
    path = generate_model_json(model_id, task, runs, output_dir, config_dir)
    print(f"  [{idx}/{total}] Generated {path}", flush=True)
    return model_id, task


def _generate_diffusion_model_json_worker(args):
    """Wrapper for generate_diffusion_model_json to use with multiprocessing.Pool."""
    model_id, task, runs, output_dir, config_dir, idx, total = args
    print(f"  [{idx}/{total}] Generating {model_id} / {task}...", flush=True)
    path = generate_diffusion_model_json(model_id, task, runs, output_dir, config_dir)
    print(f"  [{idx}/{total}] Generated {path}", flush=True)
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
    parser.add_argument(
        "--llm-config-dir",
        type=str,
        default=None,
        help="Path to LLM/MLLM config directory containing model_info.json files",
    )
    parser.add_argument(
        "--diffusion-config-dir",
        type=str,
        default=None,
        help="Path to diffusion config directory containing model_info.json files",
    )

    args = parser.parse_args()

    if not args.results_dirs:
        print("Error: At least one --results-dir must be specified")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_config_dir = Path(args.llm_config_dir) if args.llm_config_dir else None
    diffusion_config_dir = Path(args.diffusion_config_dir) if args.diffusion_config_dir else None

    # Scan for LLM/MLLM runs
    print(f"Scanning {len(args.results_dirs)} results directories for LLM/MLLM...", flush=True)
    all_runs = []
    for results_dir in args.results_dirs:
        print(f"  Scanning {results_dir}...", flush=True)
        runs = scan_results_directory(results_dir, llm_config_dir)
        all_runs.extend(runs)
        print(f"    Found {len(runs)} LLM/MLLM runs", flush=True)

    runs = all_runs
    print(f"Total: {len(runs)} LLM/MLLM benchmark runs from all directories", flush=True)

    # Scan for diffusion runs
    print(f"\nScanning {len(args.results_dirs)} results directories for diffusion models...", flush=True)
    all_diffusion_runs = []
    for results_dir in args.results_dirs:
        print(f"  Scanning {results_dir}...", flush=True)
        diffusion_runs = scan_diffusion_results_directory(results_dir)
        all_diffusion_runs.extend(diffusion_runs)
        print(f"    Found {len(diffusion_runs)} diffusion runs", flush=True)

    diffusion_runs = all_diffusion_runs
    print(f"Total: {len(diffusion_runs)} diffusion benchmark runs from all directories", flush=True)

    # Check if we have any data
    if not runs and not diffusion_runs:
        print("No benchmark runs found!")
        sys.exit(1)

    # Filter out unstable LLM/MLLM runs (OOM, server crashes, etc.)
    if runs:
        print("\nFiltering unstable LLM/MLLM runs...", flush=True)
        runs = filter_unstable_runs(runs)

    # Generate index.json with both LLM/MLLM and diffusion data
    print("\nGenerating index.json...", flush=True)
    generate_index_json(runs, diffusion_runs, output_dir, llm_config_dir, diffusion_config_dir)

    # Generate LLM/MLLM task JSONs
    if runs:
        print("\nGenerating LLM/MLLM task JSONs...", flush=True)
        tasks = sorted(set(run.task for run in runs))
        for task in tasks:
            print(f"  Generating {task}...", flush=True)
            generate_task_json(task, runs, output_dir, llm_config_dir)

    # Generate diffusion task JSONs
    if diffusion_runs:
        print("\nGenerating diffusion task JSONs...", flush=True)
        diffusion_tasks = sorted(set(run.task for run in diffusion_runs))
        for task in diffusion_tasks:
            print(f"  Generating {task}...", flush=True)
            generate_diffusion_task_json(task, diffusion_runs, output_dir, diffusion_config_dir)

    # Generate LLM/MLLM model detail JSONs
    if runs:
        print("\nGenerating LLM/MLLM model detail JSONs...")
        grouped = group_runs_by_model_task(runs)
        total_models = len(grouped)

        pool_args = [
            (model_id, task, model_runs, output_dir, llm_config_dir, idx, total_models)
            for idx, ((model_id, task), model_runs) in enumerate(grouped.items(), 1)
        ]

        num_workers = min(multiprocessing.cpu_count(), total_models)
        print(f"Using {num_workers} workers for parallel processing...", flush=True)

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(_generate_model_json_worker, pool_args)

    # Generate diffusion model detail JSONs
    if diffusion_runs:
        print("\nGenerating diffusion model detail JSONs...")
        diffusion_grouped = defaultdict(list)
        for run in diffusion_runs:
            key = (run.model_id, run.task)
            diffusion_grouped[key].append(run)

        total_diffusion_models = len(diffusion_grouped)

        pool_args = [
            (model_id, task, model_runs, output_dir, diffusion_config_dir, idx, total_diffusion_models)
            for idx, ((model_id, task), model_runs) in enumerate(diffusion_grouped.items(), 1)
        ]

        num_workers = min(multiprocessing.cpu_count(), total_diffusion_models)
        print(f"Using {num_workers} workers for parallel processing...", flush=True)

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(_generate_diffusion_model_json_worker, pool_args)

    print("\nValidating data size...", flush=True)
    validate_size(output_dir, args.max_size_mb)

    print("\nData generation complete!", flush=True)


if __name__ == "__main__":
    main()
