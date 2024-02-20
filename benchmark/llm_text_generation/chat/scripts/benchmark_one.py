from __future__ import annotations

import os
import json
import atexit
import argparse
import signal
import subprocess
from pathlib import Path
from typing import Literal


def set_power_limit(power_limit: int, gpu_ids: list[int]) -> None:
    for gpu_id in gpu_ids:
        subprocess.check_call([
            "docker", "run", "nvml",
            "nvidia-smi", "-i", str(gpu_id), "-pm", "1",
        ])
        subprocess.check_call([
            "docker", "run", "nvml",
            "nvidia-smi", "-i", str(gpu_id), "-pl", str(power_limit),
        ])


def start_server(
    backend: Literal["vllm", "tgi"],
    server_image: str,
    port: int,
    model: str,
    huggingface_token: str,
    gpu_ids: list[int],
) -> subprocess.Popen:
    gpu_str = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    gpu_str = f'"device={gpu_str}"'
    hf_cache_path = "/data/leaderboard/hfcache"
    tokconf_dir = f"{os.getcwd()}/tokenizer_configs"
    tokconf_filename = f"{model}/tokenizer_config.json"
    tokconf_path = f"{tokconf_dir}/{tokconf_filename}"

    assert Path(hf_cache_path).exists(), f"Hugging Face cache not found: {hf_cache_path}"
    assert Path(tokconf_path).exists(), f"Tokenizer config not found: {tokconf_path}"

    if backend == "vllm":
        server_cmd = [
            "docker", "run",
            "--gpus", gpu_str,
            "-e", f"HF_TOKEN={huggingface_token}",
            "-p", f"{port}:8000",
            "-v", f"{hf_cache_path}:/root/.cache/huggingface",
            server_image,
            "--model", model,
            "--chat-template", json.load(open(tokconf_path))["chat_template"],
            "--tensor-parallel-size", str(len(gpu_ids)),
        ]
    elif backend == "tgi":
        server_cmd = [
            "docker", "run",
            "--gpus", gpu_str,
            "-e", f"HUGGING_FACE_HUB_TOKEN={huggingface_token}",
            "-p", f"{port}:80",
            "-v", f"{hf_cache_path}:/root/.cache/huggingface",
            "-v", f"{tokconf_dir}:/tokenizer_configs",
            server_image,
            "--model-id", model,
            "--huggingface-hub-cache", "/root/.cache/huggingface/hub",
            "--tokenizer-config-path", f"/{tokconf_filename}",
            "--num-shard", str(len(gpu_ids)),
        ]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Server:", " ".join(server_cmd))
    return subprocess.Popen(server_cmd)


def start_client(
    backend: Literal["vllm", "tgi"],
    port: int,
    model: str,
    sharegpt_path: str,
    request_rate: str,
    gpu_ids: list[int],
    benchmark_name: str,
) -> subprocess.Popen:
    client_cmd = [
        "python", "scripts/benchmark_client.py",
        "--backend", backend,
        "--port", str(port),
        "--model", model,
        "--sharegpt-path", sharegpt_path,
        "--request-rate", request_rate,
        "--benchmark-name", benchmark_name,
    ]
    print("Client:", " ".join(client_cmd))
    return subprocess.Popen(
        client_cmd,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids)},
    )


def terminate_server(server_handle: subprocess.Popen) -> None:
    server_handle.send_signal(signal.SIGINT)
    try:
        server_handle.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_handle.kill()


def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.result_root) / args.model
    benchmark_name = str(
        results_dir / f"{args.backend}+rate{args.request_rate}+pl{args.power_limit}+gpus{len(args.gpu_ids)}",
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    port = 8000 + args.gpu_ids[0]

    server_handle = start_server(
        args.backend,
        args.server_image,
        port,
        args.model,
        args.huggingface_token,
        args.gpu_ids,
    )
    atexit.register(lambda: terminate_server(server_handle))

    set_power_limit(args.power_limit, args.gpu_ids)

    client_handle = start_client(
        args.backend,
        port,
        args.model,
        args.sharegpt_path,
        args.request_rate,
        args.gpu_ids,
        benchmark_name,
    )

    try:
        exit_code = client_handle.wait(timeout=2 * 3600)
    except subprocess.TimeoutExpired:
        client_handle.terminate()
        raise RuntimeError("Benchmark client timed out after two hours")

    if exit_code != 0:
        raise RuntimeError(f"Benchmark client exited with code {exit_code}")

    terminate_server(server_handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["vllm", "tgi"], help="Server to benchmark.")
    parser.add_argument("--server-image", help="Docker image to use for the server.")
    parser.add_argument("--model", required=True, help="Model to benchmark, e.g., meta-llama/Llama-2-7b-chat-hf.")
    parser.add_argument("--sharegpt-path", help="Path to the ShareGPT dataset to feed to the server.")
    parser.add_argument("--request-rate", required=True, help="Poisson process rate for request arrival times.")
    parser.add_argument("--power-limit", type=int, help="GPU power limit in Watts.")
    parser.add_argument("--result-root", default="results", help="Root directory to save results.")
    parser.add_argument("--huggingface-token", help="Hugging Face API token.")
    parser.add_argument("--gpu-ids", nargs="+", type=int, help="GPU IDs to use for the server.")
    main(parser.parse_args())
