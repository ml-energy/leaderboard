from __future__ import annotations

import argparse
import subprocess
from itertools import product


########### Parameter space ###########
backends: list[str] = [
    "vllm",
    "tgi",
]

request_rates: list[str] = [
    "4.00",
    "2.00",
    "1.00",
    "0.50",
]

power_limits: list[str] = [
    "300",
    "250",
    "200",
    "100",
]
######################################

server_image: dict[str, str] = {
    "vllm": "mlenergy/vllm:v0.3.3-openai",
    "tgi": "mlenergy/text-generation-inference:v1.4.2",
}


def main(args: argparse.Namespace) -> None:
    print(f"Benchmarking {args.model}")
    print(f"Request rates: {request_rates}")
    print(f"Power limits: {power_limits}")

    for backend, request_rate, power_limit in product(backends, request_rates, power_limits):
        subprocess.run(
            [
                "python",
                "scripts/benchmark_one_datapoint.py",
                "--backend", backend,
                "--server-image", server_image[backend],
                "--model", args.model,
                "--sharegpt-path", "sharegpt/ShareGPT_V3_filtered_500.json",
                "--request-rate", request_rate,
                "--power-limit", power_limit,
                "--result-root", args.result_root,
                "--huggingface-token", args.huggingface_token,
                "--gpu-ids", *args.gpu_ids,
                "--log-level", "INFO",
            ]
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="ID of the model to benchmark")
    parser.add_argument("--result-root", type=str, help="Root directory to store the results")
    parser.add_argument("--huggingface-token", type=str, help="Huggingface API token")
    parser.add_argument("--gpu-ids", type=str, nargs="+", help="GPU IDs to use")
    args = parser.parse_args()
    main(args)
