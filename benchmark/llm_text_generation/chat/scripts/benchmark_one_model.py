from __future__ import annotations

import os
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


def print_and_write(outfile, line: str, flush: bool = False):
    print(line, end="", flush=flush)
    outfile.write(line)
    if flush:
        outfile.flush()


def main(args: argparse.Namespace) -> None:
    hf_token = os.environ["HF_TOKEN"]

    outdir = f"{args.result_root}/{args.model}"
    os.makedirs(outdir, exist_ok=True)

    outfile = open(f"{outdir}/gpus{''.join(args.gpu_ids)}.out.txt", "w")

    print_and_write(outfile, f"Benchmarking {args.model}\n")
    print_and_write(outfile, f"Request rates: {request_rates}\n")
    print_and_write(outfile, f"Power limits: {power_limits}\n")

    for backend, request_rate, power_limit in product(backends, request_rates, power_limits):
        print_and_write(outfile, f"{backend=}, {request_rate=}, {power_limit=}\n", flush=True)
        with subprocess.Popen(
            args=[
                "python",
                "scripts/benchmark_one_datapoint.py",
                "--backend", backend,
                "--server-image", server_image[backend],
                "--model", args.model,
                "--sharegpt-path", "sharegpt/ShareGPT_V3_filtered_500.json",
                "--request-rate", request_rate,
                "--power-limit", power_limit,
                "--result-root", args.result_root,
                "--huggingface-token", hf_token,
                "--gpu-ids", *args.gpu_ids,
                "--log-level", "INFO",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as proc:
            if proc.stdout:
                i = 0
                for line in proc.stdout:
                    print_and_write(outfile, line, flush=i % 50 == 0)
                    i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="ID of the model to benchmark")
    parser.add_argument("--result-root", type=str, help="Root directory to store the results")
    parser.add_argument("--gpu-ids", type=str, nargs="+", help="GPU IDs to use")
    args = parser.parse_args()
    main(args)
