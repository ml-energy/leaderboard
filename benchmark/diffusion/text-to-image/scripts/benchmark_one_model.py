from __future__ import annotations

import os
import argparse
import subprocess


########### Parameter space ###########
batch_sizes: list[str] = [
    "32",
    "16",
    "8",
    "4",
]

power_limits: list[str] = [
    "300",
    "250",
    "200",
    "150",
    "100",
]
#######################################


def print_and_write(outfile, line: str, flush: bool = False):
    print(line, end="", flush=flush)
    outfile.write(line)
    if flush:
        outfile.flush()


def main(args: argparse.Namespace) -> None:
    outdir = f"{args.result_root}/{args.model}"
    os.makedirs(outdir, exist_ok=True)

    outfile = open(f"{outdir}/gpus{''.join(args.gpu_ids)}.out.txt", "w")

    print_and_write(outfile, f"Benchmarking {args.model}\n")
    print_and_write(outfile, f"Batch sizes: {batch_sizes}\n")
    print_and_write(outfile, f"Power limits: {power_limits}\n")

    for batch_size in batch_sizes:
        for power_limit in power_limits:
            print_and_write(outfile, f"{batch_size=}, {power_limit=}\n", flush=True)
            with subprocess.Popen(
                args=[
                    "docker", "run",
                    "--gpus", '"device=' + ','.join(args.gpu_ids) + '"',
                    "--cap-add", "SYS_ADMIN",
                    "-v", "/data/leaderboard/hfcache:/root/.cache/huggingface",
                    "-v", f"{os.getcwd()}:/workspace/text-to-image",
                    "mlenergy/leaderboard:diffusion-benchmark",
                    "--result-root", args.result_root,
                    "--batch-size", batch_size,
                    "--num-batches", "10",
                    "--power-limit", power_limit,
                    "--model", args.model,
                    # "--huggingface-token", args.huggingface_token,
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

            # If proc exited with non-zero status, it's probably an OOM.
            # Move on to the next batch size.
            if proc.returncode != 0:
                break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="ID of the model to benchmark")
    parser.add_argument("--result-root", type=str, help="Root directory to store the results")
    parser.add_argument("--huggingface-token", type=str, help="Huggingface API token")
    parser.add_argument("--gpu-ids", type=str, nargs="+", help="GPU IDs to use")
    args = parser.parse_args()
    main(args)
