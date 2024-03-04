# Diffusion model (Text to Image)

This benchmark suite benchmarks diffusion models with the text-to-image task.

## Setup

### Building Docker images

```sh
docker build -t mlenergy/leaderboard:diffusion-benchmark .
```

### HuggingFace cache directory

The scripts assume the HuggingFace cache directory will be under `/data/leaderboard/hfcache` on the node that runs this benchmark.

## Benchmarking

### Obtaining one datapoint

The Docker image we've build runs `python scripts/benchmark_one_datapoint.py` as its `ENTRYPOINT`.

```sh
docker run \
  --gpus '"device=0"' \
  --cap-add SYS_ADMIN \
  -v /data/leaderboard/hfcache:/root/.cache/huggingface 
  -v $(pwd):/workspace/text-to-image \
  mlenergy/leaderboard:diffusion-benchmark \
  --results-root results \
  --batch-size 2 \
  --power-limit 300 \
  --image-save-every 5 \
  --model stabilityai/stable-diffusion-2-1
```

### Obtaining all datapoints for a single model

Run `scripts/benchmark_one_model.py`.
