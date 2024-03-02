from __future__ import annotations

import time
import json
import argparse
from pathlib import Path
from contextlib import suppress
from dataclasses import dataclass, field, asdict

import torch
import pynvml
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
from transformers.trainer_utils import set_seed
from diffusers import AutoPipelineForText2Image, DiffusionPipeline  # type: ignore
from torchmetrics.functional.multimodal import clip_score
from zeus.monitor import ZeusMonitor


@dataclass
class Results:
    model: str
    gpu_model: str
    power_limit: int
    batch_size: int
    num_batches: int
    clip_score: float = 0.0
    total_runtime: float = 0.0
    total_energy: float = 0.0
    average_batch_latency: float = 0.0
    average_images_per_second: float = 0.0
    average_batch_energy: float = 0.0
    peak_memory: float = 0.0
    results: list[Result] = field(default_factory=list)


@dataclass
class ResultIntermediateBatched:
    batch_latency: float = 0.0
    batch_energy: float = 0.0
    prompts: list[str] = field(default_factory=list)
    images: np.ndarray = np.empty(0)


@dataclass
class Result:
    latency: float  # Batch latency
    energy: float   # Batch energy divided by batch size
    prompt: str
    image_path: str
    clip_score: float


# default parameters
DEVICE = "cuda:0"
WEIGHT_DTYPE = torch.bfloat16
SEED = 0
OUTPUT_FILE = "results.csv"
OUTPUT_IMAGES = "images/"


def get_pipeline(model_id: str):
    """Instantiate a Diffusers pipeline from a modes's HuggingFace Hub ID."""
    # Load args to give to `from_pretrained` from the model's kwargs.json file
    kwargs = json.load(open(f"models/{model_id}/kwargs.json"))
    with suppress(KeyError):
        kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])

    # Add additional args
    kwargs["safety_checker"] = None
    
    try:
        pipeline =  AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).cuda()
        print("Instantiated AutoPipelineForText2Image:, ", pipeline)
    except ValueError:
        pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs).cuda()
        print("Instantiated DiffusionPipeline:, ", pipeline)

    return pipeline


def load_partiprompts(batch_size: int, seed: int) -> list[list[str]]:
    """Load the parti-prompts dataset and return it as a list of batches of prompts.

    Parti-prompts has 1,632 prompts. Thus depending on the batch size, the final batch may not be full.
    """
    dataset = load_dataset("nateraw/parti-prompts", split="train").shuffle(seed=seed)
    assert isinstance(dataset, Dataset)
    prompts: list[str] = dataset["Prompt"]

    batched_prompts = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    return batched_prompts


def calculate_clip_score(images, prompts):
    """Calculate CLIP Score metric.

    Args:
        images (np.ndarray): array of images
        prompts (list): list of prompts, assumes same size as images

    Returns:
        The clip score across all images and prompts as a float.
    """
    clip_score_fn = partial(
        clip_score, model_name_or_path="openai/clip-vit-base-patch16"
    )
    images_int = (images * 255).astype("uint8")
    clip = clip_score_fn(
        torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
    ).detach()
    return float(clip)


def gpu_warmup(pipeline):
    """Warm up the GPU by running the given pipeline for 10 secs."""
    print("Warming up GPU")
    generator = torch.manual_seed(2)
    timeout_start = time.time()
    prompts, _ = load_partiprompts(1, 1)
    while time.time() < timeout_start + 10:
        _ = pipeline(
            prompts, num_images_per_prompt=10, generator=generator, output_type="np"
        ).images
    print("Finished warming up GPU")


def benchmark(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    results_dir = Path(args.results_root) / args.model_id
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_name = str(results_dir / f"bs{args.batch_size}+pl{args.power_limit}")

    arg_out_filename = f"{benchmark_name}+args.json"
    with open(arg_out_filename, "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    print(args)
    print("Benchmark args written to", arg_out_filename)

    zeus_monitor = ZeusMonitor()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(zeus_monitor.nvml_gpu_indices[0])
    gpu_model = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, args.power_limit * 1000)
    pynvml.nvmlShutdown()

    batched_prompts = load_partiprompts(args.batch_size, args.seed)

    results = Results(
        model=args.model_id,
        gpu_model=gpu_model,
        power_limit=args.power_limit,
        batch_size=args.batch_size,
        num_batches=len(batched_prompts),
    )

    generator = torch.manual_seed(args.seed)
    pipeline = get_pipeline(args.model_id)

    gpu_warmup(pipeline)

    images = []
    intermediates: list[ResultIntermediateBatched] = [
        ResultIntermediateBatched(prompts=batch) for batch in batched_prompts
    ]

    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    zeus_monitor.begin_window("benchmark", sync_cuda=False)

    for intermediate in intermediates:
        zeus_monitor.begin_window("batch", sync_cuda=False)
        images = pipeline(intermediate.prompts, generator=generator, output_type="np", **args.settings).images
        measurements = zeus_monitor.end_window("batch", sync_cuda=False)

        intermediate.images = images
        intermediate.batch_latency = measurements.time
        intermediate.batch_energy = measurements.total_energy
        # images.append(images)

    measurements = zeus_monitor.end_window("benchmark", sync_cuda=False)
    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0")

    # TODO: Continue here.
    images = np.concatenate(images)

    for saved_image, saved_prompt in zip(images[::10], prompts[::10]):
        saved_image = (saved_image * 255).astype(np.uint8)
        Image.fromarray(saved_image).save(images_path + saved_prompt + ".png")

    clip_score = calculate_clip_score(images, prompts)

    # TODO: If the final batch was smaller than the batch size, we should adjust the results accordingly.

    result = {
        "model": model_id,
        "GPU": gpu_model,
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "clip_score": clip_score,
        "avg_batch_latency": result_monitor.time / (benchmark_size / batch_size),
        "throughput": benchmark_size / result_monitor.time,
        "avg_energy": result_monitor.total_energy / benchmark_size,
        "peak_memory": peak_memory,
    }

    with open(f"{benchmark_name}+restults.json", "w") as f:
        f.write(json.dumps(asdict(results), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="The model to benchmark.")
    parser.add_argument("results_root", type=str, help="The root directory to save results to.")
    parser.add_argument("--batch_size", type=int, default=1, help="The size of each batch of prompts.")
    parser.add_argument("--power_limit", type=int, default=300, help="The power limit to set for the GPU in Watts.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for the RNG.")
    parser.add_argument("--settings", type=json.loads, default={}, help="Any additional settings to pass to pipeline forward.")
    args = parser.parse_args()

    benchmark(args)
