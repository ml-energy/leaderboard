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
from transformers import CLIPModel, CLIPProcessor
from diffusers import ModelMixin, AutoPipelineForText2Image, DiffusionPipeline  # type: ignore
from zeus.monitor import ZeusMonitor

# Disable torch gradients globally
torch.set_grad_enabled(False)


CLIP = "openai/clip-vit-large-patch14"


@dataclass
class Results:
    model: str
    num_parameters: dict[str, int]
    gpu_model: str
    power_limit: int
    batch_size: int
    num_prompts: int
    average_clip_score: float = 0.0
    total_runtime: float = 0.0
    total_energy: float = 0.0
    average_batch_latency: float = 0.0
    average_images_per_second: float = 0.0
    average_batch_energy: float = 0.0
    average_power_consumption: float = 0.0
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
    batch_latency: float
    sample_energy: float
    prompt: str
    image_path: str | None
    clip_score: float


def get_pipeline(model_id: str):
    """Instantiate a Diffusers pipeline from a modes's HuggingFace Hub ID."""
    # Load args to give to `from_pretrained` from the model's kwargs.json file
    kwargs = json.load(open(f"models/{model_id}/kwargs.json"))
    with suppress(KeyError):
        kwargs["torch_dtype"] = eval(kwargs["torch_dtype"])

    # Add additional args
    kwargs["safety_checker"] = None
    
    try:
        pipeline =  AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).to("cuda:0")
        print("Instantiated AutoPipelineForText2Image:, ", pipeline)
    except ValueError:
        pipeline = DiffusionPipeline.from_pretrained(model_id, **kwargs).to("cuda:0")
        print("Instantiated DiffusionPipeline:, ", pipeline)

    return pipeline


def load_partiprompts(batch_size: int, seed: int) -> tuple[int, list[list[str]]]:
    """Load the parti-prompts dataset and return it as a list of batches of prompts.

    Depending on the batch size, the final batch may not be full. The final batch
    is dropped in that case.

    Returns:
        Total number of prompts and a list of batches of prompts.
    """
    dataset = load_dataset("nateraw/parti-prompts", split="train").shuffle(seed=seed)
    assert isinstance(dataset, Dataset)
    prompts: list[str] = dataset["Prompt"]
    batched = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    if len(batched[-1]) < batch_size:
        batched.pop()
    return len(prompts) * batch_size, batched


def calculate_clip_score(
    model: CLIPModel,
    processor: CLIPProcessor,
    images_np: np.ndarray,
    text: list[str],
) -> torch.Tensor:
    """Calculate the CLIP score for each image and prompt pair.

    Returns:
        The clip score of each image and prompt as a list of floats.
        Tensor shape is (batch size,).
    """
    model = model.to("cuda:0")

    images_np = (images_np * 255).astype("uint8")
    images = list(torch.from_numpy(images_np).permute(0, 3, 1, 2))
    assert len(images) == len(text)
    
    processed_input = processor(text=text, images=images, return_tensors="pt", padding=True)
    img_features = model.get_image_features(processed_input["pixel_values"].to("cuda:0"))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    max_position_embeddings = model.config.text_config.max_position_embeddings
    if processed_input["attention_mask"].shape[-1] > max_position_embeddings:
        print(
            f"Input attention mask is larger than max_position_embeddings. "
            f"Truncating the attention mask to {max_position_embeddings}."
        )
        processed_input["attention_mask"] = processed_input["attention_mask"][..., :max_position_embeddings]
        processed_input["input_ids"] = processed_input["input_ids"][..., :max_position_embeddings]

    txt_features = model.get_text_features(
        processed_input["input_ids"].to("cuda:0"), processed_input["attention_mask"].to("cuda:0")
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    scores = 100 * (img_features * txt_features).sum(axis=-1)
    scores = torch.max(scores, torch.zeros_like(scores))
    return scores


def count_parameters(pipeline) -> dict[str, int]:
    """Count the number of parameters in the given pipeline."""
    num_params = {}
    for name, attr in vars(pipeline).items():
        if isinstance(attr, ModelMixin):
            num_params[name] = attr.num_parameters(only_trainable=False, exclude_embeddings=True)
    return num_params


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


@torch.no_grad
def benchmark(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    results_dir = Path(args.results_root) / args.model_id
    results_dir.mkdir(parents=True, exist_ok=True)
    benchmark_name = str(results_dir / f"bs{args.batch_size}+pl{args.power_limit}")
    image_dir = results_dir / f"bs{args.batch_size}+pl{args.power_limit}+generated"
    image_dir.mkdir(exist_ok=True)

    arg_out_filename = f"{benchmark_name}+args.json"
    with open(arg_out_filename, "w") as f:
        f.write(json.dumps(vars(args), indent=2))
    print(args)
    print("Benchmark args written to", arg_out_filename)

    zeus_monitor = ZeusMonitor()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(zeus_monitor.nvml_gpu_indices[0])
    gpu_model = pynvml.nvmlDeviceGetName(handle)
    pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, args.power_limit * 1000)
    pynvml.nvmlShutdown()

    num_prompts, batched_prompts = load_partiprompts(args.batch_size, args.seed)

    pipeline = get_pipeline(args.model_id)

    # Warmup
    for i in range(3):
        _ = pipeline(batched_prompts[i], output_type="np")

    rng = torch.manual_seed(args.seed)

    images = []
    intermediates: list[ResultIntermediateBatched] = [
        ResultIntermediateBatched(prompts=batch) for batch in batched_prompts
    ]

    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    zeus_monitor.begin_window("benchmark", sync_cuda=False)

    for intermediate in intermediates:
        zeus_monitor.begin_window("batch", sync_cuda=False)
        images = pipeline(intermediate.prompts, generator=rng, output_type="np", **args.settings).images
        batch_measurements = zeus_monitor.end_window("batch", sync_cuda=False)

        intermediate.images = images
        intermediate.batch_latency = batch_measurements.time
        intermediate.batch_energy = batch_measurements.total_energy

    measurements = zeus_monitor.end_window("benchmark", sync_cuda=False)
    peak_memory = torch.cuda.max_memory_allocated(device="cuda:0")

    # Compute the CLIP score for each image and prompt pair.
    # Code was mostly inspired from torchmetrics.multimodal.clip_score, but
    # adapted here to calculate the CLIP score for each image and prompt pair.
    clip_model: CLIPModel = CLIPModel.from_pretrained(CLIP).cuda()  # type: ignore
    clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(CLIP)  # type: ignore
    clip_score_tensors = []
    for intermediate in intermediates:
        clip_score = calculate_clip_score(
            clip_model,
            clip_processor,
            intermediate.images,
            intermediate.prompts,
        )
        clip_score_tensors.append(clip_score)

    results: list[Result] = []
    ind = 0
    for intermediate, clip_score_tensor in zip(intermediates, clip_score_tensors, strict=True):
        for image, prompt, clip_score in zip(
            intermediate.images,
            intermediate.prompts,
            clip_score_tensor.tolist(),
            strict=True,
        ):
            if ind % args.image_save_every == 0:
                image_path = str(image_dir / f"{prompt}.png")
                Image.fromarray((image * 255).astype(np.uint8)).save(image_path)
            else:
                image_path = None

            results.append(
                Result(
                    batch_latency=intermediate.batch_latency,
                    sample_energy=intermediate.batch_energy / len(intermediate.prompts),
                    prompt=prompt,
                    image_path=image_path,
                    clip_score=clip_score,
                )
            )
            ind += 1

    final_results = Results(
        model=args.model_id,
        num_parameters=count_parameters(pipeline),
        gpu_model=gpu_model,
        power_limit=args.power_limit,
        batch_size=args.batch_size,
        num_prompts=num_prompts,
        average_clip_score=sum(r.clip_score for r in results) / len(results),
        total_runtime=measurements.time,
        total_energy=measurements.total_energy,
        average_batch_latency=measurements.time / len(batched_prompts),
        average_images_per_second=num_prompts / measurements.time,
        average_batch_energy=measurements.total_energy / len(batched_prompts),
        average_power_consumption=measurements.total_energy / measurements.time,
        peak_memory=peak_memory,
        results=results,
    )

    with open(f"{benchmark_name}+restults.json", "w") as f:
        f.write(json.dumps(asdict(final_results), indent=2))
    print("Benchmark results written to", f"{benchmark_name}+restults.json")
    print("Benchmark results:", final_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="The model to benchmark.")
    parser.add_argument("results_root", type=str, help="The root directory to save results to.")
    parser.add_argument("--batch_size", type=int, default=1, help="The size of each batch of prompts.")
    parser.add_argument("--power_limit", type=int, default=300, help="The power limit to set for the GPU in Watts.")
    parser.add_argument("--seed", type=int, default=0, help="The seed to use for the RNG.")
    parser.add_argument("--image_save_every", type=int, default=10, help="Save images to file every N prompts.")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14", help="The CLIP model to use to calculate the CLIP score.")
    parser.add_argument("--settings", type=json.loads, default={}, help="Any additional settings to pass to pipeline forward.")
    args = parser.parse_args()

    benchmark(args)
