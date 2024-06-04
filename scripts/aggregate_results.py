import json
from glob import glob
from pathlib import Path
from contextlib import suppress

import tyro


FIELDS = {
    "model": "model",
    "backend": "backend",
    "gpu_model": "gpu_model",
    "num_gpus": "num_gpus",
    "request_rate": "request_rate",
    "requests_per_second": "requests_per_second",
    "latency_per_request": "latency_per_request",
    "client_side_energy_per_request": "energy_per_request",
    "num_parameters": "num_parameters",
    "batch_size": "batch_size",
    "num_inference_steps": "num_inference_steps",
    "average_batch_latency": "batch_latency",
    "average_batch_energy": "batch_energy",
}

def main(result_dir: Path, output_dir: Path) -> None:
    print(f"{result_dir} -> {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in sorted(glob(f"{result_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        target_model_dir = Path(output_dir / model_name)
        target_model_dir.mkdir(parents=True)
        for file in sorted(glob(f"{model_dir}/*+results.json")):
            pieces = filter(lambda p: "pl" not in p and "gpu" not in p, file.split("/")[-1].split("+")[:-1])
            target_path = target_model_dir / ("+".join(pieces) + ".json")
            print(f"    {target_path}")
            raw_data = json.load(open(file))
            data = {}
            for field1, field2 in FIELDS.items():
                with suppress(KeyError):
                    data[field2] = raw_data.pop(field1)
            json.dump(data, open(target_path, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
