import json
from glob import glob
from pathlib import Path
from contextlib import suppress

import tyro


FIELDS = {
    "model": "Model",
    "gpu_model": "GPU model",
    "batch_size": "Batch size",
    "num_inference_steps": "Inference steps",
    "average_batch_latency": "Batch latency (s)",
    "energy_per_image": "Energy per image (J)",
}

def main(result_dir: Path, output_dir: Path) -> None:
    print(f"{result_dir} -> {output_dir}")

    (output_dir / "30").mkdir(parents=True, exist_ok=True)
    (output_dir / "40").mkdir(parents=True, exist_ok=True)
    (output_dir / "50").mkdir(parents=True, exist_ok=True)

    for model_dir in sorted(glob(f"{result_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        (output_dir / "30" / model_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "40" / model_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "50" / model_name).mkdir(parents=True, exist_ok=True)
        for file in sorted(glob(f"{model_dir}/*pl400*+results.json")):
            pieces = list(filter(lambda p: "pl" not in p, file.split("/")[-1].split("+")[:-1]))
            if pieces[1] not in ["steps30", "steps40", "steps50"]:
                raise ValueError(f"Unknown number of inference steps: {pieces[1]}")
            target_path = output_dir / pieces[1][-2:] / model_name / f"{pieces[0]}.json"
            print(f"    {target_path}")
            raw_data = json.load(open(file))
            raw_data["energy_per_image"] = raw_data["average_batch_energy"] / raw_data["batch_size"]
            data = {}
            for field1, field2 in FIELDS.items():
                with suppress(KeyError):
                    data[field2] = raw_data.pop(field1)
            json.dump(data, open(target_path, "w"), indent=2)
        params = {name: f"{p / 1e6:.2f}M" for name, p in raw_data["num_parameters"].items()}
        print(f"      {params}, total={sum(raw_data['num_parameters'].values()) / 1e6:.2f}M")


if __name__ == "__main__":
    tyro.cli(main)
