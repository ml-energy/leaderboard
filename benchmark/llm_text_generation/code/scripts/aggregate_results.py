import json
from glob import glob
from pathlib import Path
from contextlib import suppress

import tyro


FIELDS = {
    "model": "Model",
    "backend": "Backend",
    "gpu_model": "GPU model",
    "num_gpus": "#GPUs",
    "requests_per_second": "Requests per second",
    "latency_per_request": "Latency per request (s)",
    "client_side_energy_per_request": "Energy per request (J)",
}

def main(result_dir: Path, output_dir: Path) -> None:
    print(f"{result_dir} -> {output_dir}")

    (output_dir / "vLLM").mkdir(parents=True, exist_ok=True)
    (output_dir / "TGI").mkdir(parents=True, exist_ok=True)

    for model_dir in sorted(glob(f"{result_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        (output_dir / "vLLM" / model_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "TGI" / model_name).mkdir(parents=True, exist_ok=True)
        for file in sorted(glob(f"{model_dir}/*+results.json")):
            pieces = list(filter(lambda p: "pl" not in p and "gpu" not in p, file.split("/")[-1].split("+")[:-1]))
            if pieces[0] not in ["vllm", "tgi"]:
                raise ValueError(f"Unknown backend type: {pieces[0]}")
            target_path = output_dir / ("vLLM" if pieces[0] == "vllm" else "TGI") / model_name / ("+".join(pieces[1:]) + ".json")
            print(f"    {target_path}")
            raw_data = json.load(open(file))
            data = {}
            for field1, field2 in FIELDS.items():
                with suppress(KeyError):
                    data[field2] = raw_data.pop(field1)
            json.dump(data, open(target_path, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
