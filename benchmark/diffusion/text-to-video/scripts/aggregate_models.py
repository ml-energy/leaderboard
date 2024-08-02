import json
from glob import glob
from pathlib import Path

import tyro

def main(models_dir: Path, output_file: Path) -> None:
    print(f"{models_dir} -> {output_file}")

    if not Path(models_dir).exists():
        raise ValueError(f"Directory {models_dir} does not exist")

    models = {}
    for model_dir in sorted(glob(f"{models_dir}/*/*")):
        model_name = "/".join(model_dir.split("/")[-2:])
        print(f"  {model_name}")
        model_info = dict(
            url=f"https://huggingface.co/{model_name}",
            nickname=model_name.split("/")[-1],
            total_params="NA",
            unet_params="NA",
        )
        assert model_name not in models
        models[model_name] = model_info

    json.dump(models, open(output_file, "w"), indent=2)


if __name__ == "__main__":
    tyro.cli(main)
