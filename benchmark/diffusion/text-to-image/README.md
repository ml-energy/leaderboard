# LLM Text Generation (Chat)

This benchmark suite benchmarks vLLM and TGI with the chat completion task.

## Setup

### Building Docker images

```sh
git clone git@github.com:ml-energy/vllm.git
cd vllm
git checkout v0.3.0
DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag mlenergy/vllm:v0.3.0-openai --build-arg max_jobs=16 --build-arg nvcc_threads=16
```

```sh
git clone git@github.com:ml-energy/text-generation-inference.git
cd text-generation-inference
git checkout v1.4.0
docker build -t mlenergy/tgi:v1.4.0 .
```

### Installing Benchmark Script Dependencies

```sh
pip install -r requirements.txt
```

### Starting the NVML container

Changing the power limit requires the `SYS_ADMIN` linux security capability, which we delegate to a daemon Docker container running a base CUDA image.

```sh
bash ../../common/start_nvml_container.sh
```

With the `nvml` container running, you can change power limit with something like `docker exec nvml nvidia-smi -i 0 -pl 200`.


## Benchmarking

### Obtaining one datapoint

The script `scripts/benchmark_one.py` assumes that it was ran from the directory where `scripts` is, like this:
```sh
python scripts/benchmark_one.py --help
```
