# LLM Text Generation (Chat)

This benchmark suite benchmarks vLLM and TGI with the code generation task.

## Building Docker images

```sh
git clone git@github.com:ml-energy/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build . --target vllm-api --tag mlenergy/vllm:v0.3.0-api --build-arg max_jobs=16 --build-arg nvcc_threads=16
```
