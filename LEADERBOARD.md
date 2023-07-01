The goal of the ML.ENERGY Leaderboard is to give people a sense of how much **energy** LLMs would consume.

## How is energy different?

The energy consumption of running inference on a model will depends on factors such as architecture, size, and GPU model.
However, even if we run models with the exact same architecture and size on the same GPU, the average energy consumption **per prompt** is different because different models have **different verbosity**.
That is, when asked the same thing, different models answer in different lengths.

## Columns

- `gpu`: NVIDIA GPU model name. Note that NLP evaluation was only run once on our A40 GPUs, so this column only changes system-level measurements like latency and energy.
- `task`: Name of the task. See *Tasks* below for details.
- `energy_eff`: Our definition of energy efficiency: Average NLP evaluation metric attained per Joule of energy.
- `energy` (J): The average energy consumed by the model to generate a response.
- `nlp_average`: The arithmetic average of the NLP evaluation metrics we obtained. See *NLP evaluation metrics* below for details.
- `throughput` (token/s): The average number of tokens generated per second.
- `latency` (s): The average time it took for the model to generate a response.
- `response_length` (token): The average number of tokens in the model's response.
- `parameters`: The number of parameters the model has, in units of billion.

## Tasks

For each task, every model uses the same system prompt. We still account for differences in roles, e.g. `USER`, `HUMAN`, `ASSISTANT`, `GPT`.

| Name | System prompt |
|--|--|
| chat | A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. |
| chat-concise | A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant's answers are very concise. |
| instruct | Below is an instruction that describes a task. Write a response that appropriately completes the request. |
| instruct-concise | Below is an instruction that describes a task. Write a response that appropriately completes the request. The response should be very concise. |

You can see that response length is shorter on average for the `-concise` variants of the tasks.
This affects the number of decoding iterations the model has to run in order to finish responding, thus affecting latency and energy consumption per prompt.

## Setup

Find our benchmark script for one model [here](https://github.com/ml-energy/leaderboard/blob/master/benchmark.py).

### Software

- PyTorch 2.0.1
- [Zeus](https://ml.energy/zeus) -- For GPU time and energy measurement
- [FastChat](https://github.com/lm-sys/fastchat) -- For running inference on various models
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/commit/72b7f0c00a6ff94632c5b873fc24e093ae74fa47) -- For NLP evaluation metrics

### Hardware

- NVIDIA A40 GPU
- NVIDIA A100 GPU

### Parameters

- Model
  - Batch size 1
  - FP16
- Sampling (decoding)
  - Greedy sampling from multinomial distribution
  - Temperature 0.7
  - Repetition penalty 1.0

## Data

We randomly sampled around 3000 prompts from the [cleaned ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
See [here](https://github.com/ml-energy/leaderboard/tree/master/sharegpt) for more detail on how we created the benchmark dataset.

We used identical system prompts for all models (while respecting their own *role* tokens):
```
A chat between a human user (prompter) and an artificial intelligence (AI) assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
```

## NLP evaluation metrics

- `arc`: [AI2 Reasoning Challenge](https://allenai.org/data/arc)'s `challenge` dataset, measures capability to do grade-school level question answering, 25 shot
- `hellaswag`: [HellaSwag dataset](https://allenai.org/data/hellaswag), measuring grounded commonsense, 10 shot
- `truthfulqa`: [TruthfulQA dataset](https://arxiv.org/abs/2109.07958), measuring truthfulness against questions that elicit common falsehoods, 0 shot

## Upcoming

- More optimized inference runtimes, like TensorRT.
- More GPU models, like V100.
- More models, like RWKV.

# License

This leaderboard is a research preview intended for non-commercial use only.
The use of LLaMA weights are subject to their [license](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md).
Please direct inquiries/reports of potential violation to Jae-Won Chung.

# Acknowledgements

We thank [Chameleon Cloud](https://www.chameleoncloud.org/) for the A100 80GB GPU nodes (`gpu_a100_pcie`) and [CloudLab](https://cloudlab.us/) for the V100 GPU nodes (`r7525`).