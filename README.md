# The ML.ENERGY Leaderboard

Source code for [The ML.ENERGY Leaderboard](https://ml.energy/leaderboard), which is a web leaderboard that displays results of [The ML.ENERGY benchmark](https://github.com/ml-energy/benchmark).


## Running the Leaderboard Web App

### Building the Data

After running the [benchmark](https://github.com/ml-energy/benchmark) and collecting results, you can build the data for the leaderboard using the following command (with path adjustments as necessary):

```bash
BASE_DIR="/path/to/results"
uv run --with numpy --with PyYAML scripts/build_data.py \
  --results-dir "$BASE_DIR/llm/h100/run" \
  --results-dir "$BASE_DIR/llm/b200/run" \
  --results-dir "$BASE_DIR/diffusion/h100/run" \
  --results-dir "$BASE_DIR/diffusion/b200/run" \
  --output-dir public/data \
  --llm-config-dir ../benchmark/configs/vllm \
  --diffusion-config-dir ../benchmark/configs/xdit
```

### Web App Preview

```bash
npm run dev
```
