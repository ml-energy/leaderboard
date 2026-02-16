# The ML.ENERGY Leaderboard

Source code for [The ML.ENERGY Leaderboard](https://ml.energy/leaderboard), which is a web leaderboard that displays results of [The ML.ENERGY benchmark](https://github.com/ml-energy/benchmark).


## Running the Leaderboard Web App

### Building the Data

After running the [benchmark](https://github.com/ml-energy/benchmark) and collecting results, you can build the data for the leaderboard using the following command.
This depends on [The ML.ENERGY Data Toolkit](https://github.com/ml-energy/data), which can be installed with `pip install mlenergy-data` or used with `uv run` as shown below.

```bash
uv run --with mlenergy-data scripts/build_data.py \
  --output-dir public/data
```

If you are compiling the leaderboard data from a local directory that contains the ML.ENERGY Benchmark data, add `--mlenergy-data-dir /path/to/compiled/data` to the command above.
In case you ran the benchmark on your own and have raw result data directories, modify the script to use `LLMRuns.from_raw_results`.

### Web App Preview

```bash
npm install
npm run dev
```

## Citation

```bibtex
@inproceedings{mlenergy-neuripsdb25,
    title={The {ML.ENERGY Benchmark}: Toward Automated Inference Energy Measurement and Optimization}, 
    author={Jae-Won Chung and Jeff J. Ma and Ruofan Wu and Jiachen Liu and Oh Jun Kweon and Yuxuan Xia and Zhiyu Wu and Mosharaf Chowdhury},
    year={2025},
    booktitle={NeurIPS Datasets and Benchmarks},
}
```
