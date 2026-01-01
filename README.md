# Long Context Evaluation for NL2SQL

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

We share the codes, artifacts and experiments for evaluating our long context LLM-based NL2SQL [arxiv](https://arxiv.org/abs/2501.12372). The original experiments were run using `gemini-1.5-pro` and `gemini-1.5-flash` -- via [Vertex AI](https://cloud.google.com/vertex-ai/docs/reference/rest). If you find this work, the experiments and insights useful, please cite (to appear at VLDB25)
```
@article{chung2025long,
  title={Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL},
  author={Chung, Yeounoh and Kakkar, Gaurav T and Gan, Yu and Milne, Brenton and Ozcan, Fatma},
  journal={arXiv preprint arXiv:2501.12372},
  year={2025}
}
```

Some utility functions are taken from [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub) and stored under `lc_nl2sql/third_party/` under MIT license.

All the experiments are packaged under `lc_nl2sql/experiments/` as individual shell script.



## Setup

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management.
- Google Cloud Vertex AI credentials (if using Gemini models). Ensure your environment is authenticated (e.g., `gcloud auth application-default login`).

### Installation
The dependencies are tracked inside `pyproject.toml`. We recommend `poetry` to manage the packages.
```bash
# at the root of the project
pip install poetry
poetry install
```

### Data Preparation
We recommend installing/copying each benchmark dataset under `lc_nl2sql/data/{benchmark}/` for convenience and compatibility with the existing experiment scripts and the paths.

The expected structure is:
```
lc_nl2sql/data/
├── bird/
│   ├── dev/
│   │   ├── dev.json
│   │   ├── dev_tables.json
│   │   └── dev_databases/
│   └── ...
├── spider/
├── kaggle/
└── ...
```

## Configuration
The project configuration is managed in the `lc_nl2sql/configs/` directory:
- `config.py`: Defines global paths (ROOT_PATH, DATA_PATH, OUT_DIR) and dataset-specific information.
- `model_args.py`: Defines arguments for the model (e.g., `model_name_or_path`, `cache_dir`).
- `data_args.py`: Defines arguments for data processing and prompting.

## Running the Code

### 1. Data Processing
Prepare the dataset for the model using `lc_nl2sql/data_process/sql_data_process.py`.

The full example generation relies on separate column selection results, we used the LLM-based approach from [CHESS](https://arxiv.org/abs/2405.16755). 
The example generation process in `sql_data_process.py` takes this optional parameter, 
```
--filtered_schema_file lc_nl2sql/data/bird/col_selection_schema.csv
```
The column selection results csv files for different benchmarks are also provided under `lc_nl2sql/data/{benchmark}/`

For some table retrieval experiments require separate table retrieval results, some were directly dumped from our production pipeline. The example generation process in `sql_data_process.py` takes this optional parameter, 
```
--tbr_selection_file lc_nl2sql/data/bird/crs_dump.json \
```
The results are stored under `lc_nl2sql/data/bird/crs_dump.json` and `lc_nl2sql/data/kaggle/tbr_dump.json`.

Example command:
```bash
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path lc_nl2sql/data/bird/dev/dev.json \
  --input_table_path lc_nl2sql/data/bird/dev/dev_tables.json \
  --output_file_path "$input_file_sk100" \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --tbr_selection_file lc_nl2sql/data/bird/crs_dump.json \
  --num_col_values 10 \
  --use_hint 1 \
  --filtered_schema_file lc_nl2sql/data/bird/col_selection_schema.csv \
  --use_column_filtering 1 \
  --synthetic_examples 1 \
  --num_examples 100
```

### 2. Prediction / Inference
Run the model inference using `lc_nl2sql/predict/predict.py`.

```bash
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "$input_file_sk100" \
  --num_beams 10 \
  --temperature 0.5 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/bird_benchmark"
```

## Reproducible Runs
All experiments are packaged as shell scripts under `lc_nl2sql/experiments/`.

### BIRD Benchmark
```bash
poetry run bash lc_nl2sql/experiments/bird_benchmark.sh
```
Other BIRD experiments:
- `bird_ablation_study.sh`: Ablation studies.
- `bird_challenging_example.sh`: Focus on challenging examples.
- `bird_ablation_study_EURECOM.sh`: EURECOM specific ablation studies.

### Spider Benchmark
```bash
poetry run bash lc_nl2sql/experiments/spider_benchmark.sh
```

### KaggleDBQA Benchmark
```bash
poetry run bash lc_nl2sql/experiments/kaggle_benchmark.sh
```

### BEAVER Benchmark
```bash
poetry run bash lc_nl2sql/experiments/beaver_benchmark.sh
```

The output and artifacts will be stored under `lc_nl2sql/output/pred/` or as specified in the experiment script. Execution statistics (latency, retries) are dumped at the root (e.g., `stats_bird_benchmark.txt`).

## Evaluation
We have added evaluation scripts for different benchmarks. For instance,
```
poetry run bash lc_nl2sql/scripts/process_bird_output.sh bird_benchmark_flash lc_nl2sql/output/pred/
```
will output 
```
Processing file: lc_nl2sql/output/pred//bird_benchmark_flash
start calculate
                     simple               moderate             challenging          total               
count                925                  464                  145                  1534                
====================================== Exec Accuracy =====================================
accuracy             72.00                57.76                53.79                65.97               
===========================================================================================
Finished evaluation
```

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'google.generativeai'`
This error occurs if the dependencies are not installed or if you are running the script outside the poetry environment.
**Solution:**
Ensure you have installed dependencies and run the script using `poetry run`:
```bash
poetry install
poetry run python ...
```

#### 2. KVPress `window_size` Error
If you encounter errors related to `window_size` when using KVPress (e.g., `KVPress generation failed: window_size must be provided`), it means the KVPress instance requires an explicit window size configuration.
**Solution:**
For GPUs with sufficient VRAM (e.g., RTX 3090 24GB), we recommend setting `window_size=4096` in `lc_nl2sql/llm_base/offline_model.py` during the `FinchPress` initialization:
```python
self.kvpress_instance = FinchPress(compression_ratio=0.4, window_size=4096)
```
