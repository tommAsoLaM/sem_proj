# KV Cache Compression for Text-to-SQL with LLMs

This project implements and evaluates **KV Cache Compression** techniques (specifically **Finch/KnuthPress** via NVIDIA's KVPress toolkit) for **Text-to-SQL** tasks. The goal is to mitigate the "Lost in the Middle" phenomenon in Large Language Models (LLMs) when handling long prompts containing complex database schemas, instance values, and few-shot examples.

We leverage local LLMs (e.g., **Llama-3.2-1B-Instruct**, **Qwen2.5**) and compare performance against uncompressed baselines on benchmarks like **BIRD**, **Spider**, **KaggleDBQA**, and **Beaver**.

## 📂 Project Structure

```text
lc_nl2sql/
├── configs/            # Configuration files (paths, model args, data args)
├── data_process/       # Scripts for prompt construction and data formatting
├── experiments/        # Shell scripts for reproducible benchmarks and ablation studies
├── llm_base/           # Model wrappers (OfflineModel for local, GeminiModel for API)
├── predict/            # Inference and evaluation logic
├── scripts/            # Utility scripts for data generation and post-processing
└── utils/              # Helper functions
```

## 🚀 Setup

### Prerequisites
*   **OS:** Linux or macOS (with CUDA support recommended for GPU acceleration).
*   **Python:** 3.10+
*   **Package Manager:** [Poetry](https://python-poetry.org/)
*   **Hardware:** GPU with at least 16GB VRAM recommended for running Llama/Qwen models locally.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd sem_proj
    ```

2.  **Install dependencies using Poetry:**
    This will create a virtual environment and install all required packages defined in `pyproject.toml`.
    ```bash
    pip install poetry
    poetry install
    ```

3.  **Install Flash Attention (Optional but Recommended):**
    For faster inference on supported GPUs.
    ```bash
    pip install flash-attn --no-build-isolation
    ```

4.  **Install KVPress:**
    Required for the compression experiments.
    ```bash
    pip install kvpress
    ```

### Data Preparation
Download the benchmark datasets (BIRD, Spider, etc.) and place them in the `lc_nl2sql/data/` directory. The expected structure is:

```text
lc_nl2sql/data/
├── bird/
│   ├── dev/
│   │   ├── dev.json            # Questions and Gold SQLs
│   │   ├── dev_tables.json     # Database Schemas
│   │   └── dev_databases/      # Folder containing actual .sqlite files
│   └── ...
├── spider/
├── kaggle/
└── beaver/
```

---

## ⚙️ Configuration

The project configuration is modularized under [`lc_nl2sql/configs`](lc_nl2sql/configs ):

*   **`config.py`**: Defines global paths (`ROOT_PATH`, `DATA_PATH`, `OUT_DIR`) and dataset-specific settings.
*   **`model_args.py`**: Defines arguments for the model (e.g., `model_name_or_path`, `cache_dir`, `finetuning_type`).
*   **`data_args.py`**: Defines arguments for data processing, prompting strategies, and database paths.

### Key Parameters
When running scripts, you can control the behavior using command-line arguments:

*   `--use_kvpress`: **[NEW]** Enable KV Cache Compression (e.g., SnapKV, Finch).
*   `--model_name_or_path`: Path or HuggingFace ID of the model (e.g., `meta-llama/Llama-3.2-1B-Instruct`).
*   `--num_beams`: Number of retries/candidates for Self-Correction (e.g., `1` for speed, `5` for accuracy).
*   `--use_self_correction`: Enable automatic error correction if the generated SQL fails execution.
*   `--synthetic_examples`: Enable generation of few-shot examples using the LLM.
*   `--num_col_values`: Number of example data values to include in the prompt per column.

---

## 🏃‍♂️ Running the Code

The pipeline consists of two main stages: **Data Processing** and **Prediction**.

### 1. Data Processing (`sql_data_process.py`)
This step reads the raw dataset and constructs the prompts (Instruction + Schema + Examples + Question).

```bash
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path lc_nl2sql/data/bird/dev/dev.json \
  --input_table_path lc_nl2sql/data/bird/dev/dev_tables.json \
  --output_file_path "lc_nl2sql/data/bird/dev/dev_processed.json" \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --num_col_values 3 \
  --use_hint 1 \
  --use_column_filtering 1 \
  --synthetic_examples 1 \
  --num_examples 3
```

### 2. Prediction / Inference (`predict.py`)
This step loads the LLM (and KVPress if enabled), generates SQL queries, executes them against the database, and performs self-correction.

```bash
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "lc_nl2sql/data/bird/dev/dev_processed.json" \
  --num_beams 1 \
  --temperature 0.1 \
  --use_self_correction 1 \
  --use_disambiguation 0 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/bird_results" \
  --finetuning_type lora \
  --stage sft \
  --use_kvpress 1
```

---

## 🧪 Reproducible Runs (Experiments)

We provide shell scripts in [`lc_nl2sql/experiments`](lc_nl2sql/experiments ) to reproduce specific benchmarks and ablation studies.

### Main Benchmarks
Run the full pipeline on the standard development sets.

*   **BIRD:** `poetry run bash lc_nl2sql/experiments/bird_benchmark.sh`
*   **Spider:** `poetry run bash lc_nl2sql/experiments/spider_benchmark.sh`
*   **KaggleDBQA:** `poetry run bash lc_nl2sql/experiments/kaggle_benchmark.sh`
*   **Beaver:** `poetry run bash lc_nl2sql/experiments/beaver_benchmark.sh`

### Ablation Studies & Analysis
Scripts for detailed analysis of model performance.

*   **`bird_ablation_study_EURECOM.sh`**: Comprehensive study testing pipeline components (tables, rules, hints, values, correction).
*   **`bird_lost_in_the_middle.sh`**: Analyzes performance based on the position of relevant information in the prompt.
*   **`bird_self_correction.sh`**: Evaluates the impact of the self-correction module.
*   **`bird_num_col_vals.sh`**: Tests the effect of varying the number of column values.
*   **`bird_example_selection.sh`**: Compares different few-shot example selection strategies.

---

## 📊 Evaluation

After generating predictions, use the evaluation scripts to calculate **Execution Accuracy (EX)** and **Valid Efficiency Score (VES)**.

```bash
# Example for BIRD
poetry run bash lc_nl2sql/scripts/process_bird_output.sh bird_results lc_nl2sql/output/pred/
```

**Output Example:**
```text
Processing file: lc_nl2sql/output/pred//bird_results
start calculate
                     simple               moderate             challenging          total               
count                925                  464                  145                  1534                
====================================== Exec Accuracy =====================================
accuracy             72.00                57.76                53.79                65.97               
===========================================================================================
Finished evaluation
```

## 🛠 Troubleshooting

### 1. `GPU Out of Memory` / `Killed`
*   **Cause:** The model context is too large, or too many threads are running in parallel.
*   **Solution:**
    *   Reduce `--num_beams` to `1`.
    *   Edit [`lc_nl2sql/predict/predict.py`](lc_nl2sql/predict/predict.py ) and set `num_threads = 1`.
    *   Enable KVPress compression (`--use_kvpress`).

### 2. `KVPress generation failed`
*   **Cause:** Input prompt is too short for compression, or library mismatch.
*   **Solution:** The code automatically falls back to standard generation. This is a warning, not a fatal error.

### 3. `ModuleNotFoundError: No module named 'kvpress'`
*   **Solution:** Ensure you installed the library: `pip install kvpress`.