#!/bin/bash


# 1. Input Directories
BASE_INPUT_DIR="lc_nl2sql/data/bird/dev"
DB_PATH="${BASE_INPUT_DIR}/dev_databases"
INPUT_DATA="${BASE_INPUT_DIR}/dev.json"
INPUT_TABLES="${BASE_INPUT_DIR}/dev_tables.json"

# 2. Intermediate Directory (Place to store processed JSON data)
# Adjust the ablation folder name here (e.g., ablation_2 or ablation_finch)
PROCESSED_DIR="${BASE_INPUT_DIR}/ablation"

# 3. Output Directories
# Adjust the main output folder name here
OUT_DIR="lc_nl2sql/output/pred/ablation"
TOKEN_DIR="${OUT_DIR}/token_count"

# 4. Looping Variables
POLICIES=("FinchPress" "ExpectedAttentionPress")
RATIOS=(0.2 0.4 0.6 0.8)

# ----------------------------------------------------
# INITIAL SETUP
# ----------------------------------------------------

echo "========================================================"
echo " CONFIGURATION "
echo "========================================================"
echo "DB Path      : ${DB_PATH}"
echo "Process Dir  : ${PROCESSED_DIR}"
echo "Output Dir   : ${OUT_DIR}"
echo "Token Dir    : ${TOKEN_DIR}"
echo "Policies     : ${POLICIES[*]}"
echo "Ratios       : ${RATIOS[*]}"
echo "========================================================"

# Create output directories if they don't exist (prevent FileNotFoundError)
mkdir -p "${PROCESSED_DIR}"
mkdir -p "${OUT_DIR}"
mkdir -p "${TOKEN_DIR}"

# ----------------------------------------------------
# ABLATION STEPS
# ----------------------------------------------------

# ==============================================================================
echo "---------------------------------------"
echo "Ablation 1. Use all tables from DB"

PROC_FILE_1="${PROCESSED_DIR}/dev_processed_ablation_1.json"

# [PROCESS DATA - STEP 1]
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path "${INPUT_DATA}" \
  --input_table_path "${INPUT_TABLES}" \
  --db_folder_path "${DB_PATH}" \
  --output_file_path "${PROC_FILE_1}" \
  --num_col_values 0 \
  --use_hint 0 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 1 \
  --num_examples 0

# [PREDICT LOOP - STEP 1]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 1 | $POLICY | $RATIO"
    
    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_1}" \
      --num_beams 1 \
      --temperature 0 \
      --use_self_correction 0 \
      --use_disambiguation 0 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_1_all_tables${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_1}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_1_all_tables${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 1]
echo ">> Ablation 1 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_1}" \
  --num_beams 1 \
  --temperature 0 \
  --use_self_correction 0 \
  --use_disambiguation 0 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_1_all_tables_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_1}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_1_all_tables_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation 2. + hint"

PROC_FILE_2="${PROCESSED_DIR}/dev_processed_ablation_2.json"

# [PROCESS DATA - STEP 2]
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path "${INPUT_DATA}" \
  --input_table_path "${INPUT_TABLES}" \
  --db_folder_path "${DB_PATH}" \
  --output_file_path "${PROC_FILE_2}" \
  --num_col_values 0 \
  --use_hint 1 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 0 \
  --num_examples 0

# [PREDICT LOOP - STEP 2]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 2 | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_2}" \
      --num_beams 1 \
      --temperature 0 \
      --use_self_correction 0 \
      --use_disambiguation 0 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_2_hint${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_2}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_2_hint${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 2]
echo ">> Ablation 2 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_2}" \
  --num_beams 1 \
  --temperature 0 \
  --use_self_correction 0 \
  --use_disambiguation 0 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_2_hint_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_2}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_2_hint_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation 3. + distinct column values"

PROC_FILE_3="${PROCESSED_DIR}/dev_processed_ablation_3.json"

# [PROCESS DATA - STEP 3]
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path "${INPUT_DATA}" \
  --input_table_path "${INPUT_TABLES}" \
  --db_folder_path "${DB_PATH}" \
  --output_file_path "${PROC_FILE_3}" \
  --num_col_values 10 \
  --use_hint 1 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 0 \
  --num_examples 0

# [PREDICT LOOP - STEP 3]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 3 | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_3}" \
      --num_beams 1 \
      --temperature 0 \
      --use_self_correction 0 \
      --use_disambiguation 0 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_3_col_values${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_3}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_3_col_values${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 3]
echo ">> Ablation 3 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_3}" \
  --num_beams 1 \
  --temperature 0 \
  --use_self_correction 0 \
  --use_disambiguation 0 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_3_col_values_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_3}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_3_col_values_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation 4. + self correction"
# (Use PROC_FILE_3 from the previous step)

# [PREDICT LOOP - STEP 4]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 4 | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_3}" \
      --num_beams 1 \
      --temperature 0 \
      --use_self_correction 1 \
      --use_disambiguation 0 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_4_self_correction${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_3}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_4_self_correction${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 4]
echo ">> Ablation 4 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_3}" \
  --num_beams 1 \
  --temperature 0 \
  --use_self_correction 1 \
  --use_disambiguation 0 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_4_self_correction_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_3}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_4_self_correction_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation 5. + disambiguation"
# (Still using PROC_FILE_3 from the previous step)

# [PREDICT LOOP - STEP 5]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 5 | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_3}" \
      --num_beams 1 \
      --temperature 0 \
      --use_self_correction 1 \
      --use_disambiguation 1 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_5_disambiguation${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_3}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_5_disambiguation${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 5]
echo ">> Ablation 5 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_3}" \
  --num_beams 1 \
  --temperature 0 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_5_disambiguation_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_3}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_5_disambiguation_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation 6. + synthetic examples"

PROC_FILE_6="${PROCESSED_DIR}/dev_example_synthetic_examples_100.json"

# [PROCESS DATA - STEP 6]
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path "${INPUT_DATA}" \
  --input_table_path "${INPUT_TABLES}" \
  --output_file_path "${PROC_FILE_6}" \
  --db_folder_path "${DB_PATH}" \
  --num_col_values 10 \
  --use_hint 1 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 1 \
  --num_examples 100

# [PREDICT LOOP - STEP 6]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 6 | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_6}" \
      --num_beams 1 \
      --temperature 0 \
      --use_self_correction 1 \
      --use_disambiguation 1 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_6_synthetic_examples${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_6}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_6_synthetic_examples${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 6]
echo ">> Ablation 6 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_6}" \
  --num_beams 1 \
  --temperature 0 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_6_synthetic_examples_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_6}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_6_synthetic_examples_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation 7. + verify & retry"
# (Using PROC_FILE_6 from the previous step)

# [PREDICT LOOP - STEP 7]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation 7 | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
      --predicted_input_filename "${PROC_FILE_6}" \
      --num_beams 10 \
      --temperature 0 \
      --use_self_correction 1 \
      --use_disambiguation 1 \
      --kvpress "$POLICY" \
      --compression_ratio $RATIO \
      --db_folder_path "${DB_PATH}" \
      --predicted_out_filename "${OUT_DIR}/bird_ablation_7_verify_retry${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_token.py \
      --predicted_input_filename "${PROC_FILE_6}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_7_verify_retry${SUFFIX}"

    poetry run python lc_nl2sql/predict/count_verify_token.py \
      --predicted_input_filename "${PROC_FILE_6}" \
      --predicted_out_filename "${TOKEN_DIR}/bird_ablation_7_verify_retry_verify${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - STEP 7]
echo ">> Ablation 7 | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename "${PROC_FILE_6}" \
  --num_beams 10 \
  --temperature 0 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --compression_ratio 0 \
  --db_folder_path "${DB_PATH}" \
  --predicted_out_filename "${OUT_DIR}/bird_ablation_7_verify_retry_without_KVPress"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename "${PROC_FILE_6}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_7_verify_retry_without_KVPress"

poetry run python lc_nl2sql/predict/count_verify_token.py \
  --predicted_input_filename "${PROC_FILE_6}" \
  --predicted_out_filename "${TOKEN_DIR}/bird_ablation_7_verify_retry_verify_without_KVPress"


# ==============================================================================
echo "---------------------------------------"
echo "Ablation for without hints"
# (Using PROC_FILE_6 from the previous step, ignore hints)

# [PREDICT LOOP - NO HINTS]
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${RATIOS[@]}"; do
    SUFFIX="_${POLICY}_${RATIO}"
    echo ">> Ablation No Hints | $POLICY | $RATIO"

    poetry run python lc_nl2sql/predict/predict.py \
     --predicted_input_filename "${PROC_FILE_6}" \
     --num_beams 10 \
     --temperature 0 \
     --use_self_correction 1 \
     --use_disambiguation 1 \
     --kvpress "$POLICY" \
     --compression_ratio $RATIO \
     --ignore_hints 1 \
     --db_folder_path "${DB_PATH}" \
     --predicted_out_filename "${OUT_DIR}/bird_ablation_without_hints${SUFFIX}"
  done
done

# [PREDICT WITHOUT KVPRESS - NO HINTS]
echo ">> Ablation No Hints | without KVPress"
poetry run python lc_nl2sql/predict/predict.py \
 --predicted_input_filename "${PROC_FILE_6}" \
 --num_beams 10 \
 --temperature 0 \
 --use_self_correction 1 \
 --use_disambiguation 1 \
 --compression_ratio 0 \
 --ignore_hints 1 \
 --db_folder_path "${DB_PATH}" \
 --predicted_out_filename "${OUT_DIR}/bird_ablation_without_hints_without_KVPress"

echo "All Jobs Completed."