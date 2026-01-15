#!/bin/bash

# -----------------------------------------------------
# Template Slurm Submit Script
# Python - Environment ALELEON (GPU) | rev.081025
#
# NOTES:
# 1. Isi bagian yang ditandai 4 garing (////).
# 2. Template ini bersifat referensi.
#    User dapat mengubah bagian yang perlu diubah.
# -----------------------------------------------------

# -----------------------------------------------------
# Spesifikasi job (SBATCH)
# -----------------------------------------------------

# Partisi GPU
#SBATCH --partition=ampere

# Jumlah core thread CPU, minimal 2
#SBATCH --ntasks=2

# Jumlah GPU, minimal 1
#SBATCH --gpus=1

# Jumlah memori RAM (satuan GB)
#SBATCH --mem=64GB

# Limit waktu job (HH:MM:SS atau D-HH:MM:SS)
# ex: 1 hari -> 24:00:00 atau 1-00:00:00
#SBATCH --time=04:00:00

# File output terminal, %j merekap no ID job
#SBATCH --output=result-%j.txt

# File output log status dan error (bila ada)
#SBATCH --error=error-%j.txt

# ----------------------------------------------------
# Script jalannya program
# ----------------------------------------------------

# Input isi kolom "Aktivasi script" env pilihan user
# wiki.efisonlt.com/wiki/Daftar_Environment_Python_ALELEON
# Pastikan env mendukung GPU di kolom dukungan job
AICompilation_r0.1

# Input perintah komputasi Python user
# Tentukan folder input (sesuaikan path ini jika perlu)
# Pastikan variabel di bawah ini mengarah ke folder yang benar:
#!/bin/bash

# -----------------------------------------------------
# Template Slurm Submit Script
# Python - Environment ALELEON (GPU) | rev.081025
# -----------------------------------------------------

# -----------------------------------------------------
# Spesifikasi job (SBATCH)
# -----------------------------------------------------

#SBATCH --partition=ampere
#SBATCH --ntasks=2
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00  # Waktu disesuaikan (24 jam) karena banyaknya proses
#SBATCH --output=result-%j.txt
#SBATCH --error=error-%j.txt

# ----------------------------------------------------
# Script jalannya program
# ----------------------------------------------------

AICompilation_r0.1

#!/bin/bash

# -----------------------------------------------------
# Template Slurm Submit Script
# Python - Environment ALELEON (GPU) | rev.081025
# -----------------------------------------------------

#SBATCH --partition=ampere
#SBATCH --ntasks=2
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=result-%j.txt
#SBATCH --error=error-%j.txt

# ----------------------------------------------------
# KONFIGURASI PATH & VARIABEL
# ----------------------------------------------------

AICompilation_r0.1

# 1. Input Directories
BASE_INPUT_DIR="lc_nl2sql/data/bird/dev"
DB_PATH="${BASE_INPUT_DIR}/dev_databases"
INPUT_DATA="${BASE_INPUT_DIR}/dev_trim.json"
INPUT_TABLES="${BASE_INPUT_DIR}/dev_tables.json"

# 2. Intermediate Directory (Tempat simpan hasil process data json)
# Sesuaikan nama folder ablation di sini (misal: ablation_2 atau ablation_finch)
PROCESSED_DIR="${BASE_INPUT_DIR}/ablation"

# 3. Output Directories
# Sesuaikan nama folder output utama di sini
OUT_DIR="lc_nl2sql/output/pred/ablation"
TOKEN_DIR="${OUT_DIR}/token_count"

# 4. Looping Variables
POLICIES=("FinchPress" "ExpectedAttentionPress")
RATIOS=(0.2 0.4 0.6 0.8)

# ----------------------------------------------------
# SETUP Awal
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

# Membuat folder output jika belum ada (mencegah error FileNotFoundError)
mkdir -p "${PROCESSED_DIR}"
mkdir -p "${TOKEN_DIR}"

# ----------------------------------------------------
# ABLATION STEPS
# ----------------------------------------------------

# ==============================================================================
echo "---------------------------------------"
echo "Ablation 1. Use all tables from DB"

PROC_FILE_1="${PROCESSED_DIR}/dev_trim_processed_ablation_1.json"

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

PROC_FILE_2="${PROCESSED_DIR}/dev_trim_processed_ablation_2.json"

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

PROC_FILE_3="${PROCESSED_DIR}/dev_trim_processed_ablation_3.json"

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
# (Menggunakan PROC_FILE_3 dari step sebelumnya)

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
# (Masih menggunakan PROC_FILE_3)

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
# (Menggunakan PROC_FILE_6)

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
# (Menggunakan PROC_FILE_6, ignore hints)

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