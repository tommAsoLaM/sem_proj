#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pick Python executable
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

# Paths
INPUT_FILE_SK100="lc_nl2sql/data/bird/dev/dev_processed_sk100.json"
DB_FOLDER_PATH="lc_nl2sql/data/bird/dev/dev_databases"
OUTPUT_DIR="lc_nl2sql/output/pred"
OUTPUT_BASE="${OUTPUT_DIR}/bird_benchmark"

# Ensure required paths exist
[[ -f "$INPUT_FILE_SK100" ]] || { echo "Missing input file: $INPUT_FILE_SK100"; exit 1; }
[[ -d "$DB_FOLDER_PATH" ]] || { echo "Missing DB folder: $DB_FOLDER_PATH"; exit 1; }
mkdir -p "$OUTPUT_DIR"

# Grid
COMPRESSION_RATIOS=(0.2 0.4 0.6 0.8)
POLICIES=("FinchPress" "ExpectedAttentionPress")

TOTAL_TESTS=$((${#POLICIES[@]} * ${#COMPRESSION_RATIOS[@]} + 1))
COMPLETED_TESTS=0
FAILED_TESTS=0
TEST_NUM=1

echo "=========================================="
echo "KVPress Ablation Study"
echo "Total combinations to test: $TOTAL_TESTS"
echo "=========================================="
echo ""

# Baseline: no KVPress
echo "[TEST $TEST_NUM/$TOTAL_TESTS] WITHOUT KVPress (baseline)"
OUTPUT_FILE="${OUTPUT_BASE}_NO_KVPRESS.json"
if "$PY" lc_nl2sql/predict/predict.py \
  --predicted_input_filename "$INPUT_FILE_SK100" \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --db_folder_path "$DB_FOLDER_PATH" \
  --predicted_out_filename "$OUTPUT_FILE" \
  --temperature 0.0 \
  --use_beams 1 \
  --use_kvpress 0; then
  echo "✓ Completed: $OUTPUT_FILE"
  ((COMPLETED_TESTS++))
else
  echo "✗ Failed: $OUTPUT_FILE"
  ((FAILED_TESTS++))
fi
echo ""

# KVPress runs
for POLICY in "${POLICIES[@]}"; do
  for RATIO in "${COMPRESSION_RATIOS[@]}"; do
    ((TEST_NUM++))
    OUTPUT_FILE="${OUTPUT_BASE}_${POLICY}_${RATIO}.json"
    echo "[TEST $TEST_NUM/$TOTAL_TESTS] Policy=${POLICY}, Ratio=${RATIO}"
    if "$PY" lc_nl2sql/predict/predict.py \
      --predicted_input_filename "$INPUT_FILE_SK100" \
      --use_self_correction 1 \
      --use_disambiguation 1 \
      --db_folder_path "$DB_FOLDER_PATH" \
      --predicted_out_filename "$OUTPUT_FILE" \
      --temperature 0.0 \
      --use_beams 1 \
      --use_kvpress 1 \
      --kvpress_policy "$POLICY" \
      --compression_ratio "$RATIO"; then
      echo "✓ Completed: $OUTPUT_FILE"
      ((COMPLETED_TESTS++))
    else
      echo "✗ Failed: $OUTPUT_FILE"
      ((FAILED_TESTS++))
    fi
    echo ""
  done
done

echo "=========================================="
echo "All tests completed!"
echo "Passed: $COMPLETED_TESTS / $TOTAL_TESTS"
echo "Failed: $FAILED_TESTS / $TOTAL_TESTS"
echo "Outputs: $OUTPUT_DIR"
echo "=========================================="

# List outputs
ls -lh "${OUTPUT_BASE}"*.json 2>/dev/null || true