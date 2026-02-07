#!/bin/bash

# ----THIS IS AN EDITED BASH FILE FOR EURECOM SEMESTER PROJECT (RAHMANTO, LA MALFA) INSPIRED BY ORIGINAL GOOGLE CODE----

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Baseline piepeline for other experiments
# 1. Use all tables from DB
# 2. Use hint & rules
# 3. Use self correction 
# 4. 50 col vals

# ablation baseline, add in the folloiwng order
# - all talbes from DB (x -- TODO no self correction, no distinct col vals, no hint, no rules) 
# - rules  (x -- TODO no self correction, no distinct col vals, no hint) 
# - hint (x -- TODO no self correction, no distinct col vals)
# - add distinct column values (o -- no self correction run)
# - self correction (o -- with self correciton run)
#---------------------------------- (eval baseline)
# - add examples (o -- 100 examples)
# - expensive disambiguation (x -- with examples)
# - multiple choice / selection (x)

# Complete piepeline
# 1. Use all tables from DB
# 2. Use hint and rules
# 3. Use 50 (distinct) col vals
# 4. Use self correction 
# 5. Use expensive disambiguation (all dictinct values (str))
# 6. Use 100 synthetic examples (<32k vs. >32k)
# 8. multiple choice and select
echo "Ablation 1. Use all tables from DB"
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path lc_nl2sql/data/bird/dev/dev_trim.json \
  --input_table_path lc_nl2sql/data/bird/dev/dev_tables.json \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --output_file_path lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_1.json \
  --num_col_values 0 \
  --use_hint 0 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 1 \
  --num_examples 0

poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_1.json \
  --num_beams 1 \
  --temperature 0.5 \
  --use_self_correction 0 \
  --use_disambiguation 0 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_1_all_tables"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_1.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_1_all_tables"
echo "---------------------------------------"
echo "Ablation 2. + hint"
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path lc_nl2sql/data/bird/dev/dev_trim.json \
  --input_table_path lc_nl2sql/data/bird/dev/dev_tables.json \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --output_file_path lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_2.json \
  --num_col_values 0 \
  --use_hint 1 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 0 \
  --num_examples 0

poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_2.json \
  --num_beams 1 \
  --temperature 0.5 \
  --use_self_correction 0 \
  --use_disambiguation 0 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_2_hint"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_2.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_2_hint"
echo "---------------------------------------"
echo "Ablation 3. + distinct column values"
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path lc_nl2sql/data/bird/dev/dev_trim.json \
  --input_table_path lc_nl2sql/data/bird/dev/dev_tables.json \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --output_file_path lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --num_col_values 10 \
  --use_hint 1 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 0 \
  --num_examples 0

poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --num_beams 1 \
  --temperature 0.5 \
  --use_self_correction 0 \
  --use_disambiguation 0 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_3_col_values"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_3_col_values"
echo "---------------------------------------"
echo "Ablation 4. + self correction"
# share the data from 3
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --num_beams 1 \
  --temperature 0.5 \
  --use_self_correction 1 \
  --use_disambiguation 0 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_4_self_correction"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_4_self_correction"
echo "---------------------------------------"
echo "Ablation 5. + disambiguation"
# share the data from 3 & 4.
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --num_beams 1 \
  --temperature 0.5 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_5_disambiguation"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_trim_processed_ablation_3.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_5_disambiguation"
echo "---------------------------------------"
echo "Ablation 6. + synthetic examples"
poetry run python lc_nl2sql/data_process/sql_data_process.py \
  --input_data_path lc_nl2sql/data/bird/dev/dev_trim.json \
  --input_table_path lc_nl2sql/data/bird/dev/dev_tables.json \
  --output_file_path lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --num_col_values 10 \
  --use_hint 1 \
  --use_rules 0 \
  --use_column_filtering 0 \
  --synthetic_examples 1 \
  --num_examples 100

poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --num_beams 1 \
  --temperature 0.5 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_6_synthetic_examples"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_6_synthetic_examples"
echo "---------------------------------------"
echo "Ablation 7. + verify & retry"
# share the data from 6.
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --num_beams 10 \
  --temperature 0.5 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_7_verify_retry"

poetry run python lc_nl2sql/predict/count_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_7_verify_retry"

poetry run python lc_nl2sql/predict/count_verify_token.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/token_count/bird_ablation_7_verify_retry_verify"

echo "---------------------------------------"
echo "Ablation for without hints"
poetry run python lc_nl2sql/predict/predict.py \
  --predicted_input_filename lc_nl2sql/data/bird/dev/ablation_2/dev_example_synthetic_examples_100.json \
  --num_beams 10 \
  --temperature 0.5 \
  --use_self_correction 1 \
  --use_disambiguation 1 \
  --ignore_hints 1 \
  --db_folder_path lc_nl2sql/data/bird/dev/dev_databases \
  --predicted_out_filename "lc_nl2sql/output/pred/ablation_2/bird_ablation_without_hints"
