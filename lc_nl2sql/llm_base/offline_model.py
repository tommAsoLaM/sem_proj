import pickle
import sqlite3
import logging
import re
import os
import torch
import pandas as pd
from typing import Any, Dict, Generator, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, HfArgumentParser

from lc_nl2sql.configs.config import (CHECKER_TEMPLATE, LITERAL_ERROR_TEMPLATE,
                                      MAJORITY_VOTING, VERIFY_ANSWER)
from lc_nl2sql.configs.data_args import DataArguments
from lc_nl2sql.configs.model_args import FinetuningArguments, GeneratingArguments, ModelArguments
from lc_nl2sql.llm_base.model import BaseModel

# Check for Flash Attention
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logging.warning("Flash Attention not found. Install with 'pip install flash-attn --no-build-isolation'.")

# Import KVPress and desired Policy
# Ensure kvpress library is installed or in path
try:
    # CHANGE: Import ChunkPress and KnormPress (needed as base for ChunkPress)
    from kvpress import FinchPress, ExpectedAttentionPress
    KVPRESS_AVAILABLE = True
except ImportError:
    KVPRESS_AVAILABLE = False
    logging.warning("KVPress library not found. KVPress features will be disabled.")

# Logging Configuration
logging.basicConfig(level=logging.INFO)

class OfflineModel(BaseModel):
    def __init__(self, model_name:str = "Qwen/Qwen3-4B-Instruct-2507") -> None:
        # Initialize local model variables
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # [NEW] Variables for KVPress
        self.kvpress_instance = None
        self.kvpress_policy = None
        self.use_kvpress = True  # NEW: Enable/disable KVPress globally
        self.compression_ratio = 0.4
        
        
        # Default config
        self.temperature = 0
        self.ignore_hints = False
        self.use_self_correction = True
        self.use_disambiguation = True
        self.db_folder_path = ""
        self.db_tbl_col_vals_file = ""
        
        # Load model immediately during initialization
        # This prevents 'NoneType' error on tokenizer later
        self.load_model()

    def load_model(self):
        """
        Loads Hugging Face model to GPU.
        """
        # Check if already loaded, skip to save time
        if self.model is not None and self.tokenizer is not None:
            return

        print(f"Loading local model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Fix for Llama models that sometimes don't have pad_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Determine attention implementation
            attn_impl = "flash_attention_2" if FLASH_ATTN_AVAILABLE else None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16, # Use float16 to save memory
                trust_remote_code=True,
                attn_implementation=attn_impl # Add this argument
            )
            
            # Initialize KVPress Wrapper on Model
            task_name = "text-generation"
            if KVPRESS_AVAILABLE:
                task_name = "kv-press-text-generation"
                
            
            self.pipeline = pipeline(
                task_name,
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            # Do not raise fatal error, set None to be handled in chat()
            self.model = None
            self.tokenizer = None

    def _infer_args(self, args: Optional[Dict[str, Any]] = None):
        parser = HfArgumentParser((ModelArguments, DataArguments,
                                   FinetuningArguments, GeneratingArguments))
        if args:
            self.data_args = DataArguments
            self.generating_args = GeneratingArguments
            self.use_self_correction = args.get("use_self_correction", True)
            self.use_disambiguation = args.get("use_disambiguation", True)
            self.use_column_filtering_for_correction = args.get("use_column_filtering_for_correction", False)
            self.measure_self_correction_tokens = args.get("measure_self_correction_tokens", False)
            self.db_folder_path = args.get("db_folder_path", "")
            self.temperature = args.get("temperature", 0.0)
            self.db_tbl_col_vals_file = args.get("db_tbl_col_vals_file", "db_tbl_col_vals_bird.pickle")
            self.ignore_hints = args.get("ignore_hints", False)
            
            # Get KVPress arguments
            self.use_kvpress = args.get("use_kvpress", True)
            self.kvpress_policy = args.get("kvpress", None)
            self.compression_ratio = args.get("compression_ratio", 0.4)
            logging.info(f"Using KVPress compression: {self.kvpress_policy} with compression ratio: {self.compression_ratio}")
        else:
            (
                model_args,
                self.data_args,
                finetuning_args,
                self.generating_args,
            ) = parser.parse_args_into_dataclasses()
            
            self.temperature = self.generating_args.temperature
            self.use_self_correction = self.generating_args.use_self_correction
            self.use_disambiguation = self.generating_args.use_disambiguation
            self.use_column_filtering_for_correction = self.generating_args.use_column_filtering_for_correction
            self.measure_self_correction_tokens = self.generating_args.measure_self_correction_tokens
            self.db_folder_path = self.data_args.db_folder_path
            self.db_tbl_col_vals_file = self.data_args.db_tbl_col_vals_file
            self.ignore_hints = self.generating_args.ignore_hints            
            self.use_kvpress = self.generating_args.use_kvpress
            self.kvpress_policy = self.generating_args.kvpress
            self.compression_ratio = self.generating_args.compression_ratio
            self._settingKVPress()
            logging.info(f"Using KVPress compression: {self.kvpress_policy} with compression ratio: {self.compression_ratio}")
                
        def _settingKVPress(self):
            if self.kvpress_policy == "FinchPress":
                logging.info(f"Initializing KVPress wrapper: FinchPress")
                    # Ensure window_size is provided as required by FinchPress
                self.kvpress_instance = FinchPress(compression_ratio = self.compression_ratio)
                    
                    # PENTING: Update model & tokenizer agar kenal token delimiter KVPress
                self.kvpress_instance.update_model_and_tokenizer(self.model, self.tokenizer)
                delimiter = self.kvpress_instance.delimiter_token
            elif self.kvpress_policy == "ExpectedAttentionPress":
                self.kvpress_instance = ExpectedAttentionPress(compression_ratio = self.compression_ratio)
                logging.info(f"Initializing KVPress wrapper: ExpectedAttentionPress")
            else:
                logging.info(f"Running without KVPress")
            


        if self.ignore_hints:
            logging.info("*** ignoring hints ***")
        
        # Ensure model is loaded (double check)
        if self.model is None:
            self.load_model()

    def set_temperature(self, temperature = 0):
        self.temperature = temperature
        
    def _count_token(self, prompt):
        # ADAPTATION: Using local tokenizer
        try:
            if self.tokenizer:
                return len(self.tokenizer.encode(prompt))
            return 0
        except Exception as e:
            logging.debug(f"Token counting failed: {e}")
            return 0
        
    def _compress(self, query, multiplier=1.4):
        # LOGIC SAME AS API_MODEL
        pattern = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
        query = re.sub(pattern, "", query)
        n_reduction = 0
        # Character limit adjusted slightly for local context
        while len(query) > 2000000 * multiplier and n_reduction < 8:
            processed_lines = []
            for line in query.splitlines():
                if len(line) > 20000:
                    processed_lines.append(line[:-10000])
                else:
                    processed_lines.append(line)
            query = "\n".join(processed_lines)
            n_reduction += 1
        return query
    
    def _remove_col_vals(self, query):
        # LOGIC SAME AS API_MODEL
        before = len(query)
        if "###Table column example values###" in query:
            prefix = query.split(
                "###Table column example values###")[0]
            postfix = "**************************".join(
                    query.split(
                    "###Table column example values###")[1].split(
                    "**************************")[1:]
                )
            postfix = "**************************" + postfix
        elif "###Examples###" in query:
            prefix = query.split(
                "###Examples###")[0]
            postfix = "**************************".join(
                    query.split(
                    "###Examples###")[1].split(
                    "**************************")[1:]
                )
            postfix = "**************************" + postfix
        elif len(query) >= 10 * 1024 * 1024:
            prefix = query[: 5 * 1024 * 1024]
            postfix = query[: -5 * 1024 * 1024 + 1]
        else:
            prefix, postfix = query, ""
        query = prefix + postfix
        if len(query) == before:
            logging.error(f"Failed to reduce query size from {before}!")
        return query
    
    def _remove_hints(self, query):
        # LOGIC SAME AS API_MODEL
        if query.find("(Hints:") > -1 and self.ignore_hints:
            prefix = query.split(
                "(Hints:")[0]
            postfix = "**************************".join(
                query.split("(Hints:")[1].split(
                "**************************")[1:])
            return prefix + "**************************" + postfix
        return query

    def _generate_sql(self,
                      query,
                      temperature=0.5,
                      use_flash=False,
                      max_retries=5):
        """
        Core text generation function.
        Replaces Google Gemini calls with Hugging Face Pipeline.
        """
        
        # 1. Compression & Preprocessing
        # Keep this on raw string
        query = self._compress(query)
        query = self._remove_hints(query)
        
        # Apply Llama-3 Chat Template here
        # We wrap the cleaned query into a User message format
        try:
            # Extract schema/context if KVPress is enabled
            if self.use_kvpress and self.kvpress_instance and self.kvpress_policy == "FinchPress":
                # Only FinchPress uses delimiter token
                delimiter = self.kvpress_instance.delimiter_token
                # Split query into context and actual query (assuming "###Question###" marks the boundary)
                if "###Question###" in query:
                    context_part = query[:query.find("###Question###")]
                    query_part = query[query.find("###Question###"):]
                    # Insert delimiter between context and query
                    query_with_delimiter = context_part + "\n" + delimiter + "\n" + query_part
                else:
                    # Fallback if no clear boundary
                    query_with_delimiter = query
            elif self.use_kvpress and self.kvpress_instance and self.kvpress_policy == "ExpectedAttentionPress":
                # ExpectedAttentionPress doesn't need delimiter - use query as-is
                query_with_delimiter = query
            else:
                # No KVPress enabled or no policy set
                query_with_delimiter = query
            
            messages = [{"role": "user", "content": query_with_delimiter}]
            
            # tokenize=False so that output remains string (but with <|user|> tags etc)
            # add_generation_prompt=True so the model knows it's its turn to answer (<|assistant|>)
            final_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            logging.warning(f"Failed to apply chat template: {e}. Using raw query.")
            final_prompt = query

        # Check model safety before generation
        if self.pipeline is None:
            logging.error("Pipeline is None. Attempting to reload model.")
            self.load_model()
            if self.pipeline is None:
                return "", 0

        try:
            # 2. Call Local Model
            outputs = None
            
            # [FIX] Fallback Mechanism for KVPress
            if self.use_kvpress and self.kvpress_instance:
                try:
                    logging.info(f"Generating with KVPress instance")
                    
                    # Passed press instance directly to pipeline (kv-press-text-generation)
                    outputs = self.pipeline(
                        final_prompt,
                        max_new_tokens=512,
                        do_sample=True if temperature > 0 else False,
                        temperature=0,
                        top_p=1,
                        return_full_text=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        press=self.kvpress_instance
                    )
                except Exception as e:
                    logging.warning(f"KVPress generation failed: {e}. Falling back to standard generation.")
                    outputs = None
                

            # If outputs is still None, run standard generation
            if outputs is None:
                outputs = self.pipeline(
                    final_prompt,
                    max_new_tokens=512, 
                    do_sample=True if temperature > 0 else False,
                    temperature=1.0,
                    top_p=1.0,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # [CHANGE] Handle output format differences between standard pipeline and KVPress
            if isinstance(outputs, dict) and "answer" in outputs:
                # KVPress returns a dictionary with 'answer' key
                resp = outputs["answer"]
            elif isinstance(outputs, list) and len(outputs) > 0:
                # Standard pipeline returns a list of dicts
                resp = outputs[0].get('generated_text', '')
            else:
                # Fallback
                logging.warning(f"Unexpected output format: {type(outputs)}")
                resp = str(outputs)
            
            # 3. Cleaning Response (Same as Gemini)
            # [MODIFIED] Better extraction logic to handle verbose models
            
            # [NEW CODE START] Prioritize extracting from <FINAL_SQL> tags
            final_sql_match = re.search(r"<FINAL_SQL>\s*(.*?)\s*</FINAL_SQL>", resp, re.DOTALL | re.IGNORECASE)
            if final_sql_match:
                resp = final_sql_match.group(1).strip()
            else:
                # Fallback to old logic if tags are missing (just in case)
                # First, try to extract from markdown code blocks if present
                sql_block = re.search(r"```sql\s*(.*?)\s*```", resp, re.DOTALL | re.IGNORECASE)
                if sql_block:
                    resp = sql_block.group(1)
                else:
                    # If no markdown, try to find SELECT ... ; (non-greedy match)
                    sql_semi = re.search(r"(SELECT.*?;)", resp, re.DOTALL | re.IGNORECASE)
                    if sql_semi:
                        resp = sql_semi.group(1)
                    else:
                        # Fallback: If no semicolon, take SELECT to the end
                        sql_greedy = re.search(r"(SELECT.*)", resp, re.DOTALL | re.IGNORECASE)
                        if sql_greedy:
                            resp = sql_greedy.group(1)
            # [NEW CODE END]

            # Remove common conversational prefixes if they stuck inside
            resp = resp.replace("```sql", "").replace("```", "")
            
            if "<FINAL_ANSWER>" in resp:
                resp = resp.split("<FINAL_ANSWER>")[1].split("</FINAL_ANSWER>")[0]
                
        except torch.cuda.OutOfMemoryError:
            # Memory Error Handling (OOM)
            logging.error("GPU Out of Memory in _generate_sql")
            torch.cuda.empty_cache()
            
            if max_retries > 0:
                logging.info("Retrying with reduced context (removing col vals)...")
                query = self._remove_col_vals(query)
                return self._generate_sql(query, temperature, use_flash, max_retries - 1)
            
            return "", 0 
            
        except Exception as e:
            logging.error(f"Local generation failed: {e}")
            return "", 0

        # 4. Post-processing Regex (Same as Gemini)
        resp = re.sub(r"^ite\s+", "", resp)
        resp = re.sub('\s+', ' ', resp).strip()
        
        # [MODIFIED] Final check to ensure we don't have trailing text after semicolon
        # If the string contains a semicolon, cut everything after it
        if ";" in resp:
            resp = resp.split(";")[0] + ";"

        return resp, max_retries

    def majority_voting(self, query, candidates):
        # LOGIC SAME AS API_MODEL
        should_vote = False
        for c in candidates:
            if c != candidates[0]:
                should_vote = True
                break
        if not should_vote:
            return candidates[0]
        candidates = "\n\n".join([c for c in set(candidates)])
        sql, _ = self._generate_sql(MAJORITY_VOTING.format(input=query,
                                                        candidates=candidates),
                                 use_flash=False)
        return sql
    
    def verify_answer(self, sql, question, schema, use_flash=False):
        # LOGIC SAME AS API_MODEL
        prompt = VERIFY_ANSWER.format(sql=sql, question=question, schema=schema)
        answer, _ = self._generate_sql(prompt, use_flash=use_flash)
        return answer

    def verify_and_correct(self, query, sql, db_folder_path, qid, return_invalid=True, use_flash=False):
        """
        Self-Correction feature that executes SQL on local SQLite database.
        """
        
        if not self.use_self_correction or query == "":
            # ADD, 0 (retry count)
            return sql, 0, 0

        # --- Helper Functions (Same as api_model.py) ---
        def fix_error(s, err):
            try:
                context_str = query[query.find("###Table creation statements###"
                                            ):query.find("###Question###")]
                input_str = query[query.find("###Question###"):query.find(
                    "Now generate SQLite SQL query to answer the given")]
            except:
                context_str = ""
                input_str = query
                
            new_prompt = CHECKER_TEMPLATE.format(context_str, input_str, s,
                                                 err)
            new_sql, _ = self._generate_sql(new_prompt,
                                         use_flash=use_flash,
                                         temperature=self.temperature)
            return new_sql, self._count_token(new_prompt) if self.measure_self_correction_tokens else 0

        def fix_literal_error(s, db_id, tried_sql):
            if s == "":
                return fix_error(s, "INFO:root:")
            
            tbl_col_vals = dict()
            if os.path.exists(self.db_tbl_col_vals_file):
                with open(self.db_tbl_col_vals_file, 'rb') as file:
                    try:
                        data = pickle.load(file)
                        tbl_col_vals = data.get(db_id, dict())
                    except:
                        pass

            def validate_email(email):
                pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
                return re.match(pattern, str(email)) is not None
            
            def extract_table_info(input_string):
                table_info = {}
                tables = re.findall(r"CREATE TABLE (\w+)\s*\((.*?)\);", input_string, re.DOTALL)
                for table_name, table_body in tables:
                    columns = re.findall(r"(\w+)\s+\w+.*?-- examples:\s*(.*?)\s*\|", table_body, re.DOTALL)
                    for column_name, example_values in columns:
                        values = re.findall(r"`(.*?)`", example_values)
                        table_info[f"{table_name}.{column_name}"] = values
                return table_info
            
            def format_col_vals(tbl_col_vals):
                s = ""
                for tbl, col_vals in tbl_col_vals.items():
                    for col, vals in col_vals.items():
                        if len(vals) > 0 and validate_email(vals[0]):
                            continue
                        s += f'* `{tbl}`.`{col}`: [{",".join(str(v) for v in vals[:])}]\n'
                return s
            
            if self.use_column_filtering_for_correction:
                try:
                    df = pd.read_csv(self.data_args.filtered_schema_file)
                    id_name, schema_name = 'question_id', 'selected_schema_with_connections'
                    col_selected_schemas = dict()
                    for k, v in zip(df[id_name], df[schema_name]):
                        col_selected_schemas[int(k)] = v
                    filtered_schema = col_selected_schemas.get(qid, "")
                    if filtered_schema:
                        table_info = extract_table_info(filtered_schema)
                        col_vals = ""
                        for key, value in table_info.items():
                            col_vals += f"{key}: {value}\n"
                    else:
                        col_vals = format_col_vals(tbl_col_vals)
                except:
                    col_vals = format_col_vals(tbl_col_vals)
            else:
                col_vals = format_col_vals(tbl_col_vals)
            
            try:
                context_str = query[query.find("###Table creation statements###"
                                            ):query.find("###Question###")]
                input_str = query[query.find("###Question###"):query.find(
                    "Now generate SQLite SQL query to answer the given")]
            except:
                context_str = ""
                input_str = query

            new_prompt = LITERAL_ERROR_TEMPLATE.format(context_str, col_vals,
                                                       input_str,
                                                       "\n".join(tried_sql))
            new_sql, _ = self._generate_sql(new_prompt,
                                         use_flash=use_flash,
                                         temperature=0.9)
            return new_sql, self._count_token(new_prompt) if self.measure_self_correction_tokens else 0

        def isValidSQL(sql, db_path):
            # EXECUTE SQL IN LOCAL DATABASE
            if not os.path.exists(db_path):
                # If DB path not found, assume valid to avoid getting stuck
                logging.warning(f"DB not found at {db_path}, skipping execution check.")
                return True, "", 0
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            err = ""
            rows = []
            is_valid = True
            try:
                rows = cursor.execute(sql).fetchall()
                if len(rows) == 0:
                    is_valid = False
                    err = "empty results"
            except sqlite3.Warning as warning:
                logging.debug(f"SQLite Warning: {warning}")
                err = str(warning)
                is_valid = False
            except Exception as e:
                logging.debug(e)
                err = str(e)
                is_valid = False
            finally:
                if conn:
                    conn.close()
            return is_valid, err, len(rows)

        # --- Main Correction Logic ---
        try:
            # Finding database path
            db_name = query.split("The database (\"")[1].split("\") structure")[0]
            db_path = os.path.join(db_folder_path, db_name, f"{db_name}.sqlite")
        except:
            logging.warning("Could not parse DB name, skipping execution check.")
            return sql, 0

        accumulated_token_count = 0

        _sql = sql
        retry_cnt, max_retries = 0, 5
        
        # Check if SQL is valid
        valid, err, row_cnt = isValidSQL(_sql, db_path)
        tried_sql = [_sql]
        
        # Correction loop if error
        while not valid and retry_cnt < max_retries:
            print(f"  [Correction] Try {retry_cnt+1}: Error='{err}'")
            if err == "empty results" and self.use_disambiguation:
                _sql, extra_tokens = fix_literal_error(_sql, db_name, tried_sql)
            else:
                _sql, extra_tokens = fix_error(_sql, err)
            accumulated_token_count += extra_tokens
            tried_sql.append(_sql)
            valid, err, row_cnt = isValidSQL(_sql, db_path)
            retry_cnt += 1
            
        if retry_cnt >= max_retries:
            logging.info(f"Correction failed due to {err}: {_sql}")
            if not return_invalid:
                # Add retry_cnt
                return "", accumulated_token_count, retry_cnt
        
        # Add retry_cnt
        return _sql, accumulated_token_count, retry_cnt

    def chat(self,
             query: str,
             history: Optional[List[Tuple[str, str]]] = None,
             system: Optional[str] = None,
             **input_kwargs) -> Tuple[str, Tuple[int, int]]:
        
        # Wrapper for _generate_sql to match chat format
        # IMPORTANT: Always return tuple (str, tuple) to avoid unpack error
        resp = ""
        in_tok = 0
        out_tok = 0
        
        try:
            use_flash = False
            if 'use_flash' in input_kwargs and input_kwargs['use_flash']:
                use_flash = True
            
            resp, _ = self._generate_sql(query,
                                      use_flash=use_flash,
                                      temperature=self.temperature)
            
            # Calculate tokens for report
            in_tok = self._count_token(query)
            out_tok = self._count_token(resp)
            
        except Exception as e:
            print(f'\n*** Error in chat: {e}\n')
            # Still return correct format even if empty
            return "", (0, 0)
            
        return resp, (in_tok, out_tok)

    def stream_chat(self,
                    query: str,
                    history: Optional[List[Tuple[str, str]]] = None,
                    system: Optional[str] = None,
                    **input_kwargs) -> Generator[str, None, None]:
        raise NotImplementedError("Streaming not implemented for OfflineModel.")