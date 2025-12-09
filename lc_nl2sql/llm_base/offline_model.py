import logging
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, HfArgumentParser, pipeline, AutoTokenizer
from lc_nl2sql.configs.data_args import DataArguments
from lc_nl2sql.configs.model_args import FinetuningArguments, GeneratingArguments, ModelArguments
from lc_nl2sql.llm_base.model import BaseModel
from typing import Generator, List, Tuple, Any, Optional
import re
from kvpress import KnormPress, SnapKVPress

class OfflineModel(BaseModel):
    def __init__(self, model_name:str = "HuggingFaceTB/SmolLM-135M-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.ignore_hints = False
        self.press = None
        self.load_model()

    def _infer_args(self, args: Optional[Dict[str, Any]] = None)-> None:
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
            self.temperature = args.get("temperature", 0.5)
            self.db_tbl_col_vals_file = args.get("db_tbl_col_vals_file", "db_tbl_col_vals_bird.pickle")
            self.ignore_hints = args.get("ignore_hints", False)
        else:
            (
                model_args,
                self.data_args,
                finetuning_args,
                self.generating_args,
            ) = parser.parse_args_into_dataclasses()
            # Initial generation and error correction uses this.
            # Set to 0.5 by default.
            self.temperature = self.generating_args.temperature
            self.use_self_correction = self.generating_args.use_self_correction
            self.use_disambiguation = self.generating_args.use_disambiguation
            self.use_column_filtering_for_correction = self.generating_args.use_column_filtering_for_correction
            self.measure_self_correction_tokens = self.generating_args.measure_self_correction_tokens
            self.db_folder_path = self.data_args.db_folder_path
            self.db_tbl_col_vals_file = self.data_args.db_tbl_col_vals_file
            self.ignore_hints = self.generating_args.ignore_hints
        if self.ignore_hints:
            logging.info("*** ignoring hints ***")

    def load_model(self):
        print(f"loading the model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("Applying KVPress compression...")
        self.press = KnormPress(compression_ratio=0.4)  

        self.model.eval()
        self.pipeline = pipeline(
            "text-generation",
            model = self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            temperature = 0.1,
            max_new_tokens = 512,
            do_sample = False
        )


    def chat(self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs) -> Tuple[str, Tuple[int, int]]:
        """Generate a response using the local model."""
        try:
            # Construct prompt from components
            full_prompt = ""
            if system:
                full_prompt += f"{system}\n"
            if history:
                for past_query, past_response in history:
                    full_prompt += f"User: {past_query}\nAssistant: {past_response}\n"
            full_prompt += f"User: {query}\nAssistant:"

            # Get token count before generation
            input_tokens = len(self.tokenizer.encode(full_prompt))
                
            # Generate response using local pipeline
            if self.press:
                with self.press(self.model):
                    outputs = self.pipeline(
                        full_prompt,
                        return_full_text=False,
                        **input_kwargs
                    )
            else:
                # Fallback if KVPress is not loaded
                outputs = self.pipeline(
                    full_prompt,
                    return_full_text=False,
                    **input_kwargs
                )
            generated_text = outputs[0]['generated_text']
            print("text generation done")
            sql_match = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", generated_text, re.IGNORECASE)
            if sql_match:
                final_response = sql_match.group(1).strip()
            elif "SELECT" in generated_text.upper():
                # Fallback: simple strip if no code block is found
                final_response = generated_text.strip()
            else:
                final_response = generated_text.strip()
            # Count output tokens
            output_tokens = len(self.tokenizer.encode(final_response))
            print("text checked")
                
            return generated_text, (input_tokens, output_tokens)
        except Exception as e:
            logging.error(f"Local generation error: {str(e)}")
            return final_response, (input_tokens, output_tokens)

    def stream_chat(self,
                   query: str, 
                   history: Optional[List[Tuple[str, str]]] = None,
                   system: Optional[str] = None,
                   **input_kwargs) -> Generator[str, None, None]:
        """Stream responses using local model."""
        try:
            # Use existing pipeline with streaming
            full_prompt = f"{system}\n" if system else ""
            if history:
                for q, a in history:
                    full_prompt += f"User: {q}\nAssistant: {a}\n"
            full_prompt += f"User: {query}\nAssistant:"

            # Generate text in chunks
            for output in self.pipeline(
                full_prompt,
                return_full_text=False,
                max_new_tokens=4,  # Small chunks for streaming
                **input_kwargs
            ):
                yield output[0]['generated_text']
        except Exception as e:
            logging.error(f"Local streaming error: {str(e)}")
            yield ""

    def verify_and_correct(self, 
                          query: str,
                          sql: str,
                          db_folder_path: str,
                          qid: int,
                          return_invalid: bool = True,
                          use_flash: bool = False) -> Tuple[str, int, int]:
        """Verify SQL locally without API calls."""
        verification_prompt = f"""
        Verify this SQL query:
        Question: {query}
        SQL: {sql}
        Check for syntax errors and semantic correctness.
        Provide corrected SQL if needed.
        """
        
        response, (input_tokens, output_tokens) = self.chat(verification_prompt)
        
        sql_match = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", response, re.IGNORECASE)
        if sql_match:
            # Extract the captured group (the content inside the fences)
            corrected_sql = sql_match.group(1).strip()
            
        elif "SELECT" in response.upper():
            # Fallback: If no code block, return the stripped response 
            # (assuming the LLM outputted only the SQL, which is a weak assumption)
            corrected_sql = response.strip()
            
        else:
            # Default to the original SQL if correction failed or model provided no new query
            corrected_sql = sql
            
        return corrected_sql, input_tokens, output_tokens

    def majority_voting(self, query: str, candidates: List[str]) -> str:
        """Local implementation of majority voting."""
        if not candidates:
            return ""
        
        # Simple voting prompt
        voting_prompt = (
            f"Question: {query}\n"
            "Choose the most correct SQL query:\n"
            + "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
            + "\nReturn the number of the best query."
        )
        
        response, _ = self.chat(voting_prompt)
        
        # Try to extract a number from response
        try:
            chosen = int(''.join(filter(str.isdigit, response.strip()))) - 1
            if 0 <= chosen < len(candidates):
                return candidates[chosen]
        except (ValueError, IndexError):
            pass
            
        return candidates[0]  # Default to first candidate