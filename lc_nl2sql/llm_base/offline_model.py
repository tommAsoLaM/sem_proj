import logging
from typing import Dict
import torch
from transformers import Any, AutoModelForCausalLM, HfArgumentParser, Optional, pipeline, AutoTokenizer

from lc_nl2sql.configs.data_args import DataArguments
from lc_nl2sql.configs.model_args import FinetuningArguments, GeneratingArguments, ModelArguments
from lc_nl2sql.llm_base.model import BaseModel

class OfflineModel(BaseModel):
    def __init__(self, model_name:str = "meta-llama/Llama-3.2-1B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.ignore_hints = False
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
        device_map="auto",
        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        trust_remote_code=True
        )

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