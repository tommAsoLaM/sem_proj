import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

class OfflineModel:
    def __init__(self, model_name:str = "meta-llama/Llama-3.2-1B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_model()

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
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            temperature = 0.1,
            max_new_tokens = 512,
            do_sample = False
        )