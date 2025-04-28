import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai

class LLMClient:
    def __init__(self, provider, model_name, api_key=None, device="cpu", **kwargs):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.device = device
        self.kwargs = kwargs

        if self.provider == "transformers":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto" if self.device == "auto" else None)
        elif self.provider == "openai":
            openai.api_key = self.api_key

    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.95, stop=None, **extra):
        if self.provider == "transformers":
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device in ["cuda", "gpu"] else -1
            )
            out = pipe(
                prompt,
                max_length=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **extra
            )
            text = out[0]["generated_text"]
            if stop:
                for s in stop:
                    idx = text.find(s)
                    if idx != -1:
                        text = text[:idx]
            return text.strip()
        elif self.provider == "openai":
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **extra
            )
            return response["choices"][0]["message"]["content"].strip()
        else:
            raise NotImplementedError(f"LLM provider {self.provider} desteklenmiyor.")