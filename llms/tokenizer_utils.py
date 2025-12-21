import tiktoken
from transformers import AutoProcessor, AutoTokenizer

from llms.setup_utils import infer_provider


class Tokenizer(object):
    tokenizer: tiktoken.Encoding | AutoTokenizer | None

    def __init__(self, model_name: str) -> None:
        self.provider: str = infer_provider(model_name)
        if "openai" in self.provider.lower():
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif "huggingface" in self.provider.lower():
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.processor.tokenizer
        elif "google" in self.provider.lower():
            self.tokenizer = None
        elif "anthropic" in self.provider.lower():
            self.tokenizer = None
        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if "openai" in self.provider.lower():
            return self.tokenizer.encode(text)  # type: ignore

        elif "huggingface" in self.provider.lower():
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)  # type: ignore

        elif "google" in self.provider.lower():
            raise ValueError(f"Provider {self.provider} does not provide an official tokenizer.")

        elif "anthropic" in self.provider.lower():
            raise ValueError(f"Provider {self.provider} does not provide an official tokenizer.")

        else:
            raise NotImplementedError(f"Provider {self.provider} not supported")

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if "huggingface" in self.provider.lower():
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)  # type: ignore
        return self.tokenizer.decode(ids)  # type: ignore

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)  # type: ignore
