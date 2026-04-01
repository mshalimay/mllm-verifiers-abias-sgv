
import os
import requests
from agents.vlm.vlm_helper import format_prompt, read_text_file, get_image_list

class OpenAIVLM:
    def __init__(self, model_name:str="gpt-4o", temperature_set:float=0, max_tokens:int=8192):
        self.model_name = model_name
        self.temperature_set = temperature_set
        self.max_tokens = max_tokens

    def check_api_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it in your shell, e.g. "
                "`export OPENAI_API_KEY=...`, then retry."
            )
        return api_key

    def get_response(self, text_prompt, image_list):
        formatted_prompt = format_prompt(text_prompt, image_list)

        api_key = self.check_api_key()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": self.model_name,
            "temperature": self.temperature_set,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        # Robust parsing: OpenAI errors don't include "choices"
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                f"OpenAI API returned non-JSON (status {response.status_code}): {response.text[:1000]}"
            )

        if response.status_code >= 400:
            err = data.get("error", data)
            raise RuntimeError(f"OpenAI API error (status {response.status_code}): {err}")

        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"Unexpected OpenAI response (missing 'choices'): {data}")

        msg = choices[0].get("message", {})
        content = msg.get("content")
        if content is None:
            raise RuntimeError(f"Unexpected OpenAI response (missing message.content): {data}")
        return content
    
    def get_response_from_file(self, text_prompt_file, image_list_file):
        text_prompt = read_text_file(text_prompt_file)
        image_list = get_image_list(image_list_file)
        return self.get_response(text_prompt, image_list)

