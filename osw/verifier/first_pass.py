from io import BytesIO
from pathlib import Path

from PIL import Image

from llms.llm_utils import call_llm
from verifier.prompts import FIRST_PASS_CONTEXT_PROMPT, FIRST_PASS_REQUEST_PROMPT, FIRST_PASS_SYSTEM_PROMPT


def get_prompt_messages(objective: str, screenshot: bytes) -> list:
    prompt_messages = [
        {
            "role": "system",
            "content": FIRST_PASS_SYSTEM_PROMPT.strip(),
        },
        [
            FIRST_PASS_CONTEXT_PROMPT.strip().format(objective=objective),
            Image.open(BytesIO(screenshot)),
        ],
        FIRST_PASS_REQUEST_PROMPT.strip(),
    ]
    return prompt_messages


def get_first_pass_knowledge(objective: str, screenshot: bytes, run_results_path: Path, domain: str, task_id: str) -> str:
    messages = get_prompt_messages(objective, screenshot)
    responses, texts = call_llm(
        gen_kwargs={
            "model": "gemini-2.5-flash",
            "thinking_budget": 0,
        },
        prompt=messages,
        conversation_dir=str(run_results_path / "conversations"),
        usage_dir=str(run_results_path / "usage"),
        call_id=f"{domain}_{task_id}_verifier_first_pass",
        dump_txt=False,
    )
    knowledge = texts[0].text().strip()
    return knowledge
