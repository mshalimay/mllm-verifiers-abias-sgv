import re
from enum import StrEnum
from io import BytesIO
from pathlib import Path

from PIL import Image

from llms.llm_utils import call_llm
from verifier.prompts import (
    SECOND_PASS_EXECUTION_TRACE_PROMPT,
    SECOND_PASS_KNOWLEDGE_PROMPT,
    SECOND_PASS_OBJECTIVE_PROMPT,
    SECOND_PASS_REQUEST_PROMPT_CALL_USER,
    SECOND_PASS_REQUEST_PROMPT_NORMAL,
    SECOND_PASS_SYSTEM_PROMPT_CALL_USER,
    SECOND_PASS_SYSTEM_PROMPT_NORMAL,
)


class Evaluation(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL SUCCESS"
    FAILURE = "FAILURE"
    INFEASIBLE = "INFEASIBLE"


def get_prompt_messages(objective: str, screenshots: list[bytes], thoughts: list[str], first_pass_knowledge: str, prompt_type: str) -> list:
    system_prompt = SECOND_PASS_SYSTEM_PROMPT_CALL_USER if prompt_type == "call_user" else SECOND_PASS_SYSTEM_PROMPT_NORMAL
    prompt_messages = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        },
        SECOND_PASS_OBJECTIVE_PROMPT.strip().format(objective=objective),
        SECOND_PASS_EXECUTION_TRACE_PROMPT,
    ]
    for i, (screenshot, thought) in enumerate(zip(screenshots, thoughts)):
        prompt_messages.append(
            [
                f"STATE `t-{len(thoughts) - i}` SCREENSHOT:",
                Image.open(BytesIO(screenshot)),
            ]
        )
        prompt_messages.append(
            {
                "role": "assistant",
                "content": thought.strip(),
            }
        )
    request_prompt = SECOND_PASS_REQUEST_PROMPT_CALL_USER if prompt_type == "call_user" else SECOND_PASS_REQUEST_PROMPT_NORMAL
    if first_pass_knowledge:
        prompt_messages.extend(
            [
                SECOND_PASS_KNOWLEDGE_PROMPT.strip().format(knowledge=first_pass_knowledge),
                request_prompt.strip(),
            ]
        )
    else:
        prompt_messages.append(request_prompt.strip())
    return prompt_messages


def parse_response_text(response_text: str) -> tuple[Evaluation, str]:
    pattern = re.compile(r"^([A-Z ]+):\s*(.*?)(?=^[A-Z ]+:|\Z)", flags=re.DOTALL | re.MULTILINE)
    sections = {match[1]: match[2].strip() for match in pattern.finditer(response_text)}
    evaluation = Evaluation(sections["EVALUATION"])
    feedback = sections["FEEDBACK"]
    return evaluation, feedback


def get_second_pass_evaluation(
    objective: str, screenshots: list[bytes], thoughts: list[str], first_pass_knowledge: str, prompt_type: str, run_results_path: Path, domain: str, task_id: str
) -> tuple[Evaluation, str, str]:
    messages = get_prompt_messages(objective, screenshots, thoughts, first_pass_knowledge, prompt_type)
    responses, texts = call_llm(
        gen_kwargs={
            "model": "gemini-2.5-flash",
            "thinking_budget": 0,
        },
        prompt=messages,
        conversation_dir=str(run_results_path / "conversations"),
        usage_dir=str(run_results_path / "usage"),
        call_id=f"{domain}_{task_id}_verifier_second_pass",
        dump_txt=False,
    )
    text = texts[0].text().strip()
    evaluation, feedback = parse_response_text(text)
    return evaluation, feedback, text
