# autopep8: off
# fmt: off
from agent.prompts.raw.common_prompts import action_space_infos, text_obs_infos
from core_utils.string_utils import partial_format

#-------------------------------------------------------------------------------
# System prompt and its components
#-------------------------------------------------------------------------------
sys_prompt_base = f"""You are an intelligent agent tasked with supervising an assistant navigating a web browser to accomplish a web-based task. Your job is to evaluate the assistant's work, and provide feedback so it can progress towards the objective.
\n
## Here's the information you'll have:
### The objective: This is the task the assistant is trying to complete.{{image_info}}{{text_obs_info}}{{trace_info}}{{k_info}}
\n
## Assistant's capabilities: To effectively analyze the assistant's work, consider the actions it can perform. These actions fall into the following categories:
{{action_space_info}}
{{rules}}"""

sgv_rules = """\n\n## To be successful, it is very important to consider the following:
1. You must not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
3. Your evaluation must be based strictly on **task completion**. Don't worry if the execution was inneficient, contained unnecessary actions, or other factors besides the objective completion.
4. Give utmost importance to the visual information provided, especially the screenshots in the execution trace.
5. The General web knowledge can contain noise or information not relevant to your specific context. Therefore, you should selectively use it to guide your analysis, reasoning, and judgment, making sure to **consider the context of the specific task and web navigation given to you.**"""

no_sgv_rules = """\n\n## To be successful, it is very important to consider the following:
1. You must not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
3. Your evaluation must be based strictly on **task completion**. Don't worry if the execution was inneficient, contained unnecessary actions, or other factors besides the objective completion.
4. Give utmost importance to the visual information provided, especially the screenshots in the execution trace."""

k_info = """\n### General web knowledge: This is a general description of how tasks like this are typically accomplished on the web."""

action_space_info = action_space_infos["base_verifier"]

trace_infos= {
"utt": "\n### The execution trace: This is a sequence of webpage screenshots paired with the assistant's responses, detailing the web navigation so far.",
"actions": "\n### The execution trace: This is a sequence of webpage screenshots paired with the assistant's actions, detailing the web navigation so far.",
"img_only": "\n### The execution trace: This is a sequence of webpage screenshots, detailing the web navigation so far.",
"none": "",
}

image_infos = {
"som": """\n### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.""",
"som_dot": """\n### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color. The magenta round markers, if present, indicates the destination of actions taken at the corresponding state.""",
"raw": """\n### Webpage screenshots: These are screenshots of the webpage taken at each step of the navigation.""",
"raw_dot": """\n### Webpage screenshots: These are screenshots of the webpage taken at each step of the navigation. The magenta round markers, if present, indicates the destination of actions taken at the corresponding state.""",
"none": "",
}


#-------------------------------------------------------------------------------
# Evaluation prompt
#-------------------------------------------------------------------------------
eval_prompt_template = """Now please provide your response.
\n
## Here is the evaluation criteria:
{eval_criteria}
\n
## Provide your response as follows:
{response_format}
"""

eval_criteria =\
"""
SUCCESS: The assistant executed **all of** what's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **most of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed.
"""

response_format_sgv=\
"""
REASONING: [Comprehensive step-by-step reasoning over all evidence and information to come up with your evaluation and feedback]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the assistant can progress towards the objective OR "NA" if not applicable]
"""

response_format_no_sgv=\
"""
REASONING: [Comprehensive step-by-step reasoning over all evidence and information to come up with your evaluation and feedback]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the assistant can progress towards the objective OR "NA" if not applicable]
"""

splitters = ["REASONING:", "EVALUATION:", "FEEDBACK:"]

eval_prompt_sgv = eval_prompt_template.format(eval_criteria=eval_criteria.strip(), response_format=response_format_sgv.strip()).strip()
eval_prompt_no_sgv = eval_prompt_template.format(eval_criteria=eval_criteria.strip(), response_format=response_format_no_sgv.strip()).strip()

#-------------------------------------------------------------------------------
# Execution history
#-------------------------------------------------------------------------------
intro_execution_history = """## Here is the trace of execution so far:\n"""
intro_img_obs_history = """STATE `t-{t}` SCREENSHOT:"""
objective_template = """## OBJECTIVE:\n{objective}"""

#-------------------------------------------------------------------------------
# Knowledge and error injection prompt
#-------------------------------------------------------------------------------
knowledge_injection_prompt = """## General Web Knowledge:\n{knowledge_retrieval_response}"""
parsing_error_msg_template = """\n\n## ERROR: Your previous response was:\n{response}.\nHowever, the format is incorrect. Please make sure to answer as requested above."""

#-------------------------------------------------------------------------------
# Prompt dictionary given at runtime
#-------------------------------------------------------------------------------
prompt = {
"system_prompt_sgv": partial_format(sys_prompt_base, k_info=k_info, rules=sgv_rules, action_space_info=action_space_info),
"system_prompt_no_sgv": partial_format(sys_prompt_base, k_info="", rules=no_sgv_rules, action_space_info=action_space_info),
"image_infos": image_infos,
"text_obs_infos": text_obs_infos,
"trace_infos": trace_infos,

"eval_prompt_sgv": eval_prompt_sgv,
"eval_prompt_no_sgv": eval_prompt_no_sgv,

"intro_execution_history": intro_execution_history,
"intro_img_obs_history": intro_img_obs_history,
"objective_template": objective_template,

"knowledge_injection_prompt": knowledge_injection_prompt,
'parsing_error_msg_template': parsing_error_msg_template,

# Metadata
"meta_data": {
    "prompt_constructor": "VerifierPromptConstructor",
    "no_label": True, # [TEXTAREA] element with label 'abc' and content [] -> [TEXTAREA] element with content []
    "splitters": splitters,
    "eval_scores":["SUCCESS", "PARTIAL SUCCESS", "FAILURE"],
    "eval_key": "EVALUATION",
    "feedback_key": "FEEDBACK",
    "success_keyword": "SUCCESS",
    "parsing_error_msg_key": "critique_parsing_error",
    "required_splitters": ["EVALUATION", "FEEDBACK"],
    },
}
