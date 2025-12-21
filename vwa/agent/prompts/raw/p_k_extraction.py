# autopep8: off
# fmt: off
from agent.prompts.raw.common_prompts import text_obs_infos

sys_prompt_k_expert = \
f"""You are an expert on web navigation. Your job is to provide a comprehensive and **detailed** description of how tasks like the ones provided to you are typically accomplished on the web.


## Here's the information you'll have:
### Objective: This is an english description of the task, possibly accompanied by images that provide more context on what must be accomplished.{{image_info}}{{text_obs_info}}{{trace_info}}


{{rules}}"""

rules = {
"raw": "",
"som": "**IMPORTANT**: Do not include in your response a direct solution or actions that are specific to this instantiation of the task. Your description should be comprehensive, detailed, and generally applicable to tasks of this type.",
}

trace_infos= {
"utt": "\n### The execution trace: This is a sequence of webpage screenshots paired with the assistant's responses, detailing the web navigation so far.",
"actions": "\n### The execution trace: This is a sequence of webpage screenshots paired with the assistant's actions, detailing the web navigation so far.",
"img_only": "\n### The execution trace: This is a sequence of webpage screenshots, detailing the web navigation so far.",
"none": "",
}

image_infos = {
"raw": "\n### Screenshots: These are screenshots of the webpage, giving you context of the state of the navigation for the task at hand.",
"som": "\n### Screenshots: These are screenshots of the webpage, giving you context of the state of the navigation for the task at hand.",
"none": "",
}


# Prompt dictionary given at runtime
prompt = {
"rules": rules,
"trace_infos": trace_infos,
"sys_prompt_k_expert": sys_prompt_k_expert,
"image_infos": image_infos,
"text_obs_infos": text_obs_infos,

# Execution history
"intro_execution_history":
"""## Here is the trace of execution so far:
""",

#===============================================
# Objective
#===============================================
"objective_template":
"""## OBJECTIVE:
{objective}""",


#===============================================
# Knowledge retrieval prompt
#===============================================
"k_retrieval_no_expert": """Please first provide the following: 
[Comprehensive and detailed description of how tasks like these are typically accomplished on the web]""",

"k_retrieval_expert": """Now please provide your response: how tasks like these are typically accomplished on the web?""",


# Text to prepend before each observation in the history
"intro_img_obs_history": """STATE `t-{t}` SCREENSHOT:""",

'parsing_error_msg_template':
"""\n\n## ERROR: Your previous response was:\n{response}.\nHowever, the format is incorrect. Please make sure to answer as requested above.""",


# Metadata
"meta_data": {
    "prompt_constructor": "KExtractionPromptConstructor",
    "no_label": True, # [TEXTAREA] element with label 'abc' and content [] -> [TEXTAREA] element with content []
    "splitters": [""],
    "parsing_error_msg_key": "k_extraction_parsing_error",
    },
}
