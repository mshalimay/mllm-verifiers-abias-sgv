# fmt:off

import sys

from offline_experiments.prompts.common_prompts import common_prompts

# ===============================================
# LINK: System Prompt parts
# ===============================================
sys_prompts_verifier={
"base": f"""You are an intelligent agent tasked with supervising an assistant utilizing a computer to accomplish computer-based tasks. Your job is to evaluate the assistant's work, and provide feedback so it can progress towards the objective.
\n
## Here's the information you'll have:
### The objective: This is the task the assistant is trying to complete.{{image_info}}{{trace_info}}{{summ_info}}{{k_info}}
\n
## Assistant's capabilities: To effectively analyze the assistant's work, consider the actions it can perform. These actions fall into the following categories:
### Mouse Actions:
- click(start_box='<|box_start|>(x1,y1)<|box_end|>'): Left clicks at the (x1,y1) coordinates.
- left_double(start_box='<|box_start|>(x1,y1)<|box_end|>'): Left double click at the (x1,y1) coordinates.
- right_single(start_box='<|box_start|>(x1,y1)<|box_end|>'): Right click at the (x1,y1) coordinates.
- drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>'): Drags the element at the (x1,y1) coordinates to the (x3,y3) coordinates.
- scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left'): Scrolls the mouse wheel in the specified direction.

### Keyboard Actions:
- hotkey(key=''): Press a specific key.
- type(content=''): Types the string following `content`. If followed by "\n", it means the input is submitted.
- wait(): Pauses for 5s and takes a new screenshot to check for any changes.

### Completion Action:
- finished(content='<|content|>'): This action is issued when the task is believed to be complete. `<content>` is optional, and is used to provide a reason why the task is complete.{{privileged_hint}}
{{rule}}""",
}


trace_infos = {
"utt": "\n### The execution trace: This is a sequence of screenshots of the computer screen paired with the assistant's responses, detailing the computer navigation so far.",
"actions": "\n### The execution trace: This is a sequence of screenshots of the computer screen paired with the assistant's actions, detailing the computer navigation so far.",
"last_u": "\n### The execution trace: This is a sequence of screenshots of the computer screen ending with the assistant's last response, detailing the computer navigation so far.",
"img_only": "\n### The execution trace: This is a sequence of screenshots of the computer screen, detailing the computer navigation so far.",
"actions_img_expert": "\n### The execution trace: This is a sequence of screenshots of the computer screen paired with the assistant's actions, detailing the computer navigation so far.\n### Image Annotations: These are descriptions of images contained in the webpage screenshots of the corresponding state. Important: the annotations are not perfect, so use them as a reference but make sure to check the images yourself for accurate information.",
"utt_img_expert": "\n### The execution trace: This is a sequence of screenshots of the computer screen paired with the assistant's responses, detailing the computer navigation so far.\n### Image Annotations: These are descriptions of images contained in the webpage screenshots of the corresponding state. Important: the annotations are not perfect, so use them as a reference but make sure to check the images yourself for accurate information.",
}

image_infos = {
"som": "\n### Screenshots: These are visual captures of the computer screen taken at specific states of the navigation process. Each bounding box and its respective id shares the same color.",
"coord": "\n### Screenshots: These are visual captures of the computer screen taken at specific states of the navigation process. The red markers, if present, indicates the destination of mouse actions taken at the corresponding state.",
"som_coord": "\n### Screenshots: These are visual captures of the computer screen taken at specific states of the navigation process. Each bounding box and its respective id shares the same color. The red markers, if present, indicates the destination of mouse actions taken at the corresponding state.",
"raw": "\n### Screenshots: These are visual captures of the computer screen taken at specific states of the navigation process.",
}

k_infos = {
"general": "\n### General Computer Knowledge: This is a general description of how tasks like this are typically accomplished on a computer.",
"screenshot_desc": "\n### Screenshot Descriptions: These are textual descriptions of the screenshots taken during the navigation process."
}

rules = {
"rule_1p": """\n\n## To be successful, it is very important to consider the following:
1. You must not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
3. Your evaluation should be based on task completion. Don't worry if the execution was inneficient, contained unnecessary actions, or other factors besides the objective completion.
4. Give utmost importance to the visual information provided, especially the screenshots in the execution trace.""",

"rule_1p_web_search": """\n\n## To be successful, it is very important to consider the following:
1. You must not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
3. Your evaluation must be based strictly on **task completion**. Don't worry if the execution was inneficient, contained unnecessary actions, or other factors besides the objective completion.
4. Give utmost importance to the visual information provided, especially the screenshots in the execution trace.
5. You must first research with the web search tool to find information that can support your judgement.""",

"rule_1p_k": """\n\n## To be successful, it is very important to consider the following:
1. You must not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
3. Your evaluation must be based strictly on **task completion**. Don't worry if the execution was inneficient, contained unnecessary actions, or other factors besides the objective completion.
4. Give utmost importance to the visual information provided, especially the screenshots in the execution trace.
5. Come up with a general description of how tasks like this are typically accomplished on computers.
6. Use this General Computer Knowledge as a guide. It may contain noise or information not relevant to your specific context. Therefore, you should selectively use it to guide your analysis, reasoning, and judgment, making sure to **consider the context of the specific task and computer navigation given to you.**""",

"rule_2p_k": """\n\n## To be successful, it is very important to consider the following:
1. You must not assume the assistant's work is correct or incorrect beforehand. You should come up with your own opinion based on the information provided.
2. You must connect the dots between the objective and the information provided.
4. Give utmost importance to the visual information provided, especially the screenshots in the execution trace.
5. Your evaluation must be based strictly on **task completion**. Don't worry if the execution was inneficient, contained unnecessary actions, or other factors besides the objective completion.
6. The General computer knowledge can contain noise or information not relevant to your specific context. Therefore, you should selectively use it to guide your analysis, reasoning, and judgment, making sure to **consider the context of the specific task and computer navigation given to you.**""",
}


sys_prompt_parts = {
    "trace_infos": trace_infos,
    "image_infos": image_infos,
    "rules": rules,
}

sys_prompts_expert={
"k_2p_expert": 
"""You are an expert on computer-based workflows. Your job is to provide a comprehensive and **detailed** description of how tasks like the ones provided to you are typically accomplished on computers.
\n
## Here's the information you'll have:
### Objective: This is an english description of the task that must be accomplished.
### Screenshots: These are screenshots of the computer screen, giving you context to further understand what must be accomplished.""",

"k_2p_expert_web_search": 
"""You are an expert on web navigation. Your job is to provide a comprehensive and **detailed** description of how tasks like the ones provided to you are typically accomplished on the web.
\n
## Here's the information you'll have:
### Objective: This is an english description of the task, possibly accompanied by images that provide more context on what must be accomplished.
### Screenshots: These are screenshots of the webpage, giving you context of the state of the navigation for the task at hand.

**IMPORTANT**: 
- Consider researching with the web search tool to support your response.
""",

"k_2p_vis": 
f"""You are a visual expert with deep knowledge in web navigation. You will be given a sequence of screenshots representing a navigation on the web. Your job is to provide a **comprehensive** and **detailed** description of the navigation.
\n
## Here's the information you'll have:
### Objective: This is an english description of a web task associated with the navigation, possibly accompanied by images that provide more context on what must be accomplished.{{image_info}}{{trace_info}}""",
}

# ===============================================
# LINK: CoT formats
# ===============================================
cot_phrase = common_prompts["cot_phrase"]
cot_parts = {
"no_cot": "",
    
"basic_cot": f"""
{cot_phrase}
""",
    
"desc_no_cot": f"""
EXECUTION TRACE UNDERSTANDING: [Carefully analyze each screenshot and comprehensively describe the state of the computer screens. Connect the dots between the screenshots and the assistant's work]
""",

"desc_img_no_cot": f"""
IMAGE UNDERSTANDING: [Detailed analysis of each screenshot to understand the computer navigation. Identify and describe elements that are relevant for your evaluation]
""",

"desc_cot": f"""
EXECUTION TRACE UNDERSTANDING: [Carefully analyze each screenshot and comprehensively describe the state of the computer screens. Connect the dots between the screenshots and the assistant's work]
{cot_phrase}
""",

"desc_img_cot": f"""
IMAGE UNDERSTANDING: [Detailed analysis of each screenshot to understand the computer navigation. Identify and describe elements that are relevant for your evaluation]
{cot_phrase}
""",
    
"desc_compare": f"""
EXECUTION TRACE UNDERSTANDING: [Carefully analyze each screenshot and comprehensively describe the state of the computer screens. Connect the dots between the screenshots and the assistant's work]
COMPARISON: [Compare the assistant's work with the general computer knowledge. Make a step by step comparison of what the assistant did and what is expected]
RELEVANCE: [Which of the steps missing, if any, are relevant in this context? If the steps are not relevant, explain why]
{cot_phrase}
""",

"desc_compare_no_cot": f"""
EXECUTION TRACE UNDERSTANDING: [Carefully analyze each screenshot and comprehensively describe the state of the computer screens. Connect the dots between the screenshots and the assistant's work]
COMPARISON: [Compare the assistant's work with the general computer knowledge. Make a step by step comparison of what the assistant did and what is expected]
RELEVANCE: [Which of the steps missing, if any, are relevant in this context? If the steps are not relevant, explain why]
""",
    
"desc_img_compare": f"""
IMAGE UNDERSTANDING: [Detailed analysis of each screenshot to understand the computer navigation. Identify and describe elements that are relevant for your evaluation]
COMPARISON: [Compare the assistant's work with the General computer knowledge. Make a step by step comparison of what the assistant did and what is expected]
RELEVANCE: [Which of the steps missing, if any, are relevant in this context? If the steps are not relevant, explain why]
{cot_phrase}
""",

"desc_img_compare_no_cot": f"""
IMAGE UNDERSTANDING: [Detailed analysis of each screenshot to understand the computer navigation. Identify and describe elements that are relevant for your evaluation]
COMPARISON: [Compare the assistant's work with the General computer knowledge. Make a step by step comparison of what the assistant did and what is expected]
RELEVANCE: [Which of the steps missing, if any, are relevant in this context? If the steps are not relevant, explain why]
""",
    
}

# ===============================================
# LINK: Evaluation request
# ===============================================
response_format = common_prompts["response_format"]
eval_prompt_template = common_prompts["eval_prompt_template"]

# ===============================================
# LINK: Evaluation request
# ===============================================
response_format = common_prompts["response_format"]
eval_prompt_template = common_prompts["eval_prompt_template"]

# ===============================================
# LINK: K extraction parts
# ===============================================
k_extract_1p = """GENERAL COMPUTER KNOWLEDGE: [Description of how tasks like this are typically accomplished on computers]"""
k_extract_2p_no_expert = """Please first provide the following: 
[Description of how tasks like this are typically accomplished on computers]"""
k_extract_2p_expert = """Now please provide your response: how tasks like this are typically accomplished on computers?"""

# K injection
k_injection = """## General computer knowledge:
{k}"""


k_prompt_parts = {
"k_1p": {
    "extraction": k_extract_1p,
    "injection": k_injection,
    "sys_prompt": sys_prompts_verifier["base"],
    "sys_k_info": k_infos["general"],
},

"k_1p_web_search": {
    "extraction": k_extract_1p,
    "injection": k_injection,
    "sys_k_info": k_infos["general"],
},

"k_2p": {
    "extraction": k_extract_2p_no_expert,
    "injection": k_injection,
    "sys_prompt": sys_prompts_verifier["base"],
    "rules": rules["rule_2p_k"],
    "sys_k_info": k_infos["general"],
},

"k_2p_expert": {
    "extraction": k_extract_2p_expert,
    "injection": k_injection,
    "sys_prompt": sys_prompts_expert["k_2p_expert"],
    "sys_k_info": k_infos["general"],
},

"k_2p_expert_web_search": {
    "extraction": k_extract_2p_expert,
    "injection": k_injection,
    "sys_prompt": sys_prompts_expert["k_2p_expert_web_search"],
    "sys_k_info": k_infos["general"],
},

"k_2p_vis": {
    "extraction": """Now please provide your response:\n[A careful description of the navigation]""",
    "injection": """## Screenshot descriptions:\n{k}""",
    "sys_prompt": sys_prompts_expert["k_2p_vis"],
    "sys_k_info": k_infos["screenshot_desc"],
}
}

prompts_osw = {    
    "eval_criterias": common_prompts["eval_criterias"],
    "eval_prompt_templates": common_prompts["eval_prompt_templates"],
    "response_format": common_prompts["response_format"],
    "eval_prompt_template": common_prompts["eval_prompt_template"],
}


# Add all variables to prompts_vwa, except built-ins and imported modules
_current_module = sys.modules[__name__]
for _var, _val in list(vars(_current_module).items()):
    if (
        not _var.startswith("_")
        and _var.isidentifier()
        and _var not in {"common_prompts", "sys", "prompts_vwa"}  # Exclude imports and self
        and not callable(_val)
    ):
        prompts_osw[_var] = _val

del _current_module
