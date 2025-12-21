# autopep8: off
# fmt: off
from agent.prompts.raw.common_prompts import image_infos, text_obs_infos
from core_utils.string_utils import partial_format

#===============================================
# Base prompts
#===============================================
sys_prompt_base =f"""You are an autonomous intelligent agent that was tasked with navigating a web browser to complete web-based tasks. 
Your goal now is to analyze your behavior on a previous task attempt whose execution is known to be **not successful**. You must analyze the strategy and path you took when trying to complete the task, understand the reason why you failed, and devise **concise** reflections that accounts for your mistake and can be helpful when you are solving the same task in the future.

## Here's the information you'll have:
### The objective: This is the task you were trying to complete.{{image_info}}{{text_obs_info}}{{trace_info}}{{k_info}}
### Previous reflections: This is a list of previous reflections you've made about other failed executions of this task.

## To analyze your past task attempts, consider the actions that were available to you. These actions fall into the following categories:
### Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content] [enter_after]```: Use this to type the content into the field with id. If `enter_after` is 0, the "Enter" key is not pressed after typing; otherwise, the "Enter" key is automatically pressed.
```hover [id]```: Hover over an element with id.
```press [key_comb] [text]```: Press a keyboard key or key combination (e.g., delete, ctrl+a) with optional text input if the key combination requires it (e.g., ```press [ctrl+f] [some_text]```).
```scroll [down]``` or ```scroll [up]```: Scroll the webpage up or down.

### Tab Management Actions:
```new_tab```: Open a new, empty browser tab.
```tab_focus [tab_index]```: Switch the browser's focus to a specific tab using its index.
```close_tab```: Close the currently active tab.

### URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

### Completion Action:
```stop [answer]```: Issue this action if you believe the task is complete or infeasible. If the objective is to find a text-based answer, provide the answer in the bracket. If you deem the task is infeasible, provide a reason why.
{{rules}}"""


trace_infos= {
"utt": "\n### The execution trace: This is a sequence of webpage screenshots paired with your responses, detailing the web navigation from the beginning to the end of the previous task attempt.",
"actions": "\n### The execution trace: This is a sequence of webpage screenshots paired with your actions, detailing the web navigation from the beginning to the end of the previous task attempt.",
"img_only": "\n### The execution trace: This is a sequence of webpage screenshots, detailing the web navigation from the beginning to the end of the previous task attempt.",
"none": "",
}

# System prompt parts
k_info = """\n### General web knowledge: This is a general description of how tasks like this are typically accomplished on the web."""

no_sgv_rules = """\n\n## To be successful, it is **very important** to consider the following:
1. You should carefully analyze the execution trace to come up with your response.
2. You must connect the dots between the objective and the information provided.
3. The execution trace is from a previous task attempt that is **finished**. Therefore, you should not try to continue the task nor propose actions continuing from the previous task attempt.
4. Prioritize key and general aspects. Your reflections should be applicable to future attempts to this task that will be **independent** from previous task attempts and will start from the beginning.
5. The information provided is from a previous task attempt that does not accomplish the objective. You must strive to understand possible reasons for why the objective was not achieved, even if not immediately apparent from the execution trace.
6. Try to to think differently from previous task attempts and reflections. Integrate information from previous reflections that can be pertinent for future task attempts while thinking of possible new strategies that can lead to a successful completion of this task."""

sgv_rules = no_sgv_rules + "\n7. The General web knowledge can contain noise or information not relevant to your specific context. Therefore, you should selectively use it to guide your analysis and reasoning, making sure to **consider the context of the specific task and web navigation given to you.**"


reflection_request = \
"""Now please provide your response as follows:

POSSIBLE REASONS FOR FAILURE: [What are the possible reasons why the objective was not achieved?]
REFLECTION: [reflections for future task attempts]"""


splitters = ["REASONING", "POSSIBLE REASONS FOR FAILURE", "REFLECTION"]
required_splitters = ["REFLECTION"]
reflection_key = "REFLECTION"

#=========================================================
# Observation and other prompt parts given at runtime
#=========================================================
# Observation
use_img_observation = True
use_text_observation = False


#===============================================
# Prompt dictionary given at runtime
#===============================================
prompt = {

# System prompts
"system_prompt_sgv": partial_format(sys_prompt_base, k_info=k_info, rules=sgv_rules),
"system_prompt_no_sgv": partial_format(sys_prompt_base, k_info="", rules=no_sgv_rules),

"image_infos": image_infos,
"text_obs_infos": text_obs_infos,
"trace_infos": trace_infos,

"reflection_request_sgv": reflection_request,
"reflection_request_no_sgv": reflection_request,

# Execution history
"intro_execution_history":
"""## Here is the trace of execution:
""",

# Objective
"objective_template":
"""## OBJECTIVE:
{objective}""",


# Knowledge injection prompt
"knowledge_injection_prompt":"""## General Web Knowledge:
{knowledge_retrieval_response}""",

"reflections_injection_template":"""## Previous Reflections:{previous_reflections}""",

# Text to prepend before each observation in the history
"intro_img_obs_history": """STATE `t-{t}` SCREENSHOT:""",

'parsing_error_msg_template':
"""\n\n## ERROR: Your previous response did not follow the format requested above. Please make sure to answer as requested above.""",

# Metadata
"meta_data": {
    "prompt_constructor": "ReflexionPromptConstructor",
    "no_label": True, # [TEXTAREA] element with label 'abc' and content [] -> [TEXTAREA] element with content []
    "splitters": splitters,
    "required_splitters": required_splitters,
    "parsing_error_msg_key": "reflexion_parsing_error",
    "reflection_key": reflection_key,
    },
}
