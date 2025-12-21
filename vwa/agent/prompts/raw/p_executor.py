# autopep8: off
# fmt:off
from agent.prompts.raw.common_prompts import action_space_infos, examples, websites_info
from core_utils.string_utils import partial_format

#------------------------------------------------------------------------------
# System prompt and its components
#------------------------------------------------------------------------------
system_prompt_base = f"""You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

## Here's the information you'll have:
### The objective: This is the task you're trying to complete.
### The webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.
### The text observations: These lists the IDs of all interactable elements on the current webpage frame with their text content if any, in the format [id] [tagType] [text content]. `tagType` is the type of the element, such as button, link, or textbox. `text content` is the text content of the element. For example, [1234] [button] ['Add to Cart'] means that there is a button with id 1234 and text content 'Add to Cart' on the current webpage. [] [StaticText] [text] means that the element is of some text that is not interactable.
### The current webpage's URL: This is the URL of the page you're currently navigating.
### The open tabs: These are the tabs you have open.
### The history of rationales and actions: These are your previous responses and actions taken at each webpage state. Refer to this history to reason about what you have effectively accomplished so far, so you can decide what to do next.{{reflexion_info}}


## The actions you can perform fall into the following categories:
{{action_space_info}}\n\n{{website_info}}

## To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the next action.
4. Generate the action in the correct format. Start with a "In summary, the next action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the next action I will perform is ```some_action [1234]```".
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.{{reflexion_rule}}"""

action_space_info = action_space_infos["base"]
website_info = websites_info["base"]
reflexion_info = """\n## Reflections: These are reflections and notes about failed attempts to this task, which can help you determine what to do next."""
reflexion_rule = """\n6. Reflections, if any, are from executions of this same task that are **independent** of your current execution. Therefore, you must not assume that any actions included in them were performed in your current attempt."""
action_splitter = "```"
args_wrapper = "[]"

#------------------------------------------------------------------------------
# Templates for the current observation and the execution history
#------------------------------------------------------------------------------
intro_txt_obs_history = \
"""\n### STATE `t-{t}`:\n- **URL**: {url}\n- **RATIONALE**: {utterance}\n- **ACTION**: {parsed_action}"""

template_cur_observation = \
"""## RATIONALE AND ACTION HISTORY: {rationale_action_history}
\n\nHere is the current state `t` for you to determine what to do next:
## TEXT OBSERVATION `t`:\n{text_obs}

## URL: {url}

## OBJECTIVE: {objective}"""

#------------------------------------------------------------------------------
# Feedback, reflection, error injection templates
#------------------------------------------------------------------------------
feedback_template = \
"""\n\n## FEEDBACK: Here is your previous response at the current state `t` and **feedback** about it. Use this to revise your response if you deem appropriate.
- **Previous response**: {previous_response}
- **Feedback**: {feedback}"""

reflection_injection_template = """\n\n## REFLECTIONS:\n{reflections}"""

parsing_error_msg_template = \
f"""\n\n## ERROR: Your previous response was:\n{{response}}.\nHowever, the format was incorrect. Ensure the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] {action_splitter}."""

env_parsing_error_msg_template = "\n\n## ERROR in action execution: {error_msg}"


#------------------------------------------------------------------------------
# Prompt dictionary given at runtime
#------------------------------------------------------------------------------
prompt = {
"system_prompt": partial_format(system_prompt_base, action_space_info=action_space_info, website_info=website_info),

"examples": examples["som_executor"],
"reflexion_info": reflexion_info,
"reflexion_rule": reflexion_rule,
"intro_txt_obs_history":intro_txt_obs_history,
"template_cur_observation": template_cur_observation,
"feedback_template": feedback_template,
"reflection_injection_template": reflection_injection_template,
"env_parsing_error_msg_template": env_parsing_error_msg_template,
"parsing_error_msg_template": parsing_error_msg_template,

"meta_data": {
    "observation": "image_som",
    "action_type": "som",
    "prompt_constructor": "ExecutorPromptConstructor",
    "action_splitter": action_splitter,
    "args_wrapper": args_wrapper,
    "history_type": "rationale_action",
    "use_text_observation": True,
    "use_img_observation": True,
    "parsing_error_msg_key": "executor_parsing_error",
},
}
