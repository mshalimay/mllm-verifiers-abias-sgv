# autopep8: off
# fmt: off


sys_prompt_k_expert = f"""You are an expert on web navigation. Your job is to provide a comprehensive and **detailed** description of how tasks like the ones provided to you are typically accomplished on the web.
\n
## Here's the information you'll have:
### Objective: This is an english description of the task, possibly accompanied by images that provide more context on what must be accomplished.
### Screenshots: These are screenshots of the webpage, giving you context of the state of the navigation for the task at hand."""


k_retrieval_no_expert = """
Please first provide the following: 
[Comprehensive and **detailed** description of how tasks like this are typically accomplished on the web]
"""

k_retrieval_expert = """Now please provide your response: how tasks like this are typically accomplished on the web?"""


eval_criteria = """Now please provide your response.
\n
## Here is the evaluation criteria:
SUCCESS: The assistant executed **all** that's necessary to complete the objective. The task is fully accomplished.
PARTIAL SUCCESS: The assistant executed **part of** what's necessary to complete the objective. The task is partially accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed."""

eval_criteria_binary = """Now please provide your response.
\n
## Here is the evaluation criteria:
SUCCESS: The assistant executed **all** that's necessary to complete the objective. The task is fully accomplished.
FAILURE: The assistant executed **mostly incorrect** steps. The task is not accomplished, and major revisions are needed."""


web_knowledge_info = "\n### General web knowledge: This is a general description of how tasks like this are typically accomplished on the web."
web_knowledge_rule = "\n5. Use the General web knowledge to guide your judgment, but also consider the context of the specific task given to you."


response_format_basicCoT = """## Provide your response as follows:
REASONING: [Comprehensive step-by-step reasoning over all evidence and information to come up with your evaluation and feedback]
EVALUATION: [Your evaluation following the evaluation criteria]
FEEDBACK: [Feedback so the assistant can progress towards the objective or NONE if not applicable]"""


image_infos = {
"som": """\n### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.""",
"som_dot": """\n### Webpage screenshots: These are screenshots of the webpage, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color. The red markers, if present, indicates the destination of actions taken at the corresponding state.""",
"raw": """\n### Webpage screenshots: These are screenshots of the webpage taken at each step of the navigation.""",
"raw_dot": """\n### Webpage screenshots: These are screenshots of the webpage taken at each step of the navigation. The colored dot markers, if present, indicate the destination of actions taken at the corresponding state.""",
"none": "",
}


text_obs_infos = {
"som": """\n### The text observations: These list the IDs of all interactable elements of the corresponding webpage frame with their text content if any, in the format [id] [tagType] [text content]. `tagType` is the type of the element, such as button, link, or textbox. `text content` is the text content of the element. For example, [1234] [button] ['Add to Cart'] means that there is a button with id 1234 and text content 'Add to Cart' on the current webpage. [] [StaticText] [text] means that the element is of some text that is not interactable.""",
"none": "",
}


action_space_infos = {

"base":
    
"""### Page Operation Actions:
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
```stop [answer]```: Issue this action if you believe the task is complete or infeasible. If the objective is to find a text-based answer, provide the answer in the bracket. If you deem the task is infeasible, provide a reason why.""",

"base_verifier":
"""### Page Operation Actions:
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
```stop [answer]```: This action is issued when the task is believed to be complete or deemed as infeasible. If the task requires a text-based answer, the answer is provided within the brackets.""",

}


websites_info = {
"base":
"""
## Homepage:
If the objective requires visiting other specific websites, check out the homepage at http://homepage.com. It has a list of websites you can visit.
""",
}

# Example format: list[example1, example2, ...], where example = list[state1, state2, ...], where state = dict[text input, image input, model output (text), model output (image)]

examples ={
"som_executor":
[
# EXAMPLE 1
[
# state 1
{
"text": """
## RATIONALE AND ACTION HISTORY: None


Here is the current state `t` for you to determine what to do next:
## TEXT OBSERVATION `t`:
Tab 0 (current): Postmill

[] [StaticText] [/f/food]
[] [StaticText] [[homemade] Obligatory Halloween Pumpkin Loaf!	Submitted by	kneechalice	t3_yid9lu	1 year ago]
[9] [IMG] []
[] [StaticText] [Submitted by	kneechalice	t3_yid9lu	1 year ago]
[10] [A] [kneechalice]
[11] [A] [45 comments]
[] [StaticText] [[I ate] Maple Pecan Croissant	Submitted by	AccordingtoJP	t3_y3hrpn	1 year ago]
[14] [IMG] []
[] [StaticText] [Submitted by	AccordingtoJP	t3_y3hrpn	1 year ago]
[15] [A] [AccordingtoJP]
[16] [A] [204 comments]

## URL: http://reddit.com

## OBJECTIVE: Tell me what the top comment on the croissant post says.""",
"image": "agent/prompts/som_examples/som_example2.png",
"utterance": "Let's think step-by-step. This page has a post titled '[I ate] Maple Pecan Croissant', which is the post mentioned in the objective. In order to find the top comment, I will navigate into the comments section of the post. In summary, the next action I will perform is ```click [11]```",
},
# state 2 ...
],

# EXAMPLE 2
[
{
"text": """
## RATIONALE AND ACTION HISTORY: None


Here is the current state `t` for you to determine what to do next:
## TEXT OBSERVATION `t`:
Tab 0 (current): Classifieds

[] [StaticText] [What are you looking for today?]
[5] [INPUT] []
[6] [SELECT] [Select a category]
[7] [BUTTON] [Search]
[] [StaticText] [Latest Listings]
[] [StaticText] [Atlas Powered Audio System w/ Tripod	150.00 $	Music instruments	Borough of Red Lion  (Pennsylvania)	2023/11/16]
[8] [IMG] [Atlas Powered Audio System w/ Tripod]
[9] [A] [Atlas Powered Audio System w/ Tripod]
[] [StaticText] [150.00 $]
[] [StaticText] [Neptune Gaming Console	350.00 $	Video gaming	Pennwyn  (Pennsylvania)	2023/11/16]
[10] [IMG] [Neptune Gaming Console]
[11] [A] [Neptune Gaming Console]
[] [StaticText] [350.00 $]

## URL: http://classifieds.com

## OBJECTIVE: Help me find the cheapest dark colored guitar.""",
"image": "agent/prompts/som_examples/som_example3.png",
"utterance": 'Let\'s think step-by-step. The objective is to find the cheapest dark colored guitar on the site. The site has a search box whose ID is [5]. I can search for guitars by entering "guitar". I can submit this by pressing the Enter afterwards. In summary, the next action I will perform is ```type [5] [guitar] [1]```',
},
],

[
# EXAMPLE 3
{
"text": """
## RATIONALE AND ACTION HISTORY:
### STATE `t-1`:
- **URL**: http://onestopmarket.com/
- **RATIONALE**: Let's think step-by-step. My objective is to find the price of the HP Inkjet Fax Machine. The current page lacks this information. Analyzing the current observation, I see a search box whose ID is [7]. I can use the website's search functionality to navigate to the product's page where I may find its price. I will execute the search by typing 'HP Inkjet Fax Machine' into the search bar. In summary, the next action I will perform is ```type [7] [HP Inkjet Fax Machine] [1]```.
- **ACTION**: type [7] [HP Inkjet Fax Machine] [1] where [7] is [INPUT] element with content []


Here is the current state `t` for you to determine what to do next:
## TEXT OBSERVATION `t`:
Tab 0 (current): Search results for: 'HP Inkjet Fax Machine'

[31] [IMG] [Image, description: hp fx-7010dn fax machine, url: http://onestopmarket.com/media/catalog/product/cache/89ff578b9cd87e0600daac45c9e1ea98/B/0/B08GKZ3ZKD.0.jpg]
[32] [A] [HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)]
[] [StaticText] [$279.49]
[33] [BUTTON] [Add to Cart]
[34] [A] [Add to Wish List]
[35] [A] [Add to Compare]

## URL: http://onestopmarket.com/office-products/office-electronics.html

## OBJECTIVE: What is the price of HP Inkjet Fax Machine?""",
"image": "agent/prompts/som_examples/som_example1.png",
"utterance": "Let's think step-by-step. Previously I used the search bar and the current page displays the search results. The page lists the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I think I have achieved the objective. I will issue the stop action with the answer. In summary, the next action I will perform is ```stop [$279.49]```",
},
],
],

"verifier":[],

}
