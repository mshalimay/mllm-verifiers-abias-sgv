# Documentation about changes to (Visual)WebArena
This document describes the major changes to the environment, evaluators, and task configurations.
We tried to preserve as much as possible backward compatibility with the original codebases of (Visual)WebArena.

## Changes to the environment and action parsing
The code was substantially refactored to support: 
- Proper parallelization of the Docker environments 
- Reliable autologin during parallel execution
- Configurable website names
- Evaluation from the `.raw` config files directly 

**Explanation**
In the original codebase, URLs are set via env variables and are populated in the `test_config.json` files at initiation. A single `.json` file is created for each task with the updated URLs and paths to cookies. This logic makes it difficulty to deploy several isolated environments / docker containers.

With the refactoring, URLs are tied to their corresponding Docker containers and the autologin scripts creates cookies for each isolated environment. The URLs and path to cookies are dynamically replaced during test loop utilizing a single and centralized `test_config.json` file with all the tasks. 

Finally, one might want to change the website names (e.g.: change 'reddit' to 'forum'). This affects other websites such as `homepage` and the information given by the environment such as in the open tabs. After refactoring, the `homepage` environment is populated with the proper names chosen, and those can be accessed via the `./constants` file during evaluation. 

**Main changes**
0) General modifications to setup
- All paths and configurable constants necessary to deploy the environments are now centralized in `./constants/constants.py`.
- Shell scripts in `scripts/environments`: (i) initialize the containers assigining a unique ID for each; (ii) wait and retry until containers are ready for evaluation.
- The `browser_env/auto_login.py` script resets cookies based on the unique ID of the corresponding environment
- The shell and autologin scripts, as well as the `env.browser.py` have access to the same constants.
- During evaluation, task configurations are dynamically populated based on the constants and environment ID.

1) `browser_env/processors.py` was modified to: 
    - Provide cleaner SoM text representation,
    - Support `click` actions in `select` option elements
    - Associate hidden interactive labels to their corresponding elements such as `radio` buttons (e.g.: for tasks in `reddit` where uploading images requires clicking on them).
    - Provide proper information about open tabs.
    - Storing the bounding boxes separately from screenshots to allow logging of trajectories in other formats.

2) Additional helpers were added, such as `./envs/utils.wait_for_page_to_stabilize` to support dynamic waiting for webpage stabilization and mappers of URLs to local counterpart that are more robust and adpated to the new codebase with dynamic URLs.

3) We enabled support to host the captioner on a common endpoint that all isolated environments can send requests to to minimize GPU load. See `utils_vwa/host_captioner.py`. 
    - Moreover, the code tries to load the captioner in full precision if possible, as we noted that captions are degraded when loading in half or lower precision.

4) Files in  `./environment_docker` were updated to support env parallelization:
- `docker-compose.yml` file was updated to support dynamic waiting before docker initialization and dynamic URLs.
- The homepage templates in `vwebarena-homepage` and `webarena-homepage` were updated to support dynamic URLs and flexible names for the websites.

5) `browser_env/actions.py` was updated to:
- Enable `select`, `updload_file` actions.
- Automatic map `type` action into INPUT fields meant for file upload to `updload_file`
- `click` using SoM representation sometimes can fail as it dependend only on the center of the object; the current tries clicking slightly off-center in cases the actions doesn't trigger a change.
- Enable `clear` after a `type` action and `type` without automatic pressing enter.
- Fix in several small bugs in `press` action parsing. 
- Enabled the use of `press [key_comb] [text]` for actions like `ctrl+f` that requires a text content to search for.
- Enabled actions to return an output. E.g.: number of matches after a `ctrl+f` action
  
Please check the code in `./browser_env` for other changes.


## Changes to Evaluators

### Task configuration files
- Some of the evaluation logic is embedded in the task configs. 
- Check `config_files/vwa_fixed/README.MD` for more details.
- `intent_fix_note`, `eval_fix_note` in the file `config_files/vwa_fixed/test_vwa.raw.json` containing all tasks describes the changes to intent / evaluators in each task, if any.
- All tasks that require target images loads them from local files in `environment_docker/vwebarena-homepage/static/input_images`. 
    - Previously, some task configs were populated to specific URLs to the Web or Docker containers. Retrieving them from local files reduces bugs, waiting, and allow retrieving the target images without the need for a Docker container to be running (useful for offline experimentation).

### Changes to ./evaluation_harness 

### General Changes
- Small code changes were made due to the refactoring explainde in `Changes to the environment`. For instance, the method `evaluation_harness/evaluators/Evaluator.load_configs(config_file)` was added to create the evaluation configuration based off a centralized `test_config.json` file with all the tasks.

- We enable support for other LLMs in the fuzzy match evaluations. See `evaluation_harness/helper_functions/call_llm_fuzzy_match`

### evaluation_harness/helper_functions.py
**`is_selector_present`**
- Added helper function to test for the presence of certain elements in the webpage during evaluation.


**`shopping_get_product_price`**
- Issue: Price parsing was brittle and failed for common locale formats (e.g., "1,234.56" and "1.234,56"), sometimes stripping separators incorrectly or losing decimals with multiple separators.
- Fix:
    - Rewrote the JS normalization logic:
        - Cleaning non-numeric characters in outertext
        - If both , and . exist, treat the rightmost separator as the decimal separator (e.g., "1.234,56" → "1234.56").
        - If only , exists, treat commas as thousands separators (remove all commas).
        - If only . exists, treat a single . as decimal; if multiple ., treat them as thousands separators (remove all).

- Explanation: see "issue" and fix. Examples now handled:
    - "1,234.56" -> 1234.56
    - "1.234,56" -> 1234.56
    - "1,234" -> 1234
    - "1.234.567.89" -> 123456789 (multiple . treated as thousands)

- NOTEs: 
    - When only commas, code defaults to thousands. Eg.: 4,999 -> 499; 5,99 -> 599. This is OK for shopping as it follows US format. Can include an heuristic to treat as thousand if < 3 chars after ",".
    - Use `tests/test_evaluation_harness/test_number_parsing.py` for unit tests.

- Tasks affected: all config files using `func:shopping_get_product_price(__page__)`

```python

# Old function: didn't parse prices like "1,234.56"
def old_shopping_get_product_price(page: Page | PseudoPage) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseFloat(document.querySelector(\"#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span\")
                    .outerText.substr(1));
                    return res ? res : 0;
                }})();
            """
        )
    except Exception:
        result = 0

    return result


# New function
def shopping_get_product_price(page: Page | PseudoPage) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {
                    try {
                        // get the price from the product page
                        const el = document.querySelector("#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span");

                        if (!el) { return 0; }
                        const raw = el.outerText.trim();

                        // replace all non-numeric characters with an empty string
                        const s = raw.replace(/[^\d.,]/g, "");

                        let normalized = s;
                        const hasComma = s.includes(',');
                        const hasPeriod = s.includes('.');

                        // If both commas and periods are present, assume the rightmost is the decimal separator.
                        if (hasComma && hasPeriod) {
                            if (s.lastIndexOf(',') > s.lastIndexOf('.')) {
                                normalized = s.replace(/\./g, '').replace(',', '.');
                            } else {
                                normalized = s.replace(/,/g, '');
                            }
                            
                        // If only commas are present
                        } else if (hasComma && !hasPeriod) {
                            // Always treat "," as a thousands separator.
                            normalized = s.replace(/,/g, '');
                            // Potentially add some heuristic to handle "," as a decimal separator


                        // If only periods are present
                        } else if (hasPeriod && !hasComma) {
                            // If there are multiple periods, they must be thousands separators.
                            if ((s.match(/\./g) || []).length > 1) {
                                normalized = s.replace(/\./g, '');
                            }
                            // Otherwise, it's a single period; assume it's a decimal separator
                            // Potentially add some heuristic to handle "." as a thousands separator
                        }
                        // Else, no comma or period ==> no action is needed
                        
                        const n = parseFloat(normalized);
                        return Number.isFinite(n) ? n : 0;
                    } catch (e) {
                        return 0;
                    }
                })();
            """
        )
    except Exception:
        result = 0

    return result
```
**`reddit_get_latest_comment_obj_by_username`**
Added code block to normalize time comparisons across timezones


**`call_llm_fuzzy_match`**
- Function added to modularize calls to LLMs in fuzzy match evaluation
- Added support for models from Hugging Face, Google, Anthropic 

**`llm_fuzzy_match`**
- Added support for models from Hugging Face, Google, Anthropic
- Refactored prompting to utilize the system prompt and messages format; small changes to the wording. See below for more details.
- Explanation: Small fixes reduced false positives/negatives in the fuzzy match evaluation. 

````python
# Old function
def llm_fuzzy_match_old(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += (
        "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response, response
        return 1.0

# New function
def llm_fuzzy_match(pred: str, reference: str, question: str, provider: str = "openai") -> float:
    """Check whether the prediction matches the reference with LLMs"""

    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."

    messages: list[dict[str, Any]] = []

    if provider == "openai":
        messages = [
            {"role": "system", "inputs": "You are a helpful assistant"},
            {"role": "user", "inputs": message},
        ]

    elif provider == "google":
        messages = [{"role": "system", "inputs": "You are a helpful assistant."}]
        messages.append({"role": "user", "inputs": message})

    elif provider == "huggingface":
        messages = [
            {"role": "system", "inputs": "You are a helpful assistant"},
            {"role": "user", "inputs": message},
        ]
    else:
        raise ValueError(f"Provider {provider} not supported")

    response = call_fuzzy_match_llm(messages, provider)
    print(pred)
    print(reference)
    print(response)
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response, response
        return 1.0
````


**`llm_ua_match`**
- Added support for models from Hugging Face, Google, Anthropic
- Refactored prompting to utilize a system prompt and messages format; small changes to the wording. See below for more details.
- Explanation: Small fixes reduced false positives/negatives in the fuzzy match evaluation. 

```python
# Old function
def llm_ua_match_old(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0


# New function
def llm_ua_match(pred: str, reference: str, question: str, provider: str = "openai") -> float:
    """Check whether the prediction matches the reference with LLMs"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n\n"
    message += f"actual unachievable reason: {reference}\n\n"
    message += f"reported unachievable reason: {pred}\n\n"
    # message += (
    #     "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
    #     "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
    #     "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
    #     "Determine if the reported reason aligns with the actual reason, even if implicitly. "
    #     "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    # )

    sys_prompt = (
        "You will receive a description of a task that is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": message},
    ]

    response = call_fuzzy_match_llm(messages, provider)
```

### evaluation_harness/evaluators.py
**StringEvaluator.clean_answer()**
- Added block to normalize dashes.
```python
answer = re.sub(r"(\w+)[\u2010-\u2015\u2212-](\w+)", r"\1-\2", answer)
```
- Explanation: previous function didn't handle cases like "Seattle–Tacoma International Airport" in reddit_150)
- Note: other normalizations to special characters are possible.

**TrajectoryURLExactEvaluator**
- Added evaluator to check if specific URLs are visited during the navigation.
- Explanation: in tasks like `reddit_160`, the old evaluator just checks if specific elements are NOT present in the final state via program_html. If the agent simply return STOP or just perform random actions, it receives a score of 1. Adding this evaluator checks if the relevant URLs are at least visited during the navigation, reducing false positives.

**StringEvaluator.must_include**
- The following changes were made following `TreeSearch`: https://github.com/kohjingyu/search-agents/blob/7c35ac9eb7fda663d821449efdfd44d360fd0e18/evaluation_harness/evaluators.
```python
# New method
def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
    clean_ref = StringEvaluator.clean_answer(ref)
    clean_pred = StringEvaluator.clean_answer(pred)
    # tokenize the answer if the ref is a single word
    # prevent false positive (e.g, 0)
    if tokenize and len(clean_ref) == 1 and len(word_tokenize(clean_ref)) == 1:
        tok_pred = word_tokenize(clean_pred)
        return float(clean_ref in tok_pred)
    else:
        return float(clean_ref in clean_pred)

# Old method
    def must_include(ref: str, pred: str) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        if len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)
```
### URLExactEvaluator.clean_url(url)
- Added code block for more general URL normalization. 
```python
# Old:
# Replace http://localhost with http://127.0.0.1 to keep things consistent across evals.
# url = url.replace("localhost", "127.0.0.1")

# New:
# The above doesnt work if hosting websites on other machines.
# this will find the common endpoint and map it to 127.0.0.1
url = map_endpoint_to_target(url, "127.0.0.1")
```

## Remaining issues
Some known issues that still remain are described below. Fixes can be provided depending on demand from the community.

- Some tasks could still benefit of changes in intent or evaluators. We did not change those as it can involve more subjectiveness. Examples:
    - Many tasks asks for items or actions on "this page", which might be interpreted as the whole webpage instead of the specific screen provided.
    - Some evaluators are still somewhat "lenient". For instance, tasks like `What is the color of the most expensive item in the "Over-Ear Headphones" category?` requires only answering with "black" to be declared a success
    - The ssim oracle is prone to false positives. For instance, in `reddit_184` returning this image: http://onestopmarket.com/media/catalog/product/B/0/B08YR1465Q.0.jpg (a blank logo of the shopping domain) returns ssim above threshold and higher than ssim between the two target images (which are almost identical).

- The fixes are stable for the `sync` version of the environment. We did not observe relevant performance gains from making the environments async as the webpage calls are relatively light to benefit from concurrency. We can extend upond demand.

- Parallelization for tasks across environments is not hard tested. We recommend running parallel evaluations within VisualWebArena and WebArena, but not mixing both.

- The `reddit` environment exhibit an `invalid csrf token` bug that affects tasks requiring posting comments, or uploading files. If an Agent retries the submission it typically disappears, but we could not find a reliable solution to prevent it from happening from the start.

