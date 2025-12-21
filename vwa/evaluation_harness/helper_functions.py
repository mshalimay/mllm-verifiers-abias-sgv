"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""

import json
from datetime import datetime, timezone
from typing import Any, Union
from urllib.parse import urlparse

import numpy as np
import requests
from beartype.typing import Dict, List
from browser_env.env_config import ACCOUNTS, Website
from browser_env.env_utils import get_local_url_norm
from PIL import Image
from playwright.sync_api import CDPSession, Page
from skimage.metrics import structural_similarity as ssim

from core_utils.logger_utils import logger
from llms.llm_utils import call_llm
from llms.setup_utils import infer_provider

# from agent.prompt_constructor import create_llama3_chat_input


class PseudoPage:
    def __init__(self, original_page: Page, url: str):
        self.url = url
        self.original_page = original_page

    def __getattr__(self, attr: str) -> Any:
        # Delegate attribute access to the original page object
        if attr not in ["url"]:
            return getattr(self.original_page, attr)
        else:
            return getattr(self, attr)


def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{get_local_url_norm(Website.SHOPPING)}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS[Website.SHOPPING_ADMIN]["username"],
                "password": ACCOUNTS[Website.SHOPPING_ADMIN]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(f"{get_local_url_norm(Website.SHOPPING)}/rest/V1/orders", params=params, headers=header)
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{get_local_url_norm(Website.SHOPPING)}/sales/order/view/order_id/{order_id}/"
    return order_url


def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{get_local_url_norm(Website.SHOPPING)}/rest/V1/products/{sku}/reviews", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{get_local_url_norm(Website.SHOPPING)}/rest/V1/products/{sku}/reviews", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


def shopping_get_sku_latest_review_text(sku: str) -> str:
    """Get the latest review text for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{get_local_url_norm(Website.SHOPPING)}/rest/V1/products/{sku}/reviews", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    text: str = response_obj[-1]["detail"]
    return text


def shopping_get_sku_latest_review_title(sku: str) -> str:
    """Get the latest review title for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{get_local_url_norm(Website.SHOPPING)}/rest/V1/products/{sku}/reviews", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    title: str = response_obj[-1]["title"]
    return title


def shopping_get_sku_product_page_url(sku: str) -> str:
    """Get product page url from sku"""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{get_local_url_norm(Website.SHOPPING)}/rest/V1/products/{sku}", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    for custom_attributes in response_obj["custom_attributes"]:
        if custom_attributes["attribute_code"] == "url_key":
            return f"{get_local_url_norm(Website.SHOPPING)}/{custom_attributes['value']}.html"
    return ""


def shopping_get_all_product_order(
    page: Page | PseudoPage,
) -> List[Dict[str, str]]:
    """
    Get info of all product in a given order page.

    Example output:
    [
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Strawberry, 19oz Box (12 Bars)\nSize\n12 Count (Pack of 1)",
            "options": {
                "Size": "12 Count (Pack of 1)"
            },
            "sku": "B00MXUFL0E",
            "price": "$24.50",
            "qty": "Ordered2",
            "subtotal": "$49.00"
        },
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Chocolatey Chip Cookie Dough, 19oz Box (12 Bars)",
            "sku": "B07ZD2PB9F",
            "price": "$42.30",
            "qty": "Ordered2",
            "subtotal": "$84.60"
        }
    ]
    """
    try:
        result = page.evaluate(
            f"""
(() => {{
    try {{
        const products = [...document.querySelector("#my-orders-table").getElementsByTagName('tbody')].map(
            (x) => {{
                return [...x.getElementsByTagName('td')].reduce(function(obj, y) {{
                    const key = y.className.split(' ')[1];
                    obj[key] = y.outerText;
                    // check if options exist
                    if (key === 'name' && y.querySelector('dl')) {{
                        var option_dict = {{}}
                        const options = [...y.querySelector('dl').children];
                        for (let i = 0; i < options.length; i += 2) {{
                            option_dict[options[i].outerText] = options[i+1].outerText;
                        }}
                        obj['options'] = option_dict;
                    }}
                    return obj;
                }}, {{}})
            }}
        );
        return products;
    }} catch (e) {{
        // If any errors are caught, return an empty string
        return e;
        return [];
    }}
}})();
            """
        )
        return result
    except Exception as e:
        result = []

    return result


def shopping_get_order_product_name_list(page: Page | PseudoPage) -> str:
    try:
        products = shopping_get_all_product_order(page)

        return " |OR| ".join([p["name"] for p in products])
    except Exception:
        return ""


def shopping_get_order_product_quantity(page: Page | PseudoPage, sku: str) -> int:
    try:
        if "|OR|" in sku:
            skus = sku.split(" |OR| ")
        else:
            skus = [sku]

        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() in skus:
                # Ordered{qty}
                return int(product["qty"][7:])
        return 0
    except Exception:
        return 0


def shopping_get_order_product_option(page: Page | PseudoPage, sku: str, option_name: str) -> str:
    try:
        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() == sku:
                # Ordered{qty}
                return product["options"][option_name]
        return ""
    except Exception as e:
        return ""


def shopping_get_product_attributes(page: Page | PseudoPage, attribute: str) -> str:
    # Get the values of all cells in the table for the given attribute
    try:
        result = page.evaluate(
            f"""
                (() => {{
                try {{
                    // Create an array of search terms, splitting the string by ' |OR| '
                    const searchTerms = '{attribute}'.toLowerCase().split(' |or| ');
                    // Convert the children of the tbody inside the element with the given ID into an array
                    return Array.from(
                    document.querySelector('#productDetails_detailBullets_sections1 > tbody').children
                    )
                    // Filter the array to only include elements where the first child's text includes any of the search terms
                    .filter(x =>
                    searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term))
                    )
                    // Map over the filtered elements to get the outerText of their second child
                    .map(x => x.children[1].outerText)
                    // Join all the resulting strings with a comma and a space
                    .join(', ')
                }} catch (e) {{
                    // If any errors are caught, return an empty string
                    return ''
                }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


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


def shopping_get_num_reviews(page: Page | PseudoPage) -> int:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseInt(document.querySelector(\"#tab-label-reviews-title\")
                    .outerText.split(' ')[1]);
                    return res ? res : 0; }}
                )();
            """
        )
    except Exception:
        result = 0

    return result


def shopping_get_rating_as_percentage(page: Page | PseudoPage) -> int:
    """Get the rating of the product on the shopping website as a percentage out of 100."""
    try:
        rating = page.evaluate(
            """
                (() => {{
                    ratingPercentage = parseFloat(document.querySelector('.rating-result').title.replace('%', ''));
                    return ratingPercentage ? ratingPercentage : 0;
                }})();
            """
        )
    except Exception:
        rating = 0

    return rating


def get_query_text(page: Page | PseudoPage, selector: str) -> str:
    """Get the text content of the element matching the given selector.

    Note that this function DOES NOT perform downcasing.
    """
    try:
        result = page.evaluate(
            f"""
                (() => {{
                    try {{
                        return document.querySelector('{selector}').textContent;
                    }} catch (e) {{
                        return '';
                    }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


def is_selector_present(page: Page | PseudoPage, selector: str) -> str:
    """Return True if an element matching the CSS selector exists on the page."""
    try:
        result = page.evaluate(
            f"""
                (() => {{
                    try {{
                        return !!document.querySelector('{selector}');
                    }} catch (e) {{
                        return false;
                    }}
                }})();
            """
        )
    except Exception:
        result = "false"

    return str(result).lower()


def get_query_text_lowercase(page: Page | PseudoPage, selector: str) -> str:
    """Get the lowercase text content of the element matching the given selector."""
    return get_query_text(page, selector).lower()


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(imageA.size[1], imageB.size[1])

    # Get LANCZOS
    LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

    # Resize images
    imageA = imageA.resize(new_size, LANCZOS)
    imageB = imageB.resize(new_size, LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score


def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


def reddit_get_post_comment_tree(page: Page | PseudoPage) -> Dict[str, Any]:
    try:
        comment_tree = page.evaluate(
            f"""(function buildCommentTree(node, data_level) {{
    let tree = {{
        "username": node.querySelector(".fg-inherit").outerText,
        "net_score": parseInt(node.querySelector(".vote__net-score").outerText),
        "content": node.querySelector(".comment__content").outerText,
        "time": new Date(node.querySelector('.comment__main > header > h1 > span > time').dateTime),
        "children": []
    }};
    node.querySelectorAll(".comment").forEach((child) => {{
        if (parseInt(child.getAttribute('data-level')) === data_level+1) {{
            tree['children'].push(buildCommentTree(child, data_level+1));
        }}
    }})

    return tree;
}})(document.querySelector("#main"), 0)"""
        )
    except Exception:
        comment_tree = {}

    return comment_tree


def reddit_get_latest_comment_obj_by_username(page: Page | PseudoPage, username: str) -> Dict[str, Any]:
    try:
        comment_tree = reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            # Ensure node time is timezone-aware
            node_time = node["time"]
            if node_time.tzinfo is None:
                node_time = node_time.replace(tzinfo=timezone.utc)

            if node["username"] == username:
                if node_time > latest_time:
                    comment = {
                        "username": node["username"],
                        "net_score": node["net_score"],
                        "content": node["content"],
                        "time": node_time,
                    }
                    latest_time = node_time

            for child in node["children"]:
                dfs(child)

        dfs(comment_tree)

    except Exception as e:
        comment = {}
    return comment


def reddit_get_latest_comment_content_by_username(page: Page | PseudoPage, username: str) -> str:
    try:
        comment = reddit_get_latest_comment_obj_by_username(page, username)
        content = comment["content"]

    except Exception:
        content = ""

    return content


def reddit_get_parent_comment_obj_of_latest_comment_by_username(page: Page | PseudoPage, username: str) -> Dict[str, Any]:
    try:
        comment_tree = reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            for child in node["children"]:
                if child["username"] == username:
                    if child["time"] > latest_time:
                        comment = {
                            "username": node["username"],
                            "net_score": node["net_score"],
                            "content": node["content"],
                            "time": node["time"],
                        }
                        latest_time = child["time"]
                else:
                    dfs(child)

        dfs(comment_tree)

    except Exception:
        comment = {}
    return comment


def reddit_get_parent_comment_username_of_latest_comment_by_username(page: Page | PseudoPage, username: str) -> str:
    try:
        comment = reddit_get_parent_comment_obj_of_latest_comment_by_username(page, username)
        username = comment["username"]

    except Exception:
        username = ""

    return username


def gitlab_get_project_memeber_role(page: Page | PseudoPage, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role


def call_fuzzy_match_llm(messages: list[dict[str, Any]], model: str) -> str:
    try:
        # return "correct"
        provider = infer_provider(model)
        gen_kwargs = {
            "model": model,
            "temperature": 0,
            # "max_tokens": 8192,
            "top_p": 0.01,
            "reasoning_effort": "medium",
            "thinking_budget": 24576,
            "provider": provider,
            "engine": "automodel",
            "num_generations": 1,
        }

        _, model_gens = call_llm(gen_kwargs, messages)

        response = model_gens[0].text()
        logger.info(f"Fuzzy match response: {response}")
        return response.lower()
    except Exception as e:
        logger.error(f"Error calling fuzzy match LLM: {e}", exc_info=True)
        return ""


def llm_fuzzy_match(pred: str, reference: str, question: str, model: str) -> float:
    """Check whether the prediction matches the reference with LLMs"""
    if not model:
        raise ValueError("LLM is required for fuzzy matching")
    messages: list[dict[str, Any]] = []

    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."

    messages = [
        {"role": "system", "inputs": "You are a helpful assistant"},
        {"role": "user", "inputs": message},
    ]

    response = call_fuzzy_match_llm(messages, model)
    if "partially correct" in response or "incorrect" in response:
        return 0.0
    else:
        assert "correct" in response, response
        return 1.0


def llm_ua_match(pred: str, reference: str, question: str, model: str) -> float:
    """Check whether the prediction matches the reference with LLMs"""
    if not model:
        raise ValueError("LLM is required for fuzzy matching")
    messages: list[dict[str, Any]] = []

    sys_prompt = (
        "You will receive a description of a task that is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the reported reason is aligned with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )

    # construct the question to ask
    message = ""
    message += f"task: {question}\n\n"
    message += f"actual unachievable reason: {reference}\n\n"
    message += f"reported unachievable reason: {pred}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": message},
    ]

    response = call_fuzzy_match_llm(messages, model)
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0
