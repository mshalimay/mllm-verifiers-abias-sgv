# NOTE[mandrade]: changes to add to support dynamic URLs, website display names, env parallelization, robust url mapping

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import numpy.typing as npt
import requests
from benchmark_config import URLS_FILE_TEMPLATE
from PIL import Image

from .env_config import PAGE_NAME_NORMALIZATIONS, SITE_DISPLAY_NAMES, URL_MAPPINGS, Website

URL_MAPPINGS_FLAT: list[tuple[Website, str, str]] = []
LOCAL_URLS_NORM: dict[Website, str] = {}


@dataclass
class DetachedPage:
    url: str
    content: str  # html


def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))


def get_image_from_url(url: str, headers: dict | None = None, timeout: int = 60) -> Image.Image:
    if not headers:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=timeout)
        return Image.open(response.raw)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to fetch image from URL {url}: {e}")


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None
    center: list[float] | None


class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: int
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None
    center: list[float] | None


class BrowserConfig(TypedDict):
    win_upper_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]

Observation = str | npt.NDArray[np.uint8] | tuple[npt.NDArray[np.uint8], Any, Any]


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]


def get_website_urls_from_file(instance_id: int | str) -> dict[Website, str]:
    """Get the list of websites that are being hosted."""
    site_urls = {}
    url_file = URLS_FILE_TEMPLATE.format(PROCESS_ID=instance_id)
    if not os.path.exists(url_file):
        raise FileNotFoundError(f"URLs file not found: {url_file}. ")

    with open(url_file) as urls_file:
        lines = urls_file.readlines()
    for line in lines:
        line_instance_id = line.split(" ")[0].split("-")[1]
        if line_instance_id == str(instance_id):
            website_name = line.split("-")[0]
            site_urls[Website(website_name)] = line.split(" ")[1].strip()

    if not site_urls:
        raise ValueError(f"No URLs found for instance ID {instance_id} in file {url_file}.")
    return site_urls


def build_task_config(config_file: str | Path | dict[str, Any], docker_instance_id: int) -> tuple[dict[str, Any], str, str, list[Image.Image], list[str]]:
    """
    Loads the task config from the config file.
    Args:
        config_file: The path to the config file.
        docker_instance_id: The id of the docker instance.
    Returns:
        tuple[dict[str, Any], str, str, list[Image.Image], list[str]]:
            - task_config: A dict containing the task config.
            - intent: A string containing the intent.
            - task_id: A string containing the task id.
            - intent_images: A list of PIL images for the intent, if config file contains images.
            - image_paths: A list of image paths, if config file contains images.
    """
    if isinstance(config_file, str):
        with open(config_file, "r") as f:
            config_file_string = f.read()
    else:
        config_file_string = json.dumps(config_file)

    strings_to_replace = {
        "__SHOPPING__": Website.SHOPPING,
        "__CLASSIFIEDS__": Website.CLASSIFIEDS,
        "__REDDIT__": Website.REDDIT,
        "__WIKIPEDIA__": Website.WIKIPEDIA,
        "__SHOPPING_ADMIN__": Website.SHOPPING_ADMIN,
        "__GITLAB__": Website.GITLAB,
        "__MAP__": Website.MAP,
    }
    websites_urls = get_website_urls_from_file(instance_id=docker_instance_id)
    replace_map = {string_to_replace: websites_urls[website] for string_to_replace, website in strings_to_replace.items() if website in websites_urls}
    for key, value in replace_map.items():
        config_file_string = re.sub(re.escape(key), value, config_file_string)
    # TODO: generalize docker separator
    config_file_string = re.sub(r"(?<!\d)_state.json", f"-{docker_instance_id}_state.json", config_file_string)

    task_config = json.loads(config_file_string)

    task_config["websites_urls"] = websites_urls
    intent = task_config["intent"]
    task_id = task_config["task_id"]

    # Regularize image paths data type
    image_paths = task_config.get("image", None)
    intent_images: list[Image.Image] = []
    image_paths = [] if image_paths is None else image_paths
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    assert isinstance(image_paths, list), "Image paths must be a list"

    # Add docker instance id to task config
    task_config["docker_instance_id"] = docker_instance_id

    for image_path in image_paths:
        if image_path.startswith("http"):
            input_image = get_image_from_url(image_path)
        else:
            input_image = Image.open(image_path)
        intent_images.append(input_image)

    return task_config, intent, task_id, intent_images, image_paths


def check_websites(process_id: str, site_names: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Check the status of websites listed in a file, filtered by process_id and/or site_types.
    Returns a list of dicts: { 'site': ..., 'status': 1|0, 'url': ... }
    """
    urls_file = URLS_FILE_TEMPLATE.format(PROCESS_ID=process_id)
    if not os.path.isfile(urls_file):
        print(f"URLs file does not exist: {urls_file}")
        return []

    site_names = site_names or []
    sites_to_check = []
    if process_id:
        if site_names:
            # If both process_id and site_types are provided
            sites_to_check = [f"{site_name}-{process_id}" for site_name in site_names]
        else:
            # If only process_id is provided, filter by ID in the loop
            pass
    results = []
    with open(urls_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            site_id, url = parts[0], parts[1]
            # If process_id is provided but no specific sites, check if site ends with the process_id
            if process_id and not site_names:
                if not site_id.endswith(f"-{process_id}"):
                    continue
            # If specific sites were provided, check against the sites_to_check list
            elif sites_to_check and site_id not in sites_to_check:
                continue
            try:
                resp = requests.get(url, timeout=5)
                status = 1 if resp.ok else 0
            except Exception:
                status = 0
            site_name, instance_id = site_id.split("-")
            results.append({"name": site_name, "status": status, "url": url, "instance_id": instance_id})
    return results


def map_url_to_local(url: str, docker_instance_id="") -> str:
    """Map real urls to their local counterparts
    Example: https://wikipedia.org -> http://localhost:8888/wikipedia_en_all_maxi_2022-05
    """
    if docker_instance_id:
        set_local_url_mappings(int(docker_instance_id))

    url_new = re.sub(r"https://", "http://", url, flags=re.IGNORECASE)
    url_new = re.sub("//en.", "//", url_new, flags=re.IGNORECASE)
    url_new = re.sub("www.", "", url_new, flags=re.IGNORECASE)

    url_modified = False

    for website, local_url, real_url in URL_MAPPINGS_FLAT:
        # relevant only for Wikipedia, which has an additional path for the local url
        local_url = LOCAL_URLS_NORM[website]

        # If real_url counterpart in `url`, map it to the `local` equivalent
        if real_url in url_new:
            url_new = url_new.replace(real_url, local_url)
            url_modified = True
            continue

        real_url_part = real_url.replace("http://", "")
        if re.search(real_url_part, url_new, re.IGNORECASE):
            # local_url_part = local_url.replace("http://", "")
            url_new = url_new.replace(real_url_part, local_url)
            url_new = re.sub("www.", "", url_new)
            url_modified = True
            continue

        # If the mapping is for a `.org` URL, also check for its `.com` variant.
        if real_url.endswith(".org"):
            alt_url = real_url[:-4] + ".com"
            if alt_url in url_new:
                # First, fix the URL so that it uses .org instead of .com.
                url_new = url_new.replace(alt_url, real_url)
                # Now perform the usual mapping.
                url_new = url_new.replace(real_url, local_url)
                url_modified = True
                continue

    # If no local urls found, return the original url
    return url_new if url_modified else url


def map_url_to_real(url: str) -> str:
    """Map local urls to their real world counterparts"""
    url = re.sub(r"https://", "http://", url, flags=re.IGNORECASE)
    url = re.sub("www.", "", url, flags=re.IGNORECASE)
    url = re.sub("//en.", "//", url, flags=re.IGNORECASE)

    for website, local_url, real_url in URL_MAPPINGS_FLAT:
        local_url_norm = LOCAL_URLS_NORM[website]
        if local_url_norm in url:
            url = url.replace(local_url_norm, real_url)
        elif local_url in url:
            url = url.replace(local_url, real_url)
    return url


def map_endpoint_to_target(url: str, target_endpoint: str = "127.0.0.1") -> str:
    """Map the endpoint of the url to a target counterpart.
    This is needed for String match evals if hosting the benchmark on other machines.
    Example: http://143.215.128.18:8888 -> http://127.0.0.1:8888

    Args:
        url (str): The url to map the endpoint of.
        local_endpoint (str, optional): The local endpoint to map the url to. Defaults to "127.0.0.1".

    Returns:
        str: The url with the endpoint mapped to the local counterpart.
    """
    url = re.sub(r"https://", "http://", url)

    for _, instance_url, _ in URL_MAPPINGS_FLAT:
        if instance_url in url:
            base_endpoint = re.sub(r"^https?://", "", instance_url)
            base_endpoint = re.sub(r":[^:]*$", "", base_endpoint)
            return url.replace(base_endpoint, target_endpoint)
    return url


def get_site_name(url: str) -> str:
    """Get the site name from the url"""

    for website, local_url, real_url in URL_MAPPINGS_FLAT:
        if local_url in url:
            return SITE_DISPLAY_NAMES[website]
    return ""


def get_normalized_site_name(url: str) -> tuple[str, str]:
    """Get the normalized site name from the url"""

    for website, local_url, real_url in URL_MAPPINGS_FLAT:
        if local_url in url:
            if website in PAGE_NAME_NORMALIZATIONS:
                return PAGE_NAME_NORMALIZATIONS[website]
    return "", ""


def get_running_websites(process_id: str, sites_to_exclude: list[Website] = []) -> list[Website]:
    all_sites_data = check_websites(process_id=process_id)
    all_sites = [Website(site_data["name"]) for site_data in all_sites_data if site_data["status"] == 1]
    all_sites = [website for website in all_sites if website not in sites_to_exclude]
    return all_sites


# ===============================================================================
# Internet helpers
# ===============================================================================


def log_or_print(logger: Any, log_type: str, message: str) -> None:
    if logger:
        if log_type == "info":
            logger.info(message)
        elif log_type == "warning":
            logger.warning(message)
        elif log_type == "error":
            logger.error(message)
        elif log_type == "debug":
            logger.debug(message)
        elif log_type == "critical":
            logger.critical(message)
    else:
        print(message)


def wait_for_spinners(page: Any, selector: str, max_timeout_ms: float = 2 * 1000, wait_to_appear_ms: int = 0, state="detached") -> None:
    if page.query_selector(selector):
        page.wait_for_selector(selector, state=state, timeout=max_timeout_ms)

    if not wait_to_appear_ms:
        return

    time.sleep(wait_to_appear_ms / 1000)
    if page.query_selector(selector):
        page.wait_for_selector(selector, state=state, timeout=max_timeout_ms)
    else:
        return


def get_dom_hash(page: Any) -> str:
    """Compute a hash of the page DOM to check for stability."""
    dom = page.content()
    return hashlib.md5(dom.encode("utf-8")).hexdigest()


def wait_for_dom_hash_stability(page: Any, interval_ms: float = 500, checks: int = 3) -> None:
    """Waits until the DOM is stable for `checks` consecutive intervals."""
    previous_hash = get_dom_hash(page)
    stable_count = 0

    for _ in range(checks):
        time.sleep(interval_ms / 1000)
        current_hash = get_dom_hash(page)
        if current_hash == previous_hash:
            stable_count += 1
        else:
            stable_count = 0
            previous_hash = current_hash

    if stable_count == checks:
        return
    else:
        raise Exception(f"DOM hash is not stable after {checks} checks")


def wait_with_timeouts(wait_func: Callable, timeouts: list[int], logger=None, description="operation") -> bool:
    """
    Attempts to perform a waiting operation using a series of progressive timeouts.

    Args:
        wait_func: A callable that receives a timeout (in ms) and performs the waiting operation.
        timeouts: A list of timeout values (in ms) to try sequentially.
        logger: Optional logger for reporting status messages.
        description: A textual description of the operation for logging purposes.

    Returns:
        bool: True if one attempt was successful, False if all attempts failed.
    """
    for t in timeouts:
        try:
            wait_func(t)
            if logger:
                logger.debug(f"{description} succeeded with timeout {t}ms")
            return True
        except Exception as e:
            if logger:
                logger.debug(f"{description} failed with timeout {t}ms: {e}")
    log_or_print(logger, "warning", f"All attempts for {description} failed. Tried timeouts: {timeouts}")
    return False


def wait_for_page_to_stabilize(
    page: Any,
    max_timeout_per_check_ms: float = 2 * 1000,
    logger: Any | None = None,
    min_num_trues: int = 3,
    return_early: bool = False,
    return_after: int | None = None,
    hard_sleep: float = 0.0,
    max_overall_timeout_seconds: float = 5,
    min_wait_time_seconds: float = 0.0,
) -> bool:
    """
    Wait for a page to fully stabilize using multiple waiting mechanisms.
    Stops early if min_num_trues checks pass.

    Args:
        page: Playwright page object (sync API)
        max_timeout_ms: Maximum timeout per wait function
        logger: Optional logger for reporting
        min_num_trues: Minimum number of successful checks to consider the page stable
        return_early: If True, return after min_num_trues checks pass.
        return_after: If not None, return after `return_after` successful checks; overrides return_early.
    Returns:
        bool: True if at least min_num_trues checks passed; False otherwise.
    """
    if hard_sleep > 0.0:
        log_or_print(logger, "info", f"Hard sleep for {hard_sleep} seconds.")
        time.sleep(hard_sleep)
        return True
    successful_checks = 0
    start_time = time.time()

    checks = [
        # Check 1: DOMContentLoaded
        # --------------------------
        # Wait for the DOMContentLoaded event which fires after the initial HTML
        # is loaded and parsed. This indicates that the core DOM structure is available.
        (
            lambda t: page.wait_for_load_state("domcontentloaded", timeout=min(t, max_timeout_per_check_ms)),
            [500, 500, 500],  # Wait for a max of 1500ms
            "DOMContentLoaded",
        ),
        # Check 2: Document Ready State
        # -----------------------------
        # Wait until document.readyState is 'complete', ensuring that all sub-resources,
        # such as images and stylesheets, have been fully loaded.
        (
            lambda t: page.wait_for_function("document.readyState === 'complete'", timeout=min(t, max_timeout_per_check_ms)),
            [500, 500, 500],
            "document ready state",
        ),
        # Check 3: Network Idle
        # ----------------------
        # Wait for the network to become idle. This check ensures that asynchronous
        # background tasks (e.g., API calls or lazy-loaded resources) have mostly finished.
        (
            lambda t: page.wait_for_load_state("networkidle", timeout=min(t, max_timeout_per_check_ms)),
            [500, 500, 500, 500, 500],  # Wait for a total of 2500ms
            "network idle",
        ),
        # Check 4: DOM Stability
        # -----------------------
        # Utilizes a MutationObserver to monitor changes in the DOM.
        # If no DOM mutations occur for 300ms, it is assumed that the page has stabilized.
        (
            lambda t: page.wait_for_function(
                """() => {
                    return new Promise(resolve => {
                        let timeout;
                        const observer = new MutationObserver(() => {
                            clearTimeout(timeout);
                            timeout = setTimeout(resolve, 300);
                        });
                        observer.observe(document.body, {
                            childList: true,
                            subtree: true,
                            attributes: true,
                            characterData: true
                        });
                        timeout = setTimeout(resolve, 300);
                    });
                }""",
                timeout=min(t, max_timeout_per_check_ms),
            ),
            [500, 500, 500],  # Wait for a total of 1500ms
            "DOM stability",
        ),
        (
            lambda t: wait_for_dom_hash_stability(page, interval_ms=t, checks=3),
            [50, 100],
            "DOM stability",
        ),
        # Check 5: Animation Frames
        # --------------------------
        # Uses two consecutive requestAnimationFrame calls to ensure that all rendering updates,
        # such as animations and transitions, have been flushed.
        (
            lambda t: page.wait_for_function(
                """() => new Promise(resolve => {
                    requestAnimationFrame(() => requestAnimationFrame(resolve));
                })""",
                timeout=min(t, max_timeout_per_check_ms),
            ),
            [200, 200, 100],  # Wait for a max of 500ms
            "animation frames",
        ),
        (
            lambda t: wait_for_spinners(page, selector="#checkout-loader", max_timeout_ms=min(t, max_timeout_per_check_ms), wait_to_appear_ms=5),
            [500, 500, 500],  # Wait for a max of 2000ms
            "spinner disappearance",
        ),
    ]

    # Regularize params
    min_num_trues = min(min_num_trues, len(checks))

    if not return_after:
        # If return_after is not set, use return_early or default to len(checks)
        if return_early:
            return_after = min_num_trues
        else:
            return_after = len(checks)
    else:
        # If return_after is set, use it and ensure it's not greater than len(checks)
        return_after = min(return_after, len(checks))

    # emulate_slow_network(page)
    is_stable = False
    try:
        for wait_func, timeouts, description in checks:
            if time.time() - start_time > max_overall_timeout_seconds:
                log_or_print(logger, "warning", f"Page stabilization timed out after {max_overall_timeout_seconds}s.")
                return False

            try:
                if wait_with_timeouts(wait_func, timeouts, logger=logger, description=description):
                    successful_checks += 1

                # Stop early if the minimum number of successful checks has been reached.
                if successful_checks >= return_after:
                    log_or_print(logger, "debug", f"Page stabilized early after {successful_checks} successful checks.")
                    is_stable = True
                    break
            except Exception as e:
                log_or_print(logger, "warning", f"Error in timeout check {description}: {e}")
                continue

    except Exception as e:
        log_or_print(logger, "warning", f"Error in wait_for_page_to_stabilize: {e}")

    if is_stable:
        log_or_print(logger, "debug", f"Page stabilized with {successful_checks} successful checks.")
    else:
        log_or_print(logger, "warning", f"Page possibly not stabilized.")

    if min_wait_time_seconds:
        elapsed = time.time() - start_time
        if elapsed < min_wait_time_seconds:
            remaining_time = min_wait_time_seconds - elapsed
            log_or_print(
                logger,
                "info",
                f"Webpage stability checks finished before min_wait_time_seconds {min_wait_time_seconds}s. Sleeping for {remaining_time} seconds.",
            )
            time.sleep(remaining_time)

    return is_stable


def get_local_url_norm(website: Website) -> str:
    if website in LOCAL_URLS_NORM:
        return LOCAL_URLS_NORM[website]
    else:
        raise ValueError(f"Website {website} not found in LOCAL_URLS_NORM")


def set_local_url_mappings(instance_id: int):
    global URL_MAPPINGS_FLAT, LOCAL_URLS_NORM
    website_urls = get_website_urls_from_file(instance_id)

    url_mappings_flat = []
    for website, instance_url in website_urls.items():
        for real_url in URL_MAPPINGS[website]:
            url_mappings_flat.append((website, instance_url, real_url))

        if website == Website.WIKIPEDIA:
            LOCAL_URLS_NORM[website] = instance_url + "/wikipedia_en_all_maxi_2022-05"
        else:
            LOCAL_URLS_NORM[website] = instance_url
    URL_MAPPINGS_FLAT = url_mappings_flat


def get_url_from_file(website: Website, instance_id: int) -> str:
    """Get the URL for a given website and instance ID from the URLs file."""
    url_file = URLS_FILE_TEMPLATE.format(PROCESS_ID=instance_id)
    if not os.path.exists(url_file):
        return ""
    with open(url_file) as urls_file:
        lines = urls_file.readlines()
    line = next(line for line in lines if line.startswith(f"{website}-{instance_id}"))
    url = line.split(" ")[1].strip()
    return url
