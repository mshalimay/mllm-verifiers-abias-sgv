# NOTE[mandrade]: refined SoM, fix interaction with `select` elements, correct tab metadata information, refine webpage display names, allow logging of bounding boxes and raw screenshots

import json
import re
from collections import defaultdict
from io import BytesIO, StringIO
from typing import Any, Optional, TypedDict
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import CDPSession, Page, ViewportSize

from browser_env.constants import ASCII_CHARSET, FREQ_UNICODE_CHARSET, IGNORED_ACTREE_PROPERTIES, UTTERANCE_MAX_LENGTH
from core_utils import timing_utils as timer
from core_utils.logger_utils import logger
from core_utils.timing_utils import timeit

from .env_utils import (
    AccessibilityTree,
    AccessibilityTreeNode,
    BrowserConfig,
    BrowserInfo,
    DOMNode,
    DOMTree,
    Observation,
    get_image_from_url,
    get_normalized_site_name,
    get_site_name,
    map_url_to_real,
    png_bytes_to_numpy,
    wait_for_page_to_stabilize,
)

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
CAPTIONING_BATCH_SIZE = 4


def remove_unicode(input_string):
    # Define a regex pattern to match Unicode characters
    unicode_pattern = re.compile(r"[^\x00-\x7F]+")

    # Use the pattern to replace Unicode characters with an empty string
    cleaned_string = unicode_pattern.sub("", input_string)
    return cleaned_string


class ObservationProcessor:
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class ObservationMetadata(TypedDict):
    obs_nodes_info: dict[str, Any]
    obs_nodes_bbox_info: Optional[dict[str, Any]]
    obs_nodes_semantic_info: Optional[dict[str, Any]]
    open_tabs: Optional[dict[str, Any] | str]


def create_empty_metadata() -> ObservationMetadata:
    return {
        "obs_nodes_info": {},
        "obs_nodes_bbox_info": None,
        "obs_nodes_semantic_info": None,
        "open_tabs": None,
    }


class TextObervationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ):
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_size = viewport_size
        self.observation_tag = "text"
        self.meta_data = create_empty_metadata()  # use the store meta data of this observation type

        # Cache captions.
        self.url2caption = {}
        if self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            self.captioning_fn = captioning_fn

    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    @staticmethod
    def get_bounding_client_rect(client: CDPSession, backend_node_id: str) -> dict[str, Any]:
        try:
            remote_object = client.send("DOM.resolveNode", {"backendNodeId": int(backend_node_id)})
            remote_object_id = remote_object["object"]["objectId"]
            response = client.send(
                "Runtime.callFunctionOn",
                {
                    "objectId": remote_object_id,
                    "functionDeclaration": """
                        function() {
                            if (this.nodeType == 3) {
                                var range = document.createRange();
                                range.selectNode(this);
                                var rect = range.getBoundingClientRect().toJSON();
                                range.detach();
                                return rect;
                            } else {
                                return this.getBoundingClientRect().toJSON();
                            }
                        }
                    """,
                    "returnByValue": True,
                },
            )
            return response
        except Exception as _:
            return {"result": {"subtype": "error"}}

    @staticmethod
    def get_element_in_viewport_ratio(
        elem_left_bound: float,
        elem_top_bound: float,
        width: float,
        height: float,
        config: BrowserConfig,
    ) -> float:
        elem_right_bound = elem_left_bound + width
        elem_lower_bound = elem_top_bound + height

        win_left_bound = 0
        win_right_bound = config["win_width"]
        win_top_bound = 0
        win_lower_bound = config["win_height"]

        # Compute the overlap in x and y axes
        overlap_width = max(
            0,
            min(elem_right_bound, win_right_bound) - max(elem_left_bound, win_left_bound),
        )
        overlap_height = max(
            0,
            min(elem_lower_bound, win_lower_bound) - max(elem_top_bound, win_top_bound),
        )

        # Compute the overlap area
        ratio = overlap_width * overlap_height / width * height
        return ratio

    def fetch_page_html(
        self,
        info: BrowserInfo,
        page: Page,
        client: CDPSession,
        current_viewport_only: bool,
    ) -> DOMTree:
        # adopted from [natbot](https://github.com/nat/natbot)
        tree = info["DOMTree"]
        strings = tree["strings"]
        document = tree["documents"][0]
        nodes = document["nodes"]

        # make a dom tree that is easier to navigate
        dom_tree: DOMTree = []
        graph = defaultdict(list)
        for node_idx in range(len(nodes["nodeName"])):
            cur_node: DOMNode = {
                "nodeId": "",
                "nodeType": "",
                "nodeName": "",
                "nodeValue": "",
                "attributes": "",
                "backendNodeId": "",
                "parentId": "",
                "childIds": [],
                "cursor": 0,
                "union_bound": None,
                "center": None,
            }

            node_type_idx = nodes["nodeType"][node_idx]
            node_type = "generic"
            if node_type_idx >= 0 and node_type_idx < len(strings):
                node_type = strings[node_type_idx]

            node_name = strings[nodes["nodeName"][node_idx]]

            node_value_idx = nodes["nodeValue"][node_idx]
            node_value = ""
            if node_value_idx >= 0 and node_value_idx < len(strings):
                node_value = " ".join(strings[node_value_idx].split())

            node_attributes = [strings[i] for i in nodes["attributes"][node_idx]]
            node_attributes_str = ""
            for i in range(0, len(node_attributes), 2):
                a = node_attributes[i]
                b = node_attributes[i + 1]
                b = " ".join(b.split())
                node_attributes_str += f'{a}="{b}" '
            node_attributes_str = node_attributes_str.strip()

            cur_node["nodeId"] = str(node_idx)
            cur_node["nodeType"] = node_type
            cur_node["nodeName"] = node_name
            cur_node["nodeValue"] = node_value
            cur_node["attributes"] = node_attributes_str
            cur_node["backendNodeId"] = str(nodes["backendNodeId"][node_idx])
            cur_node["parentId"] = str(nodes["parentIndex"][node_idx])

            if cur_node["parentId"] != "-1":
                graph[cur_node["parentId"]].append(str(cur_node["nodeId"]))

            # get the bound
            if cur_node["parentId"] == "-1":
                cur_node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(client, cur_node["backendNodeId"])
                if response.get("result", {}).get("subtype", "") == "error":
                    cur_node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    cur_node["union_bound"] = [x, y, width, height]

            dom_tree.append(cur_node)

        # add parent children index to the node
        for parent_id, child_ids in graph.items():
            dom_tree[int(parent_id)]["childIds"] = child_ids

        # remove the nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: DOMNode) -> None:
                # update the node information in the accessibility tree
                node_id = node["nodeId"]
                parent_id = node["parentId"]
                child_ids = node["childIds"]

                # update the children of the parent node
                assert dom_tree[int(parent_id)]["parentId"] != "[REMOVED]"
                # remove the nodeid from parent
                index = dom_tree[int(parent_id)]["childIds"].index(node_id)
                dom_tree[int(parent_id)]["childIds"].pop(index)

                # Insert children_nodeids in the same location
                for child_id in child_ids:
                    dom_tree[int(parent_id)]["childIds"].insert(index, child_id)
                    index += 1

                # update children node's parent
                for child_id in child_ids:
                    dom_tree[int(child_id)]["parentId"] = parent_id
                # mark as removed
                dom_tree[int(node_id)]["parentId"] = "[REMOVED]"

            config = info["config"]
            for cursor, node in enumerate(dom_tree):
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0.0 or height == 0.0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            dom_tree = [node for node in dom_tree if node.get("parentId", "-1") != "[REMOVED]"]

        return dom_tree

    @staticmethod
    def parse_html(dom_tree: DOMTree) -> tuple[str, dict[str, Any]]:
        """Parse the html tree into a string text"""

        obs_nodes_info = {}
        nodeid_to_cursor = {node["nodeId"]: idx for idx, node in enumerate(dom_tree)}

        def dfs(node_cursor: int, depth: int) -> str:
            tree_str = ""
            node = dom_tree[node_cursor]
            indent = "\t" * depth
            valid_node = True
            try:
                node_str = f"[{node_cursor}] <{node['nodeName']}"
                if node["attributes"]:
                    node_str += f" {node['attributes']}"
                node_str += f"> {node['nodeValue']}"
                valid_node = bool(node["attributes"] or node["nodeValue"])

                if valid_node:
                    obs_nodes_info[str(node_cursor)] = {
                        "backend_id": node["backendNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }
                    tree_str += f"{indent}{node_str}\n"

            except Exception as _:
                valid_node = False

            for child_ids in node["childIds"]:
                child_cursor = nodeid_to_cursor[child_ids]
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(child_cursor, child_depth)
                tree_str += child_str

            return tree_str

        html = dfs(0, 0)
        return html, obs_nodes_info

    def fetch_page_accessibility_tree(
        self,
        info: BrowserInfo,
        client: CDPSession,
        current_viewport_only: bool,
    ) -> AccessibilityTree:
        accessibility_tree: AccessibilityTree = client.send("Accessibility.getFullAXTree", {})["nodes"]

        # a few nodes are repeated in the accessibility tree
        seen_ids = set()
        _accessibility_tree = []
        for node in accessibility_tree:
            if node["nodeId"] not in seen_ids:
                _accessibility_tree.append(node)
                seen_ids.add(node["nodeId"])
        accessibility_tree = _accessibility_tree

        nodeid_to_cursor = {}
        for cursor, node in enumerate(accessibility_tree):
            nodeid_to_cursor[node["nodeId"]] = cursor
            # usually because the node is not visible etc
            if "backendDOMNodeId" not in node:
                node["union_bound"] = None
                continue
            backend_node_id = str(node["backendDOMNodeId"])
            if node["role"]["value"] == "RootWebArea":
                # always inside the viewport
                node["union_bound"] = [0.0, 0.0, 10.0, 10.0]
            else:
                response = self.get_bounding_client_rect(client, backend_node_id)
                if response.get("result", {}).get("subtype", "") == "error":
                    node["union_bound"] = None
                else:
                    x = response["result"]["value"]["x"]
                    y = response["result"]["value"]["y"]
                    width = response["result"]["value"]["width"]
                    height = response["result"]["value"]["height"]
                    node["union_bound"] = [x, y, width, height]

        # filter nodes that are not in the current viewport
        if current_viewport_only:

            def remove_node_in_graph(node: AccessibilityTreeNode) -> None:
                # update the node information in the accessibility tree
                nodeid = node["nodeId"]
                node_cursor = nodeid_to_cursor[nodeid]
                parent_nodeid = node["parentId"]
                children_nodeids = node["childIds"]
                parent_cursor = nodeid_to_cursor[parent_nodeid]
                # update the children of the parent node
                assert accessibility_tree[parent_cursor].get("parentId", "Root") is not None
                # remove the nodeid from parent's childIds
                index = accessibility_tree[parent_cursor]["childIds"].index(nodeid)
                accessibility_tree[parent_cursor]["childIds"].pop(index)
                # Insert children_nodeids in the same location
                for child_nodeid in children_nodeids:
                    accessibility_tree[parent_cursor]["childIds"].insert(index, child_nodeid)
                    index += 1
                # update children node's parent
                for child_nodeid in children_nodeids:
                    child_cursor = nodeid_to_cursor[child_nodeid]
                    accessibility_tree[child_cursor]["parentId"] = parent_nodeid
                # mark as removed
                accessibility_tree[node_cursor]["parentId"] = "[REMOVED]"

            config = info["config"]
            for node in accessibility_tree:
                if not node["union_bound"]:
                    remove_node_in_graph(node)
                    continue

                [x, y, width, height] = node["union_bound"]

                # invisible node
                if width == 0 or height == 0:
                    remove_node_in_graph(node)
                    continue

                in_viewport_ratio = self.get_element_in_viewport_ratio(
                    elem_left_bound=float(x),
                    elem_top_bound=float(y),
                    width=float(width),
                    height=float(height),
                    config=config,
                )

                if in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD:
                    remove_node_in_graph(node)

            accessibility_tree = [node for node in accessibility_tree if node.get("parentId", "Root") != "[REMOVED]"]

        return accessibility_tree

    @staticmethod
    def parse_accessibility_tree(
        accessibility_tree: AccessibilityTree,
    ) -> tuple[str, dict[str, Any]]:
        """Parse the accessibility tree into a string text"""
        node_id_to_idx = {}
        for idx, node in enumerate(accessibility_tree):
            node_id_to_idx[node["nodeId"]] = idx

        obs_nodes_info = {}

        def dfs(idx: int, obs_node_id: str, depth: int) -> str:
            tree_str = ""
            node = accessibility_tree[idx]
            indent = "\t" * depth
            valid_node = True
            try:
                role = node["role"]["value"]
                name = node["name"]["value"]
                node_str = f"[{obs_node_id}] {role} {repr(name)}"
                properties = []
                for property in node.get("properties", []):
                    try:
                        if property["name"] in IGNORED_ACTREE_PROPERTIES:
                            continue
                        properties.append(f"{property['name']}: {property['value']['value']}")
                    except KeyError:
                        pass

                if properties:
                    node_str += " " + " ".join(properties)

                # check valid
                if not node_str.strip():
                    valid_node = False

                # empty generic node
                if not name.strip():
                    if not properties:
                        if role in [
                            "generic",
                            "img",
                            "list",
                            "strong",
                            "paragraph",
                            "banner",
                            "navigation",
                            "Section",
                            "LabelText",
                            "Legend",
                            "listitem",
                        ]:
                            valid_node = False
                    elif role in ["listitem"]:
                        valid_node = False

                if valid_node:
                    tree_str += f"{indent}{node_str}"
                    obs_nodes_info[obs_node_id] = {
                        "backend_id": node["backendDOMNodeId"],
                        "union_bound": node["union_bound"],
                        "text": node_str,
                    }

            except Exception as _:
                valid_node = False

            for _, child_node_id in enumerate(node["childIds"]):
                if child_node_id not in node_id_to_idx:
                    continue
                # mark this to save some tokens
                child_depth = depth + 1 if valid_node else depth
                child_str = dfs(node_id_to_idx[child_node_id], child_node_id, child_depth)
                if child_str.strip():
                    if tree_str.strip():
                        tree_str += "\n"
                    tree_str += child_str

            return tree_str

        tree_str = dfs(0, accessibility_tree[0]["nodeId"], 0)
        return tree_str, obs_nodes_info

    @staticmethod
    def clean_accesibility_tree(tree_str: str) -> str:
        """further clean accesibility tree"""
        clean_lines: list[str] = []
        for line in tree_str.split("\n"):
            # remove statictext if the content already appears in the previous line
            if "statictext" in line.lower():
                prev_lines = clean_lines[-3:]
                pattern = r"\[\d+\] StaticText (.+)"

                match = re.search(pattern, line, re.DOTALL)
                if match:
                    static_text = match.group(1)[1:-1]  # remove the quotes
                    if static_text and all(static_text not in prev_line for prev_line in prev_lines):
                        clean_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines)

    def norm_page_name(self, page) -> str:
        tab_title = page.title()

        match = re.search(r"^(https?://[^/]+)", page.url)
        if not match:
            return tab_title

        partial_url = match.group(1)
        find, sub = get_normalized_site_name(partial_url)
        if find:
            tab_title = re.sub(find, sub, tab_title)
        return tab_title

    def append_site_to_tab_title(self, tab_title: str, page: Page) -> str:
        try:
            match = re.search(r"^(https?://[^/]+)", page.url)
            if not match:
                return tab_title

            partial_url = match.group(1)
            site_names = [get_site_name(partial_url)]
            if not site_names:
                return tab_title

            # Prevent duplicate site names
            if any(site_name.lower() in tab_title.lower() for site_name in site_names):
                return tab_title
            return f"{tab_title} - {site_names[0].capitalize()}"
        except Exception as _:
            return tab_title

    # NOTE[mandrade]: added method to get tab info and fix bug where tab info is not shown in `get_observation` from ObservationHandler
    def get_tab_info(self, page: Page) -> str:
        open_tabs = page.context.pages
        try:
            # NOTE[mandrade]: set proper page names to tab titles
            tab_titles = [self.norm_page_name(tab) for tab in open_tabs]

            if len(open_tabs) > 1:
                tab_titles = [self.append_site_to_tab_title(tab_titles[i], open_tabs[i]) for i in range(len(open_tabs))]

            # Build string with tab titles and indexes
            # E.g.: Tab 0: Welcome to Wikipedia, Tab 1: OneStopShop, Tab 2: DuckDuckGo
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[idx] = f"Tab {idx} (current): {tab_titles[idx]}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {tab_titles[idx]}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(["Tab {idx}" for idx in range(len(open_tabs))])
        return tab_title_str

    @timeit(custom_name="ENV:process_txt_obs")
    def process(self, page: Page, client: CDPSession) -> str:
        # get the tab info
        tab_title_str = self.get_tab_info(page)

        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=2500)
            browser_info = self.fetch_browser_info(page, client)

        if self.observation_type == "html":
            dom_tree = self.fetch_page_html(
                browser_info,
                page,
                client,
                current_viewport_only=self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_html(dom_tree)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info

        elif self.observation_type == "accessibility_tree":
            accessibility_tree = self.fetch_page_accessibility_tree(
                browser_info,
                client,
                current_viewport_only=self.current_viewport_only,
            )
            content, obs_nodes_info = self.parse_accessibility_tree(accessibility_tree)
            content = self.clean_accesibility_tree(content)
            self.obs_nodes_info = obs_nodes_info
            self.meta_data["obs_nodes_info"] = obs_nodes_info
        elif self.observation_type == "":
            content = ""
        elif self.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            timer.start("ENV:process_txt_obs:captioning")
            # Check if the current page is an image url
            if page.url.endswith((".jpg", ".jpeg", ".png")):
                # Load image from current url and run captioning on it.
                if page.url not in self.url2caption and self.captioning_fn is not None:
                    try:
                        image = get_image_from_url(page.url)
                        caption = self.captioning_fn([image])[0].strip()
                        self.url2caption[page.url] = remove_unicode(caption)
                    except Exception as e:
                        print("WARNING: ", e)

                content = self.url2caption.get(page.url, "Image")
            else:
                # Run captioning if captioning_fn is available
                if self.captioning_fn is not None:
                    try:
                        image_data = page.evaluate("""
                            () => {
                                const images = document.querySelectorAll('img');
                                return Array.from(images).map(img => img.getAttribute('src') || '');
                            }
                        """)
                    except Exception as e:
                        print("Failed to fetch image sources: ", e)
                        image_data = []

                    image_urls = []
                    for image_src in image_data:
                        try:
                            if not image_src.startswith(("http://", "https://", "www.")):
                                image_src = urljoin(page.url, image_src)
                            if image_src not in self.url2caption:
                                image_urls.append(image_src)
                        except Exception as e:
                            print("WARNING:", e)

                    # Run image captioning on image_url pixels. This is for models which use captioning as a baseline.
                    if len(image_urls) > 0:
                        image_pixels = []
                        valid_urls = []
                        for url in image_urls:
                            if "data:image/svg" in url:
                                continue
                            else:
                                try:
                                    image = get_image_from_url(url)
                                    image_pixels.append(image)
                                    valid_urls.append(url)
                                except Exception as e:
                                    print("WARNING: ", e)

                        # Call the captioning function in batches.
                        if image_pixels:
                            timer.start("ENV:process_txt_obs:captioning:captioner_call")
                            captions = []
                            for i in range(0, len(image_pixels), CAPTIONING_BATCH_SIZE):
                                try:
                                    captions.extend(self.captioning_fn(image_pixels[i : i + CAPTIONING_BATCH_SIZE]))
                                except Exception as e:
                                    print("WARNING: ", e)
                                    captions.extend([""] * len(image_pixels[i : i + CAPTIONING_BATCH_SIZE]))
                            assert len(valid_urls) == len(captions), f"len(images)={len(valid_urls)}, len(captions)={len(captions)}"
                            for image_url, caption in zip(valid_urls, captions):
                                self.url2caption[image_url] = remove_unicode(caption.strip())
                            timer.end("ENV:process_txt_obs:captioning:captioner_call")

                # Update image alt attributes with captions (if available) and URLs
                image_updates = []
                images_data = page.evaluate("""
                    () => {
                        const images = document.querySelectorAll('img');
                        return Array.from(images).map(img => ({
                            alt: img.getAttribute('alt') || '',
                            src: img.getAttribute('src')
                        }));
                    }
                """)
                for image_data in images_data:
                    try:
                        # REVIEW: remove "Image" from alt text
                        image_data["alt"] = re.sub(r"Image", "", image_data["alt"])
                        updated_alt, image_url = image_data["alt"], image_data["src"]
                        if not image_url.startswith(("http://", "https://", "www.")):
                            image_url = urljoin(page.url, image_url)

                        # Add caption if available
                        if image_url in self.url2caption:
                            if self.url2caption[image_url] not in updated_alt:
                                if updated_alt:
                                    updated_alt = f"{updated_alt}, description: {self.url2caption[image_url]}"
                                else:
                                    updated_alt = f"description: {self.url2caption[image_url]}"
                        elif "data:image/svg" not in image_url and self.captioning_fn is not None:
                            print(f"WARNING: {image_url} not in self.url2caption")

                        # Add URL to alt text. # REVIEW: provide url even if no captioner.
                        if "url:" not in updated_alt:
                            if updated_alt:
                                updated_alt = f"{updated_alt}, url: {map_url_to_real(image_url)}"
                            else:
                                updated_alt = f"url: {map_url_to_real(image_url)}"

                        safe_updated_alt = json.dumps(updated_alt)
                        # REVIEW: the json serialization above and below
                        # is duplicating "\\" at every iteration due to double escaping.
                        # Below is a simple fix; check if double serialization is really necessary; if not, remove it.
                        safe_updated_alt = safe_updated_alt.replace('"', "")  # review: added this line

                        image_updates.append({"image_url": image_url, "updated_alt": safe_updated_alt})
                    except Exception as e:
                        print("WARNING:", e)

                # Execute the batch update
                js_code = """
                    (image_updates => {
                        const images = document.querySelectorAll('img');
                        const urlToImageMap = {};
                        images.forEach(img => {
                            urlToImageMap[img.src] = img;
                        });

                        image_updates.forEach(update => {
                            const img = urlToImageMap[update.image_url];
                            if (img) {
                                img.alt = update.updated_alt;
                            }
                        });
                    })(%s);
                """ % json.dumps(image_updates)
                page.evaluate(js_code)
                if self.observation_type == "accessibility_tree_with_captioner":
                    accessibility_tree = self.fetch_page_accessibility_tree(
                        browser_info,
                        client,
                        current_viewport_only=self.current_viewport_only,
                    )
                    content, obs_nodes_info = self.parse_accessibility_tree(accessibility_tree)
                    content = self.clean_accesibility_tree(content)
                    self.obs_nodes_info = obs_nodes_info
                    self.meta_data["obs_nodes_info"] = obs_nodes_info
                else:
                    content = ""  # Not used for SoM
            timer.end("ENV:process_txt_obs:captioning")
        else:
            raise ValueError(f"Invalid observation type: {self.observation_type}")

        self.browser_config = browser_info["config"]
        content = f"{tab_title_str}\n\n{content}"
        return content

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        node_info = self.obs_nodes_info[element_id]
        node_bound = node_info["union_bound"]
        x, y, width, height = node_bound
        center_x = x + width / 2
        center_y = y + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )


class ImageObservationProcessor(ObservationProcessor):
    def __init__(
        self,
        observation_type: str,
        viewport_size: Optional[ViewportSize] = None,
    ):
        self.observation_type = observation_type
        self.observation_tag = "image"
        self.viewport_size = viewport_size
        self.meta_data = create_empty_metadata()
        self.som_to_html_bbox: dict[str, int] = {}  # dict{str:int} where: key: unique SoM ids, value: fixed IDs from HTML
        self.plot_ids: list[int] | None = None  # list[int]: (Fixed) HTML IDs to draw bounding boxes for

    def set_plot_ids(self, ids_keep: list[str] = [], ids_skip: list[str] = []) -> None:
        # Only one of ids_keep or ids_skip can be provided.
        if ids_keep and ids_skip:
            raise ValueError("Only one of ids_keep or ids_skip can be provided")

        # At least one of ids_keep or ids_skip must be provided.
        if not ids_keep and not ids_skip:
            raise ValueError("Either ids_keep or ids_skip must be provided")

        # If ids_skip is provided, keep only the ids that are not in ids_skip.
        if ids_skip:
            self.plot_ids = [int(id) for id in self.som_to_html_bbox if id not in ids_skip]
        else:
            self.plot_ids = [self.som_to_html_bbox[id] for id in ids_keep]

    def get_page_bboxes(self, page: "Page") -> str:
        """Return a CSV string of bounding boxes + metadata for visible DOM elements on the page."""
        js_script = """
        (() => {
            // Helper functions
            const isDisabled = (el) => el.hasAttribute("disabled") || el.getAttribute("aria-disabled") === "true";
            const isHidden = (el) => {
                const cs = window.getComputedStyle(el);
                const cssHidden = cs.display === "none" || cs.visibility === "hidden";
                return cssHidden;
                
                // Note: Uncomment below if wants to omit aria-hidden elements; note some relevant elements may not receive boxes.                
                // Allow IMG elements to be considered visible even if aria-hidden="true" so clickable thumbnails still get boxes.
                // const ariaHidden = el.getAttribute("aria-hidden") === "true";
                // if (el.tagName === 'IMG') {
                //    return cssHidden; // ignore aria-hidden for IMG elements
                // }
                // return ariaHidden || cssHidden;
            };
            
            const isElementOrAncestorHidden = (el) => {
                while (el) {
                    if (isHidden(el)) return true;
                    el = el.parentElement;
                }
                return false;
            };

            // Returns true if an option within a dropdown is visible in the screen, so bboxes are only added for options that are visible.
            const pointInRect = (x, y, r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
            const isCenterVisibleInScrollContainers = (el, rect) => {
                const cx = rect.left + rect.width / 2;
                const cy = rect.top + rect.height / 2;
                let node = el.parentElement;
                while (node && node !== document.body) {
                    const style = window.getComputedStyle(node);
                    const hasScrollClip = /(auto|scroll|hidden|clip)/.test(style.overflow + style.overflowY + style.overflowX);
                    if (hasScrollClip) {
                        const crect = node.getBoundingClientRect();
                        if (!pointInRect(cx, cy, crect)) return false;
                    }
                    node = node.parentElement;
                }
                return true;
            };
            // Detect visible modal/overlay that blocks clicks outside it
            const blockingModals = Array.from(document.querySelectorAll('.modal-popup._show, .modal-backdrop'))
                .filter(el => {
                    const style = window.getComputedStyle(el);
                    const r = el.getBoundingClientRect();
                    return (
                        style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        r.width > 20 &&
                        r.height > 20 &&
                        r.bottom > 0 && r.right > 0 &&
                        r.top < window.innerHeight && r.left < window.innerWidth
                    );
                });
            const pageBlocked = blockingModals.length > 0;
            const isElementInsideModal = (el) => {
                return blockingModals.some(modal => modal.contains(el));
            };

            // Interactable selectors
            const interactableSelectors = [
                'a[href]:not(:has(img:not([aria-hidden="true"])))', 'a[href] img', 'button',
                'input:not([type="hidden"]):not(.hidden)', 'textarea', 'select',
                '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                '[role="checkbox"]', '[role="menuitem"]', '[role="option"]', '[role="tab"]', '[draggable="true"]',
                '.btn', 'option', '.select2-results__option', 'a[href="/notifications"]', 'a[href*="/submit"]',
                '.fa.fa-star.is-rating-item', 'input[type="checkbox"]:not(.hidden)', 'div.review-control-vote label'
            ];

            // Text selectors (non-interactable leaves)
            const textSelectors = ['p','span','div:not(:has(*))','h1','h2','h3','h4','h5','h6','li','article','img','label'];
            const modifiedTextSelectors = textSelectors.map(selector =>
                `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
            );

            const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
            const elements = document.querySelectorAll(combinedSelectors.join(', '));

            const pixelRatio = window.devicePixelRatio;
            let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
            let counter = 1;

            elements.forEach(element => {
                //------------------------------------------------
                // 1) Filter out elements
                //------------------------------------------------
                // Exclude element with href "https://postmill.xyz/" (invalid URL)
                if (element.getAttribute && element.getAttribute("href") === "https://postmill.xyz/") return;

                if (isDisabled(element) || isElementOrAncestorHidden(element)) return;

                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                if (rect.bottom <= 0 || rect.top >= window.innerHeight || rect.right <= 0 || rect.left >= window.innerWidth) return;

                // Skip injected dropdown bits (these are handled below so the IDs follow the corresponding SELECT element)
                if (element.classList && element.classList.contains('injected-option')) return;
                if (element.closest && element.closest('.injected-dropdown')) return;

                // ensure center visible within scrollable parents for select/options
                if (element.tagName === 'SELECT' || element.matches('[role="option"], .select2-results__option')) {
                    if (!isCenterVisibleInScrollContainers(element, rect)) return;
                }
                //------------------------------------------------
                // 2) Get the text content for the element
                //------------------------------------------------

                // let textContent = element.textContent ? element.textContent.trim() : "";
                let textContent = element.innerText ? element.innerText.trim() : "";
                if (!textContent) {
                    textContent = element.textContent ? element.textContent.trim() : "";
                }
                if (!textContent) {
                    textContent = element.getAttribute("aria-label") ? element.getAttribute("aria-label").trim() : "";
                }

                // Special handling for inputs/textarea/select
                if ((element.tagName === 'INPUT' || element.tagName === 'TEXTAREA' || element.tagName === 'SELECT')) {
                    if (element.type === 'checkbox' || element.type === 'radio') {
                        textContent = element.checked ? 'checked' : 'unchecked';
                    } else if (
                        element.type === 'text' || element.type === 'email' || element.type === 'password' ||
                        element.type === 'search' || element.type === 'url' || element.type === 'tel' ||
                        element.type === 'number' || element.tagName === 'TEXTAREA'
                    ) {
                        if (element.value) textContent = element.value;

                    } else if (element.type === 'file' && !element.classList.contains('hidden') &&
                            element.style.display !== 'none' && element.offsetWidth > 0 && element.offsetHeight > 0) {
                        textContent = (element.files && element.files.length > 0)
                            ? Array.from(element.files).map(f => f.name).join(', ')
                            : '';

                    } else if (element.tagName === 'SELECT') {
                        if (element.selectedIndex >= 0 && element.options[element.selectedIndex]) {
                            textContent = (element.options[element.selectedIndex].text || '').trim();
                        } else {
                            textContent = '';
                        }
                    }
                }
                // Ignore script-y text blobs
                if (/^require\\s*\\(\\s*\\[/.test(textContent) || /<!\\[CDATA\\[/.test(textContent)) {
                    textContent = "";
                }
                // Get alt text from element or nested images.
                let altText = element.getAttribute && element.getAttribute("alt") ? element.getAttribute("alt").trim() : "";
                if (!altText) {
                    const nestedImg = element.querySelector && element.querySelector('img[alt]');
                    if (nestedImg) altText = nestedImg.getAttribute("alt").trim();
                }
                altText = altText || textContent;

                // Treat <label for="..."> of a hidden radio/checkbox as interactive
                let isLabelForHiddenInput = false;
                if (element.tagName === 'LABEL' && element.getAttribute('for')) {
                    const associatedInput = document.getElementById(element.getAttribute('for'));
                    if (associatedInput && (associatedInput.type === 'radio' || associatedInput.type === 'checkbox') &&
                        (associatedInput.classList.contains('hidden') || associatedInput.style.display === 'none' ||
                        associatedInput.offsetWidth === 0 || associatedInput.offsetHeight === 0)) {
                        isLabelForHiddenInput = true;
                    }
                }

                //------------------------------------------------
                // 3) Fill csv data
                //------------------------------------------------
                textContent = String(textContent).replace(/"/g, '');
                altText = String(altText).replace(/"/g, '');

                // --- Determine interactability ---
                const isInteractable =
                (!pageBlocked || isElementInsideModal(element)) &&
                (interactableSelectors.some(selector => element.matches(selector)) || isLabelForHiddenInput);

                const classList = element.className || '';
                const id = element.id || '';

                const dataString = [
                    counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                    (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                    (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                    altText, classList, id, textContent, isInteractable
                ].map(value => `"${value}"`).join(",");

                csvContent += dataString + "\\n";
                counter++;

                // Special logic for injected `option` elements for `select` elements. See comment in `inject_custom_dropdowns`.
                // Obs.: handling `option` elements here instead of the regular loop ensures that the ID of the options follow the ID of the corresponding SELECT element.
                if (element.tagName === 'SELECT') {
                    const selectId = element.getAttribute('data-select-id');
                    if (selectId) {
                        const dropdown = document.querySelector(`.injected-dropdown[data-select-id="${selectId}"]`);
                        if (dropdown) {
                            const options = dropdown.querySelectorAll('.injected-option');
                            options.forEach(opt => {
                                const optRect = opt.getBoundingClientRect();
                                if (optRect.width === 0 || optRect.height === 0) return;

                                // Skip options scrolled out of view within the dropdown
                                const isInteractable = isCenterVisibleInScrollContainers(opt, optRect);

                                const optText = opt.textContent.trim().replace(/"/g, '');
                                const optionDataString = [
                                    counter, 'OPTION', (optRect.top + window.scrollY) * pixelRatio,
                                    (optRect.right + window.scrollX) * pixelRatio, (optRect.bottom + window.scrollY) * pixelRatio,
                                    (optRect.left + window.scrollX) * pixelRatio, optRect.width * pixelRatio, optRect.height * pixelRatio,
                                    optText, '', '', optText, isInteractable
                                ].map(value => `"${value}"`).join(",");

                                csvContent += optionDataString + "\\n";
                                counter++;
                            });
                        }
                    }
                }
            });

            return csvContent;
        })();
        """
        # Execute in the page and return the CSV content
        return page.evaluate(js_script)

    def draw_bounding_boxes(
        self,
        data_string,
        screenshot_img,
        viewport_size=None,
        add_ids=True,
        bbox_color=None,
        min_width=8,
        min_height=8,
        bbox_padding=0,
        bbox_border=2,
        plot_ids: list[int] | None = None,
    ):
        """
        min_width and min_height: Minimum dimensions of the bounding box to be plotted.
        """
        # Read CSV data
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
        df["Area"] = df["Width"] * df["Height"]
        # Remove bounding boxes that are clipped.
        b_x, b_y = (self.browser_config["win_left_bound"], self.browser_config["win_upper_bound"])
        if viewport_size is not None:
            df = df[(df["Bottom"] - b_y >= 0) & (df["Top"] - b_y <= viewport_size["height"]) & (df["Right"] - b_x >= 0) & (df["Left"] - b_x <= viewport_size["width"])]
            viewport_area = viewport_size["width"] * viewport_size["height"]
            # Filter out bounding boxes that too large (more than 80% of the viewport)
            df = df[df["Area"] <= 0.8 * viewport_area]

        # Open the screenshot image
        img = screenshot_img.copy()
        draw = ImageDraw.Draw(img)

        # Load a TTF font with a larger size
        font_path = "media/SourceCodePro-SemiBold.ttf"
        font_size, padding = 16, 2
        font = ImageFont.truetype(font_path, font_size)

        # Create a color cycle using one of the categorical color palettes in matplotlib
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bbox_id2visid = {}
        bbox_id2desc = {}
        index = 0
        id2center = {}
        existing_text_rectangles = []
        text_to_draw = []
        # Provide [id] textContent inputs to the model as text.
        text_content_elements = []
        text_content_text = set()  # Store text of interactable elements
        id2semantic = {}
        id2bbox = {}

        # Iterate through each row in the CSV and draw bounding boxes
        for _, row in df.iterrows():
            if not row["Interactable"]:
                content = ""
                # Add image alt-text to the text representation.
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += re.sub(r"\s+", " ", row["Alt"]).strip()
                # Add HTML textContent (if any) to the text representation.
                if pd.notna(row["TextContent"]):
                    # Limit to 200 characters to avoid having too much text
                    content += re.sub(r"\s+", " ", row["TextContent"]).strip()[:200]

                # Check if the text is a CSS selector
                if content and not (content.startswith(".") and "{" in content):
                    # Add elements which are not interactable as StaticText
                    if content not in text_content_text:
                        text_content_elements.append(f"[] [StaticText] [{content}]")
                        text_content_text.add(content)
                continue

            # Skip bounding boxes that are not in the plot_ids list
            if (plot_ids is not None) and (row["ID"] not in plot_ids):
                continue

            unique_id = str(index + 1)
            bbox_id2visid[row["ID"]] = unique_id  # map the bounding box ID to the unique character ID
            top, right, bottom, left, width, height = (
                row["Top"],
                row["Right"],
                row["Bottom"],
                row["Left"],
                row["Width"],
                row["Height"],
            )
            left, right, top, bottom = left - b_x, right - b_x, top - b_y, bottom - b_y
            id2center[unique_id] = ((left + right) / 2, (bottom + top) / 2, width, height)

            if width >= min_width and height >= min_height:
                # Get the next color in the cycle
                color = bbox_color or color_cycle[index % len(color_cycle)]
                draw.rectangle(
                    [
                        left - bbox_padding,
                        top - bbox_padding,
                        right + bbox_padding,
                        bottom + bbox_padding,
                    ],
                    outline=color,
                    width=bbox_border,
                )
                bbox_id2desc[row["ID"]] = color
                id2bbox[unique_id] = {
                    "bbox": [left, top, right, bottom],
                    "color": color,
                    "padding": bbox_padding,
                    "border": bbox_border,
                }

                # Draw the text on top of the rectangle
                if add_ids:
                    # Calculate list of possible text positions
                    text_positions = [
                        (left - font_size, top - font_size),  # Top-left corner
                        (
                            left,
                            top - font_size,
                        ),  # A little to the right of the top-left corner
                        (right, top - font_size),  # Top-right corner
                        (
                            right - font_size - 2 * padding,
                            top - font_size,
                        ),  # A little to the left of the top-right corner
                        (left - font_size, bottom),  # Bottom-left corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                    ]
                    text_width = draw.textlength(unique_id, font=font)
                    text_height = font_size  # Assume the text is one line

                    if viewport_size is not None:
                        for text_position in text_positions:
                            new_text_rectangle = [
                                text_position[0] - padding,
                                text_position[1] - padding,
                                text_position[0] + text_width + padding,
                                text_position[1] + text_height + padding,
                            ]

                            # Check if the new text rectangle is within the viewport
                            if new_text_rectangle[0] >= 0 and new_text_rectangle[1] >= 0 and new_text_rectangle[2] <= viewport_size["width"] and new_text_rectangle[3] <= viewport_size["height"]:
                                # If the rectangle is within the viewport, check for overlaps
                                overlaps = False
                                for existing_rectangle in existing_text_rectangles:
                                    if self.rectangles_overlap(
                                        new_text_rectangle,
                                        existing_rectangle,
                                        padding * 2,
                                    ):
                                        overlaps = True
                                        break

                                if not overlaps:
                                    break
                            else:
                                # If the rectangle is outside the viewport, try the next position
                                continue
                    else:
                        # If none of the corners work, move the text rectangle by a fixed amount
                        text_position = (
                            text_positions[0][0] + padding,
                            text_positions[0][1],
                        )
                        new_text_rectangle = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding,
                        ]

                    existing_text_rectangles.append(new_text_rectangle)  # type: ignore
                    text_to_draw.append((new_text_rectangle, text_position, unique_id, color))  # type: ignore

                    # Save text rectangle and position
                    if unique_id in id2bbox:
                        id2bbox[unique_id]["text_rect"] = new_text_rectangle  # type: ignore
                        id2bbox[unique_id]["text_position"] = list(text_position)  # type: ignore

                    content = ""
                    if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                        content += re.sub(r"\s+", " ", row["Alt"]).strip()
                    if pd.notna(row["TextContent"]):
                        # Limit to 200 characters
                        content += re.sub(r"\s+", " ", row["TextContent"]).strip()[:200]

                    # Show labels associated with hidden inputs as the input type for better UX
                    element_display = row["Element"]
                    input_type = row.get("InputType", None)
                    if row["Element"] == "LABEL" and pd.notna(input_type) and input_type:
                        element_display = input_type.upper()

                    # Check if this element has label text and add it to semantic info
                    if pd.notna(row.get("LabelText", "")) and row.get("LabelText", "").strip():
                        label_text = row["LabelText"].strip()
                        text_content_elements.append(f"[{unique_id}] [{element_display}, label: {label_text}] [{content}]")
                        id2semantic[unique_id] = f"[{element_display}] element with label '{label_text}' and content [{content}]"
                    else:
                        text_content_elements.append(f"[{unique_id}] [{element_display}] [{content}]")
                        id2semantic[unique_id] = f"[{element_display}] element with content [{content}]"

                    # Remove text_content_elements with content
                    if content in text_content_text:
                        text_content_elements = [element for element in text_content_elements if element.strip() != content]
                    text_content_text.add(content)

            index += 1

        for text_rectangle, text_position, unique_id, color in text_to_draw:
            # Draw a background rectangle for the text
            draw.rectangle(text_rectangle, fill=color)
            draw.text(text_position, unique_id, font=font, fill="white")

        content_str = "\n".join(text_content_elements)
        self.som_to_html_bbox = {v: k for k, v in bbox_id2visid.items()}  # map the SoM unique character ID to the JS bbox box ID
        self.plot_ids = None
        return img, id2center, content_str, id2semantic, id2bbox

    def rectangles_overlap(self, rect1, rect2, padding):
        """
        Check if two rectangles overlap.
        Each rectangle is represented as a list [x1, y1, x2, y2].
        """
        return not (rect1[2] < rect2[0] + padding or rect1[0] > rect2[2] - padding or rect1[1] > rect2[3] - padding or rect1[3] < rect2[1] + padding)

    @timeit(custom_name="ENV:process_img_obs")
    def process(self, page: Page, client: CDPSession) -> tuple[np.ndarray, np.ndarray, str]:
        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=2500)
            browser_info = self.fetch_browser_info(page, client)

        self.browser_config = browser_info["config"]

        # NOTE[mandrade]: inject custom dropdowns to allow for click on options of <select> elements
        self.inject_custom_dropdowns(page)
        if self.observation_type == "image_som":
            # Produce the SoM image, with bounding boxes
            try:
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str, id2semantic, id2bbox = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                    plot_ids=self.plot_ids,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                self.meta_data["obs_nodes_semantic_info"] = id2semantic
                self.meta_data["obs_nodes_bbox_info"] = id2bbox
                screenshot_som = np.array(bbox_img)
                return screenshot_som, np.array(screenshot_img), content_str
            except Exception as _:
                page.wait_for_event("load")
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str, id2semantic, id2bbox = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                self.meta_data["obs_nodes_semantic_info"] = id2semantic
                self.meta_data["obs_nodes_bbox_info"] = id2bbox
                screenshot_som = np.array(bbox_img)
                return screenshot_som, np.array(screenshot_img), content_str
        else:
            try:
                screenshot = png_bytes_to_numpy(page.screenshot())
            except Exception as _:
                page.wait_for_event("load")
                screenshot = png_bytes_to_numpy(page.screenshot())
            return screenshot, screenshot, ""

    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ) -> BrowserInfo:
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )
        assert self.viewport_size is not None, "viewport_size is not set."

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        config: BrowserConfig = page.evaluate("""
            () => {
                return {
                    win_upper_bound: window.pageYOffset,
                    win_left_bound: window.pageXOffset,
                    win_width: window.screen.width,
                    win_height: window.screen.height,
                    win_right_bound: window.pageXOffset + window.screen.width,
                    win_lower_bound: window.pageYOffset + window.screen.height,
                    device_pixel_ratio: window.devicePixelRatio
                }
            }
        """)

        assert config["device_pixel_ratio"] == 1.0, "devicePixelRatio is not 1.0"
        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    def get_element_center(self, element_id: str) -> tuple[float, float]:
        if not self.observation_type == "image_som":
            raise ValueError("get_element_center() is only supported for 'image_som' observation type.")

        center_x, center_y, width, height = self.som_id_info[element_id]
        assert self.viewport_size is not None, "viewport_size is not set."
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )

    def inject_custom_dropdowns(self, page: Page):
        # NOTE[mandrade]: added to allow for clicking on <select> options
        """
        Workaround to interact with <select> elements using only (x, y) clicks.
        Creates shadow dropdowns so Playwright can simulate real user clicks,
        since native dropdowns are OS-rendered and not directly clickable.
        """
        js_script = """
        () => {
            function updateAllDropdownPositions() {
                document.querySelectorAll('.injected-dropdown').forEach(dd => {
                    const sid = dd.getAttribute('data-select-id');
                    const sel = document.querySelector(`select[data-select-id="${sid}"]`);
                    if (sel) {
                        const r = sel.getBoundingClientRect();
                        dd.style.top = `${r.bottom + window.scrollY}px`;
                        dd.style.left = `${r.left + window.scrollX}px`;
                        dd.style.width = `${r.width}px`;
                    }
                });
            }
            updateAllDropdownPositions();
            document.querySelectorAll('select:not([disabled]):not([aria-hidden="true"]):not([hidden]):not([data-injected-dropdown])').forEach((select) => {
                // Global ID counter
                if (window.__customDropdownNextId === undefined) {
                    window.__customDropdownNextId = 0;
                }
                const selectId = window.__customDropdownNextId++;  // Use globally incrementing ID
                const rect = select.getBoundingClientRect();
                const computedStyle = window.getComputedStyle(select);

                // Store computed styles
                const fontSizeStr = computedStyle.fontSize;
                const lineHeightStr = computedStyle.lineHeight;
                const fontFamilyStr = computedStyle.fontFamily;

                // Option styling constants
                const optionPaddingVertical = 4;  // pixels
                const optionPaddingHorizontal = 8; // pixels
                const optionPaddingStr = `${optionPaddingVertical}px ${optionPaddingHorizontal}px`;

                // Calculate option height from computed styles
                const fontSize = parseFloat(fontSizeStr) || 16;
                const lineHeight = parseFloat(lineHeightStr) || fontSize * 1.2;
                const optionHeight = lineHeight + (optionPaddingVertical * 2); // top + bottom

                // Calculate max height dynamically
                const availableSpaceBelow = window.innerHeight - rect.bottom;
                const selectSize = parseInt(select.getAttribute('size')) || 0;
                
                let maxHeight;
                if (selectSize > 1) {
                    // Honor the size attribute if present
                    maxHeight = selectSize * optionHeight;
                } else {
                    // Use available space, capped at 200px minimum and 70% of available space
                    maxHeight = Math.min(Math.max(availableSpaceBelow * 0.7, 200), 400);
                }

                // Create dropdown container
                const dropdown = document.createElement('div');
                dropdown.classList.add('injected-dropdown'); // clearly distinctive class
                dropdown.setAttribute('data-select-id', selectId); // Add unique identifier to link with select
                dropdown.setAttribute('tabindex', '-1'); // Make focusable for scroll actions
                dropdown.style.position = 'absolute';
                dropdown.style.top = `${rect.bottom + window.scrollY}px`;
                dropdown.style.left = `${rect.left + window.scrollX}px`;
                dropdown.style.width = `${rect.width}px`;
                dropdown.style.maxHeight = `${maxHeight}px`; // Dynamic height
                dropdown.style.overflowY = 'auto'; // Add scrollbar when needed
                dropdown.style.background = '#fff';
                dropdown.style.border = '1px solid #c2c2c2';
                dropdown.style.zIndex = 999999;
                dropdown.style.boxSizing = 'border-box';
                dropdown.style.padding = '0';
                dropdown.style.margin = '0';
                dropdown.style.fontFamily = fontFamilyStr;
                dropdown.style.fontSize = fontSizeStr;
                dropdown.style.lineHeight = lineHeightStr;
                dropdown.style.display = 'none'; // Initially hidden
                dropdown.style.outline = 'none'; // Remove focus outline
                
                // Add the same identifier to the select element
                select.setAttribute('data-select-id', selectId);
                select.setAttribute('data-injected-dropdown', 'true');

                // Populate options
                Array.from(select.options).forEach(option => {
                    const optionEl = document.createElement('div');
                    optionEl.textContent = option.textContent;
                    optionEl.classList.add('injected-option');
                    optionEl.setAttribute('role', 'option');
                    optionEl.setAttribute('data-select-id', selectId); // Add same identifier to options
                    optionEl.style.padding = optionPaddingStr;
                    optionEl.style.cursor = 'pointer';
                    optionEl.onmouseover = () => optionEl.style.backgroundColor = '#f0f0f0';
                    optionEl.onmouseout = () => optionEl.style.backgroundColor = '';

                    optionEl.onclick = () => {
                        select.value = option.value;
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                        dropdown.style.display = 'none';
                    };

                    dropdown.appendChild(optionEl);
                });

                // Insert dropdown into DOM
                document.body.appendChild(dropdown);

                // Use ONLY the mousedown event to prevent native dropdown and toggle custom one
                select.addEventListener('mousedown', (e) => {
                    e.preventDefault();  // Prevent native dropdown
                    e.stopPropagation();
                    const isOpening = dropdown.style.display === 'none';
                    dropdown.style.display = isOpening ? 'block' : 'none';
                    // Focus dropdown when opened so scroll actions work
                    if (isOpening) {
                        setTimeout(() => dropdown.focus(), 0);
                    }
                });
            });

            // Add global listeners only once to prevent memory leaks
            if (!window.__customDropdownsInitialized) {
                window.__customDropdownsInitialized = true;

                // Close dropdowns when clicking elsewhere
                document.addEventListener('click', (e) => {
                    document.querySelectorAll('.injected-dropdown').forEach(dd => {
                        const sid = dd.getAttribute('data-select-id');
                        const sel = document.querySelector(`select[data-select-id="${sid}"]`);
                        if (sel && !sel.contains(e.target) && !dd.contains(e.target)) {
                            dd.style.display = 'none';
                        }
                    });
                });

                // Reposition dropdowns on scroll and resize
                window.addEventListener('scroll', updateAllDropdownPositions, true);
                window.addEventListener('resize', updateAllDropdownPositions);
            }
        }
        """
        try:
            page.evaluate(js_script)
        except Exception as e:
            logger.warning(f"Error in inject_custom_dropdowns: {e}. Retrying after wait_for_page_to_stabilize.")
            wait_for_page_to_stabilize(page, return_early=True, min_num_trues=4, max_overall_timeout_seconds=3, logger=logger)
            page.evaluate(js_script)


class ObservationHandler:
    """Main entry point to access all observation processor"""

    def __init__(
        self,
        main_observation_type: str,
        text_observation_type: str,
        image_observation_type: str,
        current_viewport_only: bool,
        viewport_size: ViewportSize,
        captioning_fn=None,
    ) -> None:
        self.main_observation_type = main_observation_type
        self.text_processor = TextObervationProcessor(
            text_observation_type,
            current_viewport_only,
            viewport_size,
            captioning_fn,
        )
        self.image_processor = ImageObservationProcessor(image_observation_type, viewport_size)
        self.viewport_size = viewport_size

    def get_observation_space(self) -> spaces.Dict:
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )

        image_space = spaces.Box(
            # Each position stores the RGB values. Note the swapped axes (height first).
            np.zeros(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            ),
            np.ones(
                (self.viewport_size["height"], self.viewport_size["width"], 3),
                dtype=np.uint8,
            )
            * 255.0,
            dtype=np.uint8,
        )

        return spaces.Dict({"text": text_space, "image": image_space})

    def get_observation(self, page: Page, client: CDPSession) -> dict[str, Observation]:
        text_obs = self.text_processor.process(page, client)
        image_obs, raw_screenshot_obs, content_str = self.image_processor.process(page, client)
        self.text_processor.meta_data["open_tabs"] = self.text_processor.get_tab_info(page)
        self.image_processor.meta_data["open_tabs"] = self.text_processor.get_tab_info(page)

        if content_str != "":
            # NOTE[mandrade]: fixed bug; added tab info when text_obs comes from 'content_str'
            # obs: Adding this here, but ideally should revise sub-classes of ObservationProcessor.
            tab_title_str = self.text_processor.get_tab_info(page)
            text_obs = f"{tab_title_str}\n\n{content_str}"
        return {"text": text_obs, "image": image_obs, "raw_screenshot": raw_screenshot_obs}

    def get_observation_metadata(self) -> dict[str, ObservationMetadata]:
        return {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }

    @property
    def action_processor(self) -> ObservationProcessor:
        """Return the main processor that is associated with the action space"""
        if self.main_observation_type == "text":
            return self.text_processor
        elif self.main_observation_type == "image":
            return self.image_processor
        else:
            raise ValueError("Invalid main observation type")
