"""
Browser Env action space.
Inspited by Farama-Foundation/miniwob-plusplus
"""
# NOTE[mandrade]: some refactoring for better action parsing & execution, support for new actions (e.g.: select, ctrl+f), action outputs, etc
# NOTE[mandrade]: async implementation not stable; see VWA-Updates-Documentation.md

import ast
import random
import re
import string
import time
from enum import IntEnum
from itertools import chain
from typing import Any, Optional, TypedDict, cast

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from playwright._impl._api_structures import ViewportSize
from playwright.async_api import BrowserContext as ABrowserContext
from playwright.async_api import Locator as ALocator
from playwright.async_api import Page as APage
from playwright.sync_api import BrowserContext, ElementHandle, FileChooser, JSHandle, Locator, Page
from playwright.sync_api import TimeoutError as PWTimeoutError

from browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    MAX_ANSWER_LENGTH,
    MAX_ELEMENT_ID,
    MAX_ELEMENT_INDEX_IN_VIEWPORT,
    MAX_PAGE_NUMBER,
    MAX_VANILLA_STR_LENGTH,
    PLAYWRIGHT_ACTIONS,
    PLAYWRIGHT_LOCATORS,
    ROLES,
    SPECIAL_KEY_MAPPINGS,
    SPECIAL_KEYS,
    SPECIAL_LOCATORS,
    TEXT_MAX_LENGTH,
    TYPING_MAX_LENGTH,
    URL_MAX_LENGTH,
    RolesType,
)
from browser_env.env_utils import map_url_to_local, wait_for_page_to_stabilize
from browser_env.image_utils import any_to_path
from browser_env.processors import ObservationProcessor, TextObervationProcessor
from core_utils.logger_utils import logger


class ActionParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ParsedPlaywrightCode(TypedDict):
    function_name: str
    arguments: list[str]
    keywords: dict[str, Any]


class Action(TypedDict):
    action_type: int
    coords: npt.NDArray[np.float32]
    element_role: int
    element_name: str
    text: list[int]
    page_number: int
    url: str
    nth: int
    element_id: str
    direction: str
    key_comb: str
    pw_code: str
    answer: str
    raw_prediction: str  # raw prediction from the model
    element_center: Optional[tuple[float, float]]  # normalized center coordinates of the element
    action_output: Optional[Any]  # NOTE[mandrade]: allow actions to return output. E.g.: number of matches for ctrl+f
    wait_for: Optional[float]  # NOTE[mandrade]: allow actions to specify wait time after execution


class ActionTypes(IntEnum):
    """Valid action types for browser env."""

    NONE = 0
    # mouse wheel and keyboard, universal across all action spaces
    SCROLL = 1
    KEY_PRESS = 2

    # low level mouse and keyboard actions
    MOUSE_CLICK = 3
    KEYBOARD_TYPE = 4
    MOUSE_HOVER = 5

    # mid level mouse and keyboard actions
    CLICK = 6
    TYPE = 7
    HOVER = 8

    # page level actions, universal across all action spaces
    PAGE_FOCUS = 9
    NEW_TAB = 10
    GO_BACK = 11
    GO_FORWARD = 12
    GOTO_URL = 13
    PAGE_CLOSE = 14

    # high-leval actions that playwright support
    CHECK = 15
    SELECT_OPTION = 16

    STOP = 17
    CLEAR = 18
    UPLOAD = 19
    REFINE_INPUT = 99

    def __str__(self) -> str:
        return f"ACTION_TYPES.{self.name}"


# ===============================================================================
# NOTE[mandrade]: Helper functions
# ===============================================================================

# Shared JavaScript code for intelligent scroll targeting
_SCROLL_JS_CODE = """
(dir) => {
  const isScrollable = (el) => {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const overflowY = style.overflowY;
    if (!(overflowY === 'auto' || overflowY === 'scroll')) return false;
    return (el.scrollHeight || 0) > (el.clientHeight || 0);
  };

  const findScrollableInSubtree = (root) => {
    if (isScrollable(root)) return root;
    for (const child of root.children || []) {
      const result = findScrollableInSubtree(child);
      if (result) return result;
    }
    return null;
  };

  // Start from the focused element
  let el = document.activeElement;
  // If the focus is inside a shadow root, try to get its host
  if (el && el.shadowRoot && el.shadowRoot.activeElement) {
    el = el.shadowRoot.activeElement;
  }

  let target = null;
  
  // First, check if the focused element itself is scrollable
  if (isScrollable(el)) {
    target = el;
  }
  
  // Look for scrollable elements nearby by checking siblings at multiple levels,
  // but limit the search to stay within the same UI component
  let container = el;
  let levelsUp = 0;
  const MAX_LEVELS = 4; // Max depth of the DOM tree to search for scrollable elements
  
  while (container && container !== document.body && levelsUp < MAX_LEVELS && !target) {
    if (container.parentElement) {
      // Check siblings at this level
      for (const sibling of container.parentElement.children || []) {
        if (sibling !== container) {
          const scrollableInSibling = findScrollableInSubtree(sibling);
          if (scrollableInSibling) {
            target = scrollableInSibling;
            break;
          }
        }
      }
      
      // Stop early if we hit certain container boundaries that indicate we're leaving the component
      const containerClasses = container.className || '';
      if (containerClasses.includes('site-content') || 
          containerClasses.includes('main') || 
          containerClasses.includes('page-') ||
          container.tagName === 'MAIN' ||
          container.tagName === 'SECTION' ||
          container.tagName === 'ARTICLE') {
        break;
      }
    }
    container = container.parentElement;
    levelsUp++;
  }

  // Final fallback to document
  if (!target) {
    target = document.scrollingElement || document.body;
  }

  const delta = target === (document.scrollingElement || document.body) ? window.innerHeight : target.clientHeight;
  const amount = String(dir).toLowerCase().includes('up') ? -delta : delta;
  target.scrollTop = (target.scrollTop || 0) + amount;
}
"""

_GET_PAGE_STATE_JS = """
() => {
    // Serialize current form state
    const forms = document.querySelectorAll('form');
    let formState = '';
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (input.type === 'radio' || input.type === 'checkbox') {
                formState += input.id + ':' + input.checked + ';';
            } else if (input.tagName === 'SELECT') {
                formState += input.id + ':' + input.selectedIndex + ';';
            } else {
                formState += input.id + ':' + input.value + ';';
            }
        });
    });
    
    // Check for class or attribute changes on interactive elements
    const interactables = document.querySelectorAll('a, button, input, select, textarea, [role="button"]');
    let attrState = '';
    interactables.forEach(el => {
        attrState += el.className + el.getAttribute('aria-checked') + el.getAttribute('aria-selected');
    });
    
    return formState + '|' + attrState;
}
"""


def _dispose_js_handle(handle: JSHandle | None) -> None:
    if handle is None:
        return
    try:
        handle.dispose()
    except Exception:
        pass


def get_select_element_at(left: float, top: float, page: Page) -> ElementHandle | None:
    """
    Return the actual <select> element (or equivalent) at a given normalized coordinate.
    Handles both native <select> and JS-based dropdowns like Select2.
    """
    el, handle = get_element_at(left, top, page)
    if not el:
        return None

    try:
        # Try to resolve to a true <select> via JS context
        resolved = page.evaluate_handle(
            """(el) => {
                if (!el) return null;

                // --- Case 1: native <select> ---
                if (el.tagName === 'SELECT') return el;

                // --- Case 2: inside a Select2 container ---
                const select2Container = el.closest('.select2-container');
                if (select2Container) {
                    const hiddenSelect = select2Container.previousElementSibling;
                    if (hiddenSelect && hiddenSelect.tagName === 'SELECT') return hiddenSelect;
                }

                // --- Case 3: Choices.js or TomSelect ---
                const choices = el.closest('.choices');
                if (choices) {
                    const hiddenSelect = choices.querySelector('select');
                    if (hiddenSelect) return hiddenSelect;
                }

                // --- Case 4: role-based combobox (React, MUI, etc.) ---
                const roleCombo = el.closest('[role="combobox"]');
                if (roleCombo && roleCombo.previousElementSibling?.tagName === 'SELECT') {
                    return roleCombo.previousElementSibling;
                }

                // --- Case 5: <span> or <div> that directly labels a hidden <select> sibling ---
                if (el.nextElementSibling?.tagName === 'SELECT') return el.nextElementSibling;
                if (el.previousElementSibling?.tagName === 'SELECT') return el.previousElementSibling;

                return null;
            }""",
            el,
        )

        sel_el = resolved.as_element()
        return sel_el

    finally:
        _dispose_js_handle(handle)


def get_element_at(left: float, top: float, page: Page, dispose_js_handle: bool = False) -> tuple[ElementHandle | None, JSHandle | None]:
    """Return the JSHandle and ElementHandle at (left, top) coordinates in the viewport."""
    try:
        vp = page.viewport_size
        assert vp, "Viewport size not set"
        x = left * vp["width"]
        y = top * vp["height"]
        handle = page.evaluate_handle("([x,y]) => document.elementFromPoint(x,y)", [x, y])
        el = handle.as_element()
        if dispose_js_handle:
            _dispose_js_handle(handle)
        return el, handle
    except Exception:
        return None, None


def resolve_file_input_at(left: float, top: float, page: Page) -> ElementHandle | None:
    vp = page.viewport_size
    assert vp
    x, y = left * vp["width"], top * vp["height"]

    h = page.evaluate_handle(
        """
    ([x,y]) => {
      const at = document.elementFromPoint(x,y);
      const isFile = el => el && el.tagName?.toLowerCase() === 'input' && el.type === 'file';

      const fromLabel = (label) => {
        if (!label) return null;
        if (label.control && isFile(label.control)) return label.control;
        const contained = label.querySelector?.('input[type="file"]');
        if (contained) return contained;
        if (label.htmlFor) {
          const byFor = document.getElementById(label.htmlFor);
          if (isFile(byFor)) return byFor;
        }
        return null;
      };

      const resolve = (el) => {
        if (!el) return null;

        if (isFile(el)) return el;

        // up through ancestors, catch <label> wrappers
        for (let n = el; n; n = n.parentElement) {
          if (n.tagName?.toLowerCase() === 'label') {
            const r = fromLabel(n);
            if (r) return r;
          }
        }

        // input inside the clicked element
        const inside = el.querySelector?.('input[type="file"]');
        if (inside) return inside;

        // aria-controls indirection
        const ac = el.getAttribute?.('aria-controls');
        if (ac) {
          for (const id of ac.split(/\\s+/)) {
            const t = document.getElementById(id);
            if (isFile(t)) return t;
          }
        }

        // labels pointing to THIS element (edge case)
        if (el.id) {
          const labs = document.querySelectorAll(`label[for="${el.id}"]`);
          for (const lb of labs) {
            const r = fromLabel(lb);
            if (r) return r;
          }
        }
        return null;
      };

      return resolve(at);
    }
    """,
        [x, y],
    )

    try:
        el = h.as_element() if h else None
        if not el:
            return None
        # sanity check
        tag = el.evaluate("e => e.tagName.toLowerCase()")
        typ = el.get_attribute("type") or ""
        return el if (tag == "input" and typ.lower() == "file") else None
    finally:
        try:
            h.dispose()
        except Exception:
            pass


def get_file_input_handle_at(left: float, top: float, page: Page, behavior_timeout_ms: int = 200) -> ElementHandle | None:
    # Fast static pass (direct or proxy)
    el = resolve_file_input_at(left, top, page)
    if el:
        return el

    # As an optional safety net, a tiny behavioral wait:
    if behavior_timeout_ms > 0:
        vp = page.viewport_size
        assert vp
        x, y = left * vp["width"], top * vp["height"]
        try:
            with page.expect_file_chooser(timeout=behavior_timeout_ms) as fc_info:
                page.mouse.click(x, y)
            fc = fc_info.value
            if isinstance(fc, FileChooser) and (fc.element is None or (fc.element.get_attribute("type") or "").lower() == "file"):
                # Click has already happened; caller can proceed to set files via fc.element or ignore and do nothing else.
                # Return the underlying element if present, so caller can standardize on ElementHandle.
                return fc.element if fc.element else None
        except PWTimeoutError:
            pass
    return None


def is_in_viewport(element: Locator, viewport: ViewportSize, threshold: float = 0.3) -> bool:
    """Given a playwright locator, check if it is in the viewport"""
    box = element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(0, min(boxy1, viewporty1) - max(boxy0, viewporty0))
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


async def async_is_in_viewport(element: ALocator, viewport: ViewportSize, threshold: float = 0.3) -> bool:
    box = await element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(0, min(boxy1, viewporty1) - max(boxy0, viewporty0))
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


def _parse_select_arg(s: str) -> dict[str, Any]:
    """
    Accepts any of:
      "value=ca"     -> {"value": "ca"}
      "label=Canada" -> {"label": "Canada"}
      "index=1"      -> {"index": 1}
      "Canada"       -> {"label": "Canada"}  (fallback)
    """
    s = s.strip()
    if s.startswith("value="):
        return {"value": s[len("value=") :]}
    if s.startswith("label="):
        return {"label": s[len("label=") :]}
    if s.startswith("index="):
        try:
            return {"index": int(s[len("index=") :])}
        except ValueError:
            pass
    # default: treat raw string as label
    return {"label": s}


def focus_by_coords(x: float, y: float, page: Page) -> None:
    viewport_size = page.viewport_size
    assert viewport_size
    try:
        center = [x * viewport_size["width"], y * viewport_size["height"]]
        page.evaluate(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                if (el && typeof el.focus === 'function') {
                    el.focus();
                    return true;
                }
                return false;
            }""",
            center,
        )
    except Exception:
        execute_mouse_click(x, y, page)


async def a_focus_by_coords(x: float, y: float, page: APage) -> None:
    viewport_size = page.viewport_size
    assert viewport_size
    center = (x * viewport_size["width"], y * viewport_size["height"])
    try:
        await page.evaluate(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                if (el && typeof el.focus === 'function') {
                    el.focus();
                    return true;
                }
                return false;
            }""",
            center,
        )
    except Exception:
        await aexecute_mouse_click(x, y, page)


def _replace_url(match: re.Match) -> str:
    url = match.group(0)
    # NOTE: can add other normalizations
    # normalized = normalize_url(url)

    # Maps URL to local counterpart if there is one; otherwise, returns the original URL.
    local_url = map_url_to_local(url)
    return local_url


def normalize_urls_in_string(text: str) -> str:
    """
    Finds any URL substrings in 'text' (identified as starting with http(s):// or www.)
    and replaces them with a normalized, local-mapped version.
    """
    import re

    # matches URLs starting with http://, https://, or www.
    pattern = r"(http[s]?://\S+|www\.\S+)"

    return re.sub(pattern, _replace_url, text)


def find_in_page(page, text) -> int:
    js = """
(needle) => {
  if (!needle) return 0;
  const toLower = s => String(s).toLowerCase();
  const n = toLower(needle);
  if (!n) return 0;

  const isVisible = (el) => {
    if (!el) return false;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity || '1') === 0) return false;
    return true;
  };

  const countOccurrences = (haystack, needle) => {
    if (!needle) return 0;
    let count = 0;
    let pos = 0;
    const h = toLower(haystack);
    const nn = needle;
    while (true) {
      const idx = h.indexOf(nn, pos);
      if (idx === -1) break;
      count++;
      pos = idx + nn.length;
    }
    return count;
  };

  let total = 0;

  const countInRoot = (root) => {
    try {
      const base = root.body || root;
      const walker = root.createTreeWalker(base, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        const parent = node.parentElement;
        if (!parent || !isVisible(parent)) continue;
        total += countOccurrences(String(node.nodeValue || ''), n);
      }
      const nodes = base.querySelectorAll ? base.querySelectorAll('*') : [];
      for (const el of nodes) {
        if (el.shadowRoot) countInRoot(el.shadowRoot);
      }
    } catch (e) {}
  };

  countInRoot(document);
  const frames = document.querySelectorAll('iframe,frame');
  for (const f of frames) {
    try {
      if (f.contentDocument) countInRoot(f.contentDocument);
    } catch (e) {}
  }

  return total;
}
"""
    return page.evaluate(js, text)


def find_and_highlight_in_page(page, text) -> int:
    js = """
(needle) => {
  if (!needle) return 0;
  const toLower = s => String(s).toLowerCase();
  const n = toLower(needle);
  if (!n) return 0;

  // Remove existing highlights
  document.querySelectorAll("mark.__customHighlight").forEach(el => {
    const parent = el.parentNode;
    if (parent) {
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    }
  });

  const isVisible = (el) => {
    if (!el) return false;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity || '1') === 0) return false;
    return true;
  };

  let total = 0;

  const highlightInNode = (node, needle) => {
    const text = node.nodeValue;
    const lower = text.toLowerCase();
    let start = 0;
    let idx;
    const frag = document.createDocumentFragment();

    while ((idx = lower.indexOf(needle, start)) !== -1) {
      if (idx > start) {
        frag.appendChild(document.createTextNode(text.slice(start, idx)));
      }
      const mark = document.createElement("mark");
      mark.className = "__customHighlight"; // tag so we can clean later
      mark.textContent = text.slice(idx, idx + needle.length);
      frag.appendChild(mark);
      total++;
      start = idx + needle.length;
    }

    if (start < text.length) {
      frag.appendChild(document.createTextNode(text.slice(start)));
    }

    node.parentNode.replaceChild(frag, node);
  };

  const processRoot = (root) => {
    try {
      const base = root.body || root;
      const walker = root.createTreeWalker(base, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);

      for (const node of nodes) {
        const parent = node.parentElement;
        if (!parent || !isVisible(parent)) continue;
        if (node.nodeValue.toLowerCase().includes(n)) {
          highlightInNode(node, n);
        }
      }

      const els = base.querySelectorAll ? base.querySelectorAll('*') : [];
      for (const el of els) {
        if (el.shadowRoot) processRoot(el.shadowRoot);
      }
    } catch (e) {}
  };

  processRoot(document);

  const frames = document.querySelectorAll('iframe,frame');
  for (const f of frames) {
    try {
      if (f.contentDocument) processRoot(f.contentDocument);
    } catch (e) {}
  }

  return total;
}
"""
    return page.evaluate(js, text)


def clear_highlights(page):
    js = """
() => {
  const marks = document.querySelectorAll("mark.__customHighlight");
  for (const el of marks) {
    const parent = el.parentNode;
    if (!parent) continue;
    // Replace the <mark> with a plain text node
    parent.replaceChild(document.createTextNode(el.textContent), el);
    parent.normalize(); // merge adjacent text nodes
  }
  return marks.length; // how many highlights were removed
}
"""
    return page.evaluate(js)


def _page_content_changed(page: Page, initial_content: str, previous_state) -> bool:
    """Check if page content or state has changed.

    This checks both static HTML content and dynamic DOM state including:
    - Form element states (radio, checkbox, select)
    - Attributes and classes
    - Visual changes
    """
    try:
        # First check: static HTML content
        curr_content = page.content()
        if curr_content != initial_content:
            return True

        # Second check: dynamic DOM state (form elements, attributes, etc.) that don't modify the static HTML
        curr_state = page.evaluate(_GET_PAGE_STATE_JS)

        return curr_state != previous_state

    except Exception as e:
        if "page is navigating and changing the content" in str(e):
            return True
        else:
            logger.debug("Waiting for page to stabilize during action execution for content change check.")
            wait_for_page_to_stabilize(page, return_early=True, min_num_trues=4, max_overall_timeout_seconds=1, logger=logger)
            # Retry after stabilization
            try:
                curr_content = page.content()
                return curr_content != initial_content
            except Exception:
                return False


def safe_get_page_content(page: Page, max_retries: int = 1) -> str:
    retries = 0
    while retries < max_retries:
        try:
            return page.content()
        except Exception as _:
            wait_for_page_to_stabilize(page, return_early=True, min_num_trues=4, max_overall_timeout_seconds=1, logger=logger)
        retries += 1
    return ""


def safe_get_page_state(page: Page, max_retries: int = 1) -> str:
    retries = 0
    while retries < max_retries:
        try:
            return page.evaluate(_GET_PAGE_STATE_JS)
        except Exception as _:
            wait_for_page_to_stabilize(page, return_early=True, min_num_trues=4, max_overall_timeout_seconds=1, logger=logger)
            retries += 1
    return ""


# ==============================================================================
# Create action functions
# ==============================================================================


def is_equivalent(a: Action, b: Action) -> bool:
    """Return True if two actions are equal."""
    if a["action_type"] != b["action_type"]:
        return False
    match a["action_type"]:
        case ActionTypes.NONE:
            return True
        case ActionTypes.SCROLL:
            da = "up" if "up" in a["direction"] else "down"
            db = "up" if "up" in b["direction"] else "down"
            return da == db
        case ActionTypes.KEY_PRESS:
            return a["key_comb"] == b["key_comb"]
        case ActionTypes.MOUSE_CLICK | ActionTypes.MOUSE_HOVER:
            return np.allclose(a["coords"], b["coords"])
        case ActionTypes.KEYBOARD_TYPE:
            return a["text"] == b["text"]
        case ActionTypes.CLICK | ActionTypes.HOVER | ActionTypes.TYPE:
            if a["element_id"] and b["element_id"]:
                return a["element_id"] == b["element_id"]
            elif a["element_role"] and b["element_role"]:
                return a["element_role"] == b["element_role"] and a["element_name"] == b["element_name"]
            elif a["pw_code"] and b["pw_code"]:
                return a["pw_code"] == b["pw_code"]
            else:
                return False
        case ActionTypes.CLEAR:
            return np.allclose(a["coords"], b["coords"])
        case ActionTypes.PAGE_FOCUS:
            return a["page_number"] == b["page_number"]
        case ActionTypes.NEW_TAB:
            return True
        case ActionTypes.GO_BACK:
            return True
        case ActionTypes.GO_FORWARD:
            return True
        case ActionTypes.GOTO_URL:
            return a["url"] == b["url"]
        case ActionTypes.PAGE_CLOSE:
            return True
        case ActionTypes.CHECK | ActionTypes.SELECT_OPTION:
            return a["pw_code"] == b["pw_code"]
        case ActionTypes.STOP:
            return a["answer"] == b["answer"]
        case _:
            raise ValueError(f"Unknown action type: {a['action_type']}")


def action2str(action: Action, action_set_tag: str, semantic_element: str = "") -> str:
    """Return the string representation of an action

    sementic_element: the semantic information of the element
    such as a line in an accessibility tree
    """
    # NOTE[mandrade]: include action output if any to text representation + fix / add text representations.
    if action_set_tag in [
        "id_accessibility_tree",
        "id_accessibility_tree_with_captioner",
    ]:
        element_id = action["element_id"]
        match action["action_type"]:
            case ActionTypes.CLICK:
                # [ID=X] xxxxx
                action_str = f"click [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.TYPE:
                text = "".join([_id2key[i] for i in action["text"]])
                action_str = f"type [{element_id}] [{text}] where [{element_id}] is {semantic_element}"
            case ActionTypes.HOVER:
                action_str = f"hover [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.SCROLL:
                action_str = f"scroll [{action['direction']}]"
            case ActionTypes.KEY_PRESS:
                action_str = f"press [{action['key_comb']}]" + (f" [{action['text']}]" if action["text"] else "")
                if action.get("action_output"):
                    action_str = action_str + f". Output: {action['action_output']}"
            case ActionTypes.GOTO_URL:
                action_str = f"goto [{action['url']}]"
            case ActionTypes.NEW_TAB:
                action_str = "new_tab"
            case ActionTypes.PAGE_CLOSE:
                action_str = "close_tab"
            case ActionTypes.GO_BACK:
                action_str = "go_back"
            case ActionTypes.GO_FORWARD:
                action_str = "go_forward"
            case ActionTypes.PAGE_FOCUS:
                action_str = f"page_focus [{action['page_number']}]"
            case ActionTypes.CLEAR:
                action_str = f"clear [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.UPLOAD:
                action_str = f"upload [{action['text']}] to [{element_id}]"
            case ActionTypes.STOP:
                action_str = f"stop [{action['answer']}]"
            case ActionTypes.NONE:
                action_str = "none"
            case ActionTypes.SELECT_OPTION:
                action_str = f"select [{action['text']}] from [{element_id}]"
            case _:
                raise ValueError(f"Unknown action type {action['action_type']}")
    elif action_set_tag == "som":
        element_id = action["element_id"]
        match action["action_type"]:
            case ActionTypes.CLICK:
                action_str = f"click [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.CLEAR:
                action_str = f"clear [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.TYPE:
                text = "".join([_id2key[i] for i in action["text"]])
                action_str = f"type [{element_id}] [{text}] where [{element_id}] is {semantic_element}"
            case ActionTypes.HOVER:
                action_str = f"hover [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.SCROLL:
                action_str = f"scroll [{action['direction']}]"
            case ActionTypes.KEY_PRESS:
                action_str = f"press [{action['key_comb']}]" + (f" [{action['text']}]" if action["text"] else "")
                if action.get("action_output"):  # NOTE[mandrade]: added
                    action_str = action_str + f". Output: {action['action_output']}"
            case ActionTypes.GOTO_URL:
                action_str = f"goto [{action['url']}]"
            case ActionTypes.NEW_TAB:
                action_str = "new_tab"
            case ActionTypes.PAGE_CLOSE:
                action_str = "close_tab"
            case ActionTypes.GO_BACK:
                action_str = "go_back"
            case ActionTypes.GO_FORWARD:
                action_str = "go_forward"
            case ActionTypes.PAGE_FOCUS:
                action_str = f"page_focus [{action['page_number']}]"
            case ActionTypes.CLEAR:
                action_str = f"clear [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.STOP:
                action_str = f"stop [{action['answer']}]"
            case ActionTypes.UPLOAD:  # NOTE[mandrade]: added
                action_str = f"upload [{action['text']}] to [{element_id}]"
            case ActionTypes.SELECT_OPTION:  # NOTE[mandrade]: added
                text = "".join([_id2key[i] for i in action["text"]])
                action_str = f"select [{text}] from [{element_id}] where [{element_id}] is {semantic_element}"
            case ActionTypes.NONE:
                action_str = "none"
            case _:
                raise ValueError(f"Unknown action type {action['action_type']}")
    else:
        raise NotImplementedError(f"Unknown action set tag {action_set_tag}")

    return action_str


def action2create_function(action: Action) -> str:
    match action["action_type"]:
        case ActionTypes.NONE:
            return "create_none_action()"
        # mouse wheel and keyboard action
        case ActionTypes.SCROLL:
            direction = "up" if "up" in action["direction"] else "down"
            return f"create_scroll_action({repr(direction)})"
        case ActionTypes.KEY_PRESS:
            return f"create_key_press_action({repr(action['key_comb'])})"
        # inter-page actions
        case ActionTypes.PAGE_FOCUS:
            return f"create_page_focus_action({action['page_number']})"
        case ActionTypes.NEW_TAB:
            return "create_new_tab_action()"
        case ActionTypes.GO_BACK:
            return "create_go_back_action()"
        case ActionTypes.GO_FORWARD:
            return "create_go_forward_action()"
        case ActionTypes.GOTO_URL:
            return f"create_goto_url_action({repr(action['url'])})"
        case ActionTypes.PAGE_CLOSE:
            return "create_page_close_action()"

        # low-level keyboard and mouse actions
        case ActionTypes.MOUSE_CLICK:
            return f"create_mouse_click_action({action['coords'][0]}, {action['coords'][1]})"
        case ActionTypes.MOUSE_HOVER:
            return f"create_mouse_hover_action({action['coords'][0]}, {action['coords'][1]})"
        case ActionTypes.KEYBOARD_TYPE:
            return f"create_keyboard_type_action({list(map(lambda x: _id2key[x], action['text']))})"

        # mid-level keyboard and mouse actions
        case ActionTypes.CLICK:
            args = []
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(f"element_role={repr(_id2role[action['element_role']])}")
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_click_action({args_str})"
        case ActionTypes.CLEAR:
            args = []
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(f"element_role={repr(_id2role[action['element_role']])}")
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_clear_action({args_str})"
        case ActionTypes.UPLOAD:
            args = []
            text = "".join(map(lambda x: _id2key[x], action["text"]))
            args.append(f"text={repr(text)}")
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(f"element_role={repr(_id2role[action['element_role']])}")
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_upload_action({args_str})"
        case ActionTypes.HOVER:
            args = []
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(f"element_role={repr(_id2role[action['element_role']])}")
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_hover_action({args_str})"
        case ActionTypes.TYPE:
            args = []
            text = "".join(map(lambda x: _id2key[x], action["text"]))
            args.append(f"text={repr(text)}")
            args.append(f"element_id={repr(action['element_id'])}")
            args.append(f"element_role={repr(_id2role[action['element_role']])}")
            args.append(f"element_name={repr(action['element_name'])}")
            args.append(f"pw_code={repr(action['pw_code'])}")
            args_str = ", ".join(args)
            return f"create_type_action({args_str})"

        # high-level actions, only support locators from playwright
        case ActionTypes.CHECK:
            return f"create_check_action(pw_code={repr(action['pw_code'])})"
        case ActionTypes.SELECT_OPTION:
            return f"create_select_option_action(pw_code={repr(action['pw_code'])})"
        case ActionTypes.STOP:
            return f"create_stop_action({repr(action['answer'])})"

    raise ValueError(f"Invalid action type: {action['action_type']}")


_key2id: dict[str, int] = {key: i for i, key in enumerate(chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"]))}
_id2key: list[str] = sorted(_key2id, key=_key2id.get)  # type: ignore[arg-type]
_role2id: dict[RolesType, int] = {cast(RolesType, role): i for i, role in enumerate(chain(ROLES, SPECIAL_LOCATORS))}
_id2role: list[RolesType] = sorted(_role2id, key=_role2id.get)  # type: ignore[arg-type]


def _keys2ids(keys: list[int | str] | str) -> list[int]:
    # fmt:off
    return list(
        map(
            lambda key: _key2id.get(str(key), _key2id.get(key, " ")) 
            if isinstance(key, str) 
            else int(key),
            keys,
        )
    )  # type: ignore
    # fmt:on


def get_action_space() -> spaces.Dict:
    """Return the space of serialized actions."""
    space = spaces.Dict(
        {
            "action_type": spaces.Discrete(len(ActionTypes)),
            # coords (left, top) is used for COORD_CLICK
            "coords": spaces.Box(
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ),
            # element role is used for FOCUS_AND_CLICK and FOCUS_AND_TYPE
            "element_role": spaces.Discrete(len(ROLES) + len(SPECIAL_LOCATORS)),
            # element name is used with element role
            "element_name": spaces.Text(TEXT_MAX_LENGTH),
            "element_id": spaces.Text(TEXT_MAX_LENGTH),
            # text is only used for TYPE and FOCUS_AND_TYPE
            "text": spaces.MultiDiscrete([len(ASCII_CHARSET) + len(SPECIAL_KEYS) + len(FREQ_UNICODE_CHARSET)] * TYPING_MAX_LENGTH),
            "page_number": spaces.Discrete(MAX_PAGE_NUMBER),
            "url": spaces.Text(URL_MAX_LENGTH),
            "nth": spaces.Discrete(MAX_ELEMENT_INDEX_IN_VIEWPORT),
            "key_comb": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "direction": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "pw_code": spaces.Text(MAX_VANILLA_STR_LENGTH),
            "answer": spaces.Text(MAX_ANSWER_LENGTH),
        }
    )
    return space


def create_random_action() -> Action:
    """Return a random action."""
    return {
        "action_type": np.random.randint(len(ActionTypes)),
        "coords": np.random.rand(2).astype(np.float32),
        "element_role": np.random.randint(len(ROLES) + len(SPECIAL_LOCATORS)),
        "element_name": "".join(random.choices(ASCII_CHARSET, k=np.random.randint(TEXT_MAX_LENGTH))),
        "text": list(
            random.choices(
                list(range(len(ASCII_CHARSET))),
                k=np.random.randint(TYPING_MAX_LENGTH),
            )
        ),
        "page_number": np.random.randint(MAX_PAGE_NUMBER),
        "url": "".join(random.choices(ASCII_CHARSET, k=np.random.randint(URL_MAX_LENGTH))),
        "nth": np.random.randint(MAX_ELEMENT_INDEX_IN_VIEWPORT),
        "element_id": str(np.random.randint(MAX_ELEMENT_ID)),
        "key_comb": "+".join(random.choices(SPECIAL_KEYS, k=np.random.randint(3))),
        "direction": random.choice(["up", "down"]),
        "pw_code": "".join(
            random.choices(
                string.ascii_uppercase + string.digits,
                k=np.random.randint(MAX_VANILLA_STR_LENGTH),
            )
        ),
        "answer": str(np.random.randint(MAX_ANSWER_LENGTH)),
        "raw_prediction": str(np.random.randint(MAX_ANSWER_LENGTH)),
        "action_output": None,
        "element_center": None,
        "wait_for": None,
    }


def create_none_action() -> Action:
    """Return a valid action object that does nothing."""
    return {
        "action_type": ActionTypes.NONE,
        "coords": np.zeros(2, dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "pw_code": "",  # str that requires further processing
        "element_id": "",
        "key_comb": "",
        "direction": "",
        "answer": "",
        "raw_prediction": "",
        "action_output": None,
        "element_center": None,
        "wait_for": None,
    }


def create_refine_action(ids_keep: list[str] | None = None, ids_remove: list[str] | None = None) -> Action:
    # TODO[mandrade]: re-integrate refine action
    action = create_none_action()
    action.update({"action_type": ActionTypes.REFINE_INPUT, "ids_keep": ids_keep, "ids_remove": ids_remove, "wait_for": 0})  # type: ignore
    return action


def create_stop_action(answer: str) -> Action:
    action = create_none_action()
    # NOTE[mandrade]: normalize URLs in the answer before creating the stop action
    answer = normalize_urls_in_string(answer.strip())
    action.update({"action_type": ActionTypes.STOP, "answer": answer, "wait_for": 0.1})
    return action


def create_scroll_action(direction: str) -> Action:
    """Return the playwright action"""
    assert direction in ["up", "down"]
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.SCROLL,
            "direction": direction,
        }
    )
    return action


def create_mouse_hover_action(left: float | None = None, top: float | None = None) -> Action:
    """Return a valid action object with type COORD_CLICK."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.MOUSE_HOVER,
            "coords": np.array([left, top], dtype=np.float32),
        }
    )
    return action


def create_key_press_action(key_comb: str, text: str | None = None) -> Action:
    """Return the key press action"""
    # NOTE[mandrade]: more robust version of key_press action

    def map_keys(key_comb: str) -> str:
        keys = key_comb.split("+")
        mapped_keys = []
        for key in keys:
            mapped_key = SPECIAL_KEY_MAPPINGS.get(key.lower(), key)
            mapped_keys.append(mapped_key)
        return "+".join(mapped_keys)

    action = create_none_action()
    mapped_key_comb = map_keys(key_comb)

    # NOTE[mandrade]: allow key press with text input (e.g.: ctrl+f 'find text')
    if text:
        action.update({"action_type": ActionTypes.KEY_PRESS, "key_comb": mapped_key_comb, "text": _keys2ids(text)})  # type: ignore
    else:
        action.update({"action_type": ActionTypes.KEY_PRESS, "key_comb": mapped_key_comb})
    return action


def create_page_focus_action(page_number: int) -> Action:
    """Return a valid action object with type PAGE_FOCUS."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.PAGE_FOCUS,
            "page_number": page_number,
        }
    )
    return action


def create_new_tab_action() -> Action:
    """Return a valid action object with type NEW_TAB."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.NEW_TAB,
        }
    )
    return action


def create_go_back_action() -> Action:
    """Return a valid action object with type GO_BACK."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GO_BACK,
        }
    )
    return action


def create_go_forward_action() -> Action:
    """Return a valid action object with type GO_FORWARD."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GO_FORWARD,
        }
    )
    return action


def create_goto_url_action(url: str) -> Action:
    """Return a valid action object with type GOTO_URL."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.GOTO_URL,
            "url": url,
        }
    )
    return action


def create_page_close_action() -> Action:
    """Return a valid action object with type PAGE_CLOSE."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.PAGE_CLOSE,
        }
    )
    return action


def create_mouse_click_action(left: float | None = None, top: float | None = None) -> Action:
    """Return a valid action object with type COORD_CLICK."""
    action = create_none_action()
    if left and top:
        action.update(
            {
                "action_type": ActionTypes.MOUSE_CLICK,
                "coords": np.array([left, top], dtype=np.float32),
            }
        )
    elif (not left) and (not top):
        action.update(
            {
                "action_type": ActionTypes.CLICK,
            }
        )
    else:
        raise ValueError("left and top must be both None or both not None")
    return action


def create_clear_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLEAR,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


def create_upload_action(
    text: str,
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.UPLOAD,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "text": _keys2ids(text),
            "pw_code": pw_code,
        }
    )  # type: ignore
    return action


def create_keyboard_type_action(keys: list[int | str] | str) -> Action:
    """Return a valid action object with type TYPE."""
    action = create_none_action()
    action.update({"action_type": ActionTypes.KEYBOARD_TYPE, "text": _keys2ids(keys)})  # type: ignore
    return action


def create_click_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLICK,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


def create_hover_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.HOVER,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "pw_code": pw_code,
        }
    )
    return action


def create_type_action(
    text: str,
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.TYPE,
            "element_id": element_id,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
            "text": _keys2ids(text),
            "pw_code": pw_code,
        }
    )
    return action


def create_select_id_action(element_id: str, option: str, nth: int = 0) -> Action:
    # NOTE[mandrade]: enable select action by id + option text
    a = create_none_action()
    a.update(
        {
            "action_type": ActionTypes.SELECT_OPTION,
            "element_id": element_id,
            "nth": nth,
            "text": _keys2ids(option),
            "pw_code": "",
        }
    )
    return a


def create_check_action(pw_code: str) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CHECK,
            "pw_code": pw_code,
        }
    )
    return action


def create_select_option_action(
    pw_code: str,
) -> Action:
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.SELECT_OPTION,
            "pw_code": pw_code,
        }
    )
    return action


def create_focus_action(element_role: RolesType, element_name: str = "", nth: int = 0) -> Action:
    """Return a valid action object with type CLICK.

    Keep compatible with the old version."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLICK,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
        }
    )
    return action


def create_focus_and_click_action(element_role: RolesType, element_name: str = "", nth: int = 0) -> Action:
    """Return a valid action object with type CLICK.

    Keep compatible with the old version."""

    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.CLICK,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "nth": nth,
        }
    )
    return action


def create_focus_and_type_action(
    keys: list[int | str] | str,
    element_role: RolesType,
    element_name: str = "",
    nth: int = 0,
) -> Action:
    """Return a valid action object with type TYPE.

    Keep compatible with the old version."""
    action = create_none_action()
    action.update(
        {
            "action_type": ActionTypes.TYPE,
            "element_role": _role2id[element_role],
            "element_name": element_name,
            "text": _keys2ids(keys),
            "nth": nth,
        }
    )
    return action


def create_playwright_action(playwright_code: str) -> Action:
    """Main function to return individual playwright action"""
    # get the last action
    regex = r"\.(?![^\(\)]*\))"
    action = re.split(regex, playwright_code)[-1].split("(")[0].lower()
    match action:
        case "press":
            p = r'press\((?:"|\')(.+?)(?:"|\')\)'
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError(f"Invalid press action, required to be page.press(KEY_COMB_STR)")
            key_comb = match.group(1)
            return create_key_press_action(key_comb=key_comb)
        case "scroll":
            direction = "up" if "up" in playwright_code else "down"
            return create_scroll_action(direction=direction)
        case "click":
            return create_click_action(pw_code=playwright_code)
        case "clear":
            return create_clear_action(pw_code=playwright_code)
        case "upload":
            return create_upload_action(text="", pw_code=playwright_code)
        case "hover":
            return create_hover_action(pw_code=playwright_code)
        case "type" | "fill":
            p = r'type|fill\((?:"|\')(.+?)(?:"|\')\)'
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError(f"Invalid type/fill action, required to be page.type(TEXT)")
            text = match.group(1)
            return create_type_action(text=text, pw_code=playwright_code)
        case "select_option":
            return create_select_option_action(pw_code=playwright_code)
        case "check":
            return create_check_action(pw_code=playwright_code)
        case "goto":
            p = r'goto\((?:"|\')(.+?)(?:"|\')\)'
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError(f"Invalid goto action, required to be page.goto(URL_STR)")
            url = match.group(1)
            return create_goto_url_action(url)
        case "page_focus":
            # get the page number
            p = r"page_focus\((\d+)\)"
            match = re.search(p, playwright_code)
            if not match:
                raise ActionParsingError("page focus requires a page number")
            page_num = int(match.group(1))
            return create_page_focus_action(page_num)
        case "new_tab":
            return create_new_tab_action()
        case "go_back":
            return create_go_back_action()
        case "go_forward":
            return create_go_forward_action()
        case "page_close":
            return create_page_close_action()
        case "stop":  # page.stop(answer)
            p = r'stop\(?"(.+)?"\)'
            match = re.search(p, playwright_code)
            if not match:
                answer = ""
            else:
                answer = match.group(1)
            return create_stop_action(answer)

    raise ActionParsingError(f"Unknown playwright action {action}")


def create_id_based_action(action_str: str) -> Action:
    """Main function to return individual id based action"""
    original_action_str = action_str
    action_str = action_str.strip().lower()
    if "[" in action_str:
        action = action_str.split("[")[0].strip()
    else:
        actions = action_str.split()
        if actions:
            action = actions[0].strip()
        else:
            raise ActionParsingError(f"No action specified: {action_str}")
    match action:
        case "click":
            match = re.search(r"click ?\[(\d+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid click action {action_str}")
            element_id = match.group(1)
            return create_click_action(element_id=element_id)
        case "clear":
            match = re.search(r"clear ?\[(\d+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid clear action {action_str}")
            element_id = match.group(1)
            return create_clear_action(element_id=element_id)
        case "upload":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(r"type|upload ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid type action {action_str}")
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                text += "\n"
            return create_upload_action(text=text, element_id=element_id)
        case "hover":
            match = re.search(r"hover ?\[(\d+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid hover action {action_str}")
            element_id = match.group(1)
            return create_hover_action(element_id=element_id)
        case "type":
            # add default enter flag
            if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                action_str += " [1]"

            match = re.search(r"type ?\[(\d+)\] ?\[(.*)\] ?\[(\d+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid type action {action_str}")
            element_id, text, enter_flag = (
                match.group(1),
                match.group(2),
                match.group(3),
            )
            if enter_flag == "1":
                text += "\n"
            return create_type_action(text=text, element_id=element_id)

        case "select":
            match = re.search(r"select ?\[(\d+)\] ?\[(.+)\]", original_action_str, re.IGNORECASE)
            if not match:
                raise ActionParsingError(f"Invalid select action {original_action_str}")
            element_id, option = match.group(1), match.group(2)
            return create_select_id_action(element_id=element_id, option=option)

        case "press":
            # NOTE[mandrade]: Add possibility to emit `press [key_comb=KEY_COMB] [text=TEXT]` for actions like "ctrl+f"
            match = re.search(r"press ?\[(.+?)\](?: ?\[(.+?)\])?", action_str)
            if not match:
                raise ActionParsingError(f"Invalid press action {action_str}")
            key_comb = match.group(1)

            # NOTE[mandrade]: prune spaces; Playwright does not deal well with them
            key_comb = re.sub(r"[ \t\r]+", "", key_comb).strip()
            text = match.group(2).strip() if match.group(2) else None

            return create_key_press_action(key_comb=key_comb, text=text)
        case "scroll":
            # NOTE[mandrade]: Add possibility to emit `scroll [direction=up]` or `scroll [direction=down]`
            match = re.search(r"scroll \[(?:direction=)?(up|down)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid scroll action {action_str}")
            direction = match.group(1)
            return create_scroll_action(direction=direction)
        case "goto":
            match = re.search(r"goto ?\[(.+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid goto action {action_str}")
            url = match.group(1)
            return create_goto_url_action(url=url)
        case "new_tab":
            return create_new_tab_action()
        case "go_back":
            return create_go_back_action()
        case "go_forward":
            return create_go_forward_action()
        case "tab_focus":
            match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
            if not match:
                raise ActionParsingError(f"Invalid tab_focus action {action_str}")
            page_number = int(match.group(1))
            return create_page_focus_action(page_number)
        case "close_tab":
            return create_page_close_action()

        case "stop":  # stop answer
            # REVIEW[mandrade] original did not allow for a invalid stop action and emit undesired VALID stop actions with NULL response.
            # E.g.:  ```stop "Sprite Stasis Ball 65 cm"``` would be parsed as ```stop```

            answer = ""

            # If stop action has an answer and is valid:
            match = re.search(r"stop ?\[(.*?)\]", action_str, re.DOTALL)
            if match:
                answer = match.group(1)
                return create_stop_action(answer)

            # If stop with null answer:
            match2 = re.search(r"stop ?(.+)", action_str)
            if not match2:
                # return null response for generations like: 'stop '
                return create_stop_action(answer)

            # If not null, check for possibly invalid format:
            substring = match2.group(1)  # Get answer
            substring = re.sub(r"\s+", "", substring)  # Remove all space-like characters from answer
            # e.g.: stop [  ] -> substring = [ ] -> substring = []

            # Raise error for stop actions like: stop "Sprite Stasis Ball 65 cm"
            if len(substring) > 0 and substring != "[]":
                raise ActionParsingError(f"Invalid stop action {action_str}")
            else:
                # return null response for generations like: `stop []` or `stop [ ]`` or `stop    `
                return create_stop_action(answer)

    raise ActionParsingError(f"Invalid action {action_str}")


# ==============================================================================
# action execution functions
# ==============================================================================


def safe_select_option(page, element_center: tuple[float, float], option: str, max_retries: int = 3, timeout: float = 2 * 1000) -> None:
    """Safely select an option with retry logic for DOM attachment issues."""
    # NOTE[mandrade]: Safer way of selecting options in select elements

    parsed_option = _parse_select_arg(option)
    for attempt in range(max_retries):
        select_element = get_select_element_at(element_center[0], element_center[1], page)
        if not select_element:
            raise ValueError(f"No element found at {element_center}")

        try:
            # Check if element is still attached to DOM by trying to get its tag
            try:
                tag = select_element.evaluate("e => e?.tagName?.toLowerCase?.() ?? null")
                if not tag:
                    raise ValueError("Element is not attached to DOM")
            except Exception:
                raise ValueError("Element is not attached to DOM")

            # Attempt the selection
            select_element.select_option(**parsed_option, timeout=timeout)
            return  # Success

        except Exception as e:
            if "not attached to the DOM" in str(e) or "Element is not attached to DOM" in str(e):
                if attempt < max_retries - 1:
                    # Wait a bit and retry
                    time.sleep(0.1)
                    continue
                else:
                    # Final attempt failed
                    raise ValueError(f"Element became detached from DOM after {max_retries} attempts: {e}")
            else:
                # Different error, don't retry
                raise e


def execute_scroll(direction: str, page: Page) -> None:
    # perform the action: find nearby scrollable siblings, fallback to document
    page.evaluate(_SCROLL_JS_CODE, direction)


async def aexecute_scroll(direction: str, page: APage) -> None:
    # perform the action: find nearby scrollable siblings, fallback to document
    await page.evaluate(_SCROLL_JS_CODE, direction)


def execute_key_press(key: str, page: Page, text: list[int] | None = None) -> Any | None:
    """Press a key."""
    action_output = None
    # Remove all spaces between keys; playwright does not deal well with spaces
    key = re.sub(r"[ \t\r]+", "", key).strip()

    # More robust parsing of ctrl/command combinations
    if "meta" in key.lower() or "command" in key.lower():
        if "Mac" not in page.evaluate("navigator.platform"):
            # For non-Mac platforms, substitute "meta" or "command" with "Control".
            key = re.sub(r"(meta|command)", "Control", key, flags=re.IGNORECASE)
        else:
            # On Mac, substitute with "Meta" to reflect the Command key.
            key = re.sub(r"(meta|command)", "Meta", key, flags=re.IGNORECASE)
    elif "ctrl" in key.lower() or "control" in key.lower():
        if "Mac" in page.evaluate("navigator.platform"):
            # On Mac, use "Meta" instead of Control.
            key = re.sub(r"(ctrl|control)", "Meta", key, flags=re.IGNORECASE)
        else:
            key = re.sub(r"(ctrl|control)", "Control", key, flags=re.IGNORECASE)

    # If `find` command:
    if re.search(r"(Control|Meta)\s*\+\s*f", key, flags=re.IGNORECASE):
        _text = "".join([_id2key[key] for key in text]) if text else ""
        # Trigger the find command
        count = find_and_highlight_in_page(page, _text)
        action_output = f"Found {count} occurrences of '{_text}'"
        _ = page.evaluate(f"window.find('{_text}')")
    else:
        page.keyboard.press(key)

    return action_output


async def aexecute_key_press(key: str, page: APage) -> None:
    """Press a key."""
    if "Meta" in key and "Mac" not in await page.evaluate("navigator.platform"):
        key = key.replace("Meta", "Control")
    await page.keyboard.press(key)


def execute_mouse_hover(left: float, top: float, page: Page) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    page.mouse.move(left * viewport_size["width"], top * viewport_size["height"])


async def aexecute_mouse_hover(left: float, top: float, page: APage) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    await page.mouse.move(left * viewport_size["width"], top * viewport_size["height"])


def execute_mouse_click(left: float, top: float, page: Page, try_offset: bool = False) -> tuple[float, float]:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    # NOTE[mandrade]: For SOM type, env clicks at the CENTER of bounding box.
    # However, some cases the element is not exactly in the center.
    # Below tries to click slightly off-center until the page content changes.
    # NOTE: ad-hoc offsets; can be improved considering bounding boxes.
    offsets = [
        (1.00, 1.00),  # same
        (1.00, 0.99),  # Up
        (1.00, 1.01),  # Down
        (0.99, 1.00),  # Left
        (1.01, 1.00),  # Right
        (0.99, 0.99),  # Up-left
        (1.01, 0.99),  # Up-right
        (0.99, 1.01),  # Down-left
        (1.01, 1.01),  # Down-right
    ]
    if not try_offset:
        page.mouse.click(left * viewport_size["width"], top * viewport_size["height"])
        return left, top

    # Else, try to click with offsets
    try:
        pg_content_0 = safe_get_page_content(page)
        state_0 = safe_get_page_state(page)

        for offset in offsets:
            page.mouse.click(left * viewport_size["width"] * offset[0], top * viewport_size["height"] * offset[1])
            if _page_content_changed(page, pg_content_0, state_0):
                return left * offset[0], top * offset[1]
    except Exception as _:
        # Fallback to normal behavior if error occurs
        page.mouse.click(left * viewport_size["width"], top * viewport_size["height"])
    return left, top


async def aexecute_mouse_click(left: float, top: float, page: APage) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    await page.mouse.click(left * viewport_size["width"], top * viewport_size["height"])


def execute_upload(left: float, top: float, path: str, page: Page, max_wait: float = 2_000) -> None:
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    resolved_path = any_to_path(path)
    with page.expect_file_chooser(timeout=max_wait) as fc_info:
        page.mouse.click(left * viewport_size["width"], top * viewport_size["height"])
    file_chooser = fc_info.value
    file_chooser.set_files(resolved_path)


async def aexecute_upload(left: float, top: float, path: str, page: APage, max_wait: float = 2_000) -> None:
    # NOTE[mandrade]: not tested
    """Click at coordinates (left, top)."""
    viewport_size = page.viewport_size
    assert viewport_size
    async with page.expect_file_chooser(timeout=max_wait) as fc_info:
        await page.mouse.click(left * viewport_size["width"], top * viewport_size["height"])
    file_chooser = fc_info.value
    await file_chooser.set_files(path)


def execute_keyboard_type(text: str, page: Page) -> None:
    """Fill the focused element with text."""
    page.keyboard.type(text)


async def aexecute_keyboard_type(text: str, page: APage) -> None:
    """Fill the focused element with text."""
    await page.keyboard.type(text)


def execute_click_current(page: Page) -> None:
    """Click at the current mouse position."""
    locators = page.locator("*:focus")
    if not locators.count():
        for frame in page.frames[1:]:
            locators = frame.locator("*:focus")
            if locators.count():
                break
    locators.click()


async def aexecute_click_current(page: APage) -> None:
    """Click at the current mouse position."""
    locators = page.locator("*:focus")
    locator_count = await locators.count()
    if not locator_count:
        for frame in page.frames[1:]:
            locators = frame.locator("*:focus")
            locator_count = await locators.count()
            if locator_count:
                break
    await locators.click()
    await page.wait_for_load_state("load")


def execute_clear(page: Page) -> None:
    execute_key_press("Meta+A", page)
    execute_key_press("Backspace", page)


def execute_type(keys: list[int], page: Page) -> None:
    """Send keystrokes to the focused element."""
    text = "".join([_id2key[key] for key in keys])
    # NOTE[mandrade]: Clear the focused element before typing
    execute_clear(page)
    # type text into focused element
    page.keyboard.type(text)


async def aexecute_type(keys: list[int], page: APage) -> None:
    """Send keystrokes to the focused element."""
    text = "".join([_id2key[key] for key in keys])
    await page.keyboard.type(text)


def execute_focus(element_role: int, element_name: str, nth: int, page: Page) -> None:
    """Click the specified DOM element."""
    element_role_str = _id2role[element_role]
    if page.viewport_size is None:
        raise ValueError("Viewport size is not set for the current page")
    element_location_list: list[tuple[Locator, float, float]] = []
    for frame in page.frames:
        match element_role_str:
            case "alt_text":
                locators = frame.get_by_alt_text(element_name)
            case "label":
                locators = frame.get_by_label(element_name)
            case "placeholder":
                locators = frame.get_by_placeholder(element_name)
            case _:
                locators = frame.get_by_role(role=element_role_str, name=element_name)
        for locator_idx in range(locators.count()):
            locator = locators.nth(locator_idx)
            if is_in_viewport(locator, page.viewport_size):
                bounding_box = locator.bounding_box()
                assert bounding_box
                element_location_list.append((locator, bounding_box["x"], bounding_box["y"]))
    if len(element_location_list) <= nth:
        raise ValueError(f"There are only {len(element_location_list)} elements found in viewport, but {nth + 1} is requested")
    element_location_list.sort(key=lambda x: (x[2], x[1]))  # row major order
    element_location_list[nth][0].focus()


async def aexecute_focus(element_role: int, element_name: str, nth: int, page: APage) -> None:
    """Click the specified DOM element."""
    element_role_str = _id2role[element_role]
    if page.viewport_size is None:
        raise ValueError("Viewport size is not set for the current page")
    element_location_list: list[tuple[ALocator, float, float]] = []
    for frame in page.frames:
        match element_role_str:
            case "alt_text":
                locators = frame.get_by_alt_text(element_name)
            case "label":
                locators = frame.get_by_label(element_name)
            case "placeholder":
                locators = frame.get_by_placeholder(element_name)
            case _:
                locators = frame.get_by_role(role=element_role_str, name=element_name)
        for locator_idx in range(await locators.count()):
            locator = locators.nth(locator_idx)
            if await async_is_in_viewport(locator, page.viewport_size):
                bounding_box = await locator.bounding_box()
                assert bounding_box
                element_location_list.append((locator, bounding_box["x"], bounding_box["y"]))
    if len(element_location_list) <= nth:
        raise ValueError(f"There are only {len(element_location_list)} elements found in viewport, but {nth + 1} is requested")
    element_location_list.sort(key=lambda x: (x[2], x[1]))  # row major order
    await element_location_list[nth][0].focus()


def locate(locator_calls: list[ParsedPlaywrightCode], page: Page) -> Locator:
    locator = page
    for call in locator_calls:
        function_name = call["function_name"]
        arguments = call["arguments"]
        keywords = call["keywords"]
        locator = getattr(locator, function_name)(*arguments, **keywords)
    return locator  # type: ignore[return-value]


async def alocate(locator_calls: list[ParsedPlaywrightCode], page: APage) -> ALocator:
    locator = page
    for call in locator_calls:
        function_name = call["function_name"]
        arguments = call["arguments"]
        keywords = call["keywords"]
        locator = await getattr(locator, function_name)(*arguments, **keywords)
    return locator  # type: ignore[return-value]


def execute_playwright_click(
    locator_code: list[ParsedPlaywrightCode],
    page: Page,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)

    # perform the action
    locator.click(*pw_action_args, **pw_action_kwargs)


async def aexecute_playwright_click(
    locator_code: list[ParsedPlaywrightCode],
    page: APage,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = await alocate(locator_code, page)

    # perform the action
    await locator.click(*pw_action_args, **pw_action_kwargs)


def execute_playwright_hover(locator_code: list[ParsedPlaywrightCode], page: Page) -> None:
    locator = locate(locator_code, page)

    # perform the action
    locator.hover()


async def aexecute_playwright_hover(locator_code: list[ParsedPlaywrightCode], page: APage) -> None:
    locator = await alocate(locator_code, page)

    # perform the action
    await locator.hover()


def execute_playwright_type(
    text: str,
    locator_code: list[ParsedPlaywrightCode],
    page: Page,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    pw_action_args = [text] + pw_action_args  # text is the first argument
    locator.type(*pw_action_args, **pw_action_kwargs)


async def aexecute_playwright_type(
    text: str,
    locator_code: list[ParsedPlaywrightCode],
    page: APage,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = await alocate(locator_code, page)
    # perform the action
    pw_action_args = [text] + pw_action_args  # text is the first argument
    await locator.type(*pw_action_args, **pw_action_kwargs)


def execute_playwright_select_option(
    locator_code: list[ParsedPlaywrightCode],
    page: Page,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = locate(locator_code, page)
    # perform the action
    locator.select_option(*pw_action_args, **pw_action_kwargs)


async def aexecute_playwright_select_option(
    locator_code: list[ParsedPlaywrightCode],
    page: APage,
    pw_action_args: list[str] = [],
    pw_action_kwargs: dict[str, Any] = {},
) -> None:
    locator = await alocate(locator_code, page)
    # perform the action
    await locator.select_option(*pw_action_args, **pw_action_kwargs)


def execute_playwright_check(locator_code: list[ParsedPlaywrightCode], page: Page) -> None:
    locator = locate(locator_code, page)
    # perform the action
    locator.check()


async def aexecute_playwright_check(locator_code: list[ParsedPlaywrightCode], page: APage) -> None:
    locator = await alocate(locator_code, page)
    # perform the action
    await locator.check()


def execute_action(
    action: Action,
    page: Page,
    browser_ctx: BrowserContext,
    observation_processor: ObservationProcessor,
) -> Page:
    """Execute the action on the ChromeDriver."""
    action_type = action["action_type"]
    clear_highlights(page)

    match action_type:
        case ActionTypes.NONE:
            pass

        case ActionTypes.SCROLL:
            direction = "up" if "up" in action["direction"] else "down"
            execute_scroll(direction, page)

        case ActionTypes.KEY_PRESS:
            keys = action["key_comb"]
            text = action.get("text", "")
            action_output = execute_key_press(keys, page, text)
            action["action_output"] = action_output

        case ActionTypes.MOUSE_CLICK:
            execute_mouse_click(action["coords"][0], action["coords"][1], page)

        case ActionTypes.CLEAR:
            element_id = action["element_id"]
            element_center = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            execute_mouse_click(element_center[0], element_center[1], page)
            execute_key_press("Meta+A", page)
            execute_key_press("Backspace", page)

        case ActionTypes.MOUSE_HOVER:
            execute_mouse_hover(action["coords"][0], action["coords"][1], page)

        case ActionTypes.KEYBOARD_TYPE:
            execute_type(action["text"], page)

        case ActionTypes.CLICK:
            # check each kind of locator in order
            # TODO[shuyanzh]: order is temp now
            if action["element_id"]:
                element_id = action["element_id"]
                if action.get("element_center") is None:
                    action["element_center"] = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]

                left, top = execute_mouse_click(action["element_center"][0], action["element_center"][1], page, try_offset=True)  # type: ignore[arg-type]
                # NOTE[mandrade]: Update element center after possible offset click
                action["element_center"] = (left, top)

            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                execute_focus(element_role, element_name, nth, page)
                execute_click_current(page)

            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                execute_playwright_click(locator_code=locator_code, page=page)

            else:
                raise ValueError("No proper locator found for click action")
        case ActionTypes.HOVER:
            if action["element_id"]:
                element_id = action["element_id"]
                if action.get("element_center") is None:
                    action["element_center"] = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                element_center = action["element_center"]
                execute_mouse_hover(element_center[0], element_center[1], page)  # type: ignore[arg-type]
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                execute_focus(element_role, element_name, nth, page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                execute_playwright_hover(locator_code=locator_code, page=page)
            else:
                raise NotImplementedError("No proper locator found for hover action")
        case ActionTypes.TYPE:
            if action["element_id"]:
                element_id = action["element_id"]
                if action.get("element_center") is None:
                    action["element_center"] = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                element_center = action["element_center"]

                if not element_center:
                    raise ValueError("Element center could not be determined for type action")
                # NOTE[mandrade]: Fast check if it is an upload file action. If so, upload the file.
                # Obs.: some webpages use INPUT elements for file upload where the agent can type the file uri.
                file_input_handle = get_file_input_handle_at(element_center[0], element_center[1], page, behavior_timeout_ms=0)
                if file_input_handle:
                    path = "".join([_id2key[key] for key in action["text"]]).strip()
                    resolved_path = any_to_path(path)
                    try:
                        file_input_handle.set_input_files(resolved_path)
                    except Exception as _:
                        execute_upload(element_center[0], element_center[1], path, page)
                # Otherwise, it is a normal type action.
                else:
                    # NOTE[mandrade]: Focus the element before typing
                    focus_by_coords(element_center[0], element_center[1], page)
                    execute_type(action["text"], page)

            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                execute_focus(element_role, element_name, nth, page)
                execute_type(action["text"], page)

            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                text = parsed_code[-1]["arguments"][0]
                # [shuyanzh], don't support action args and kwargs now
                execute_playwright_type(text=text, locator_code=locator_code, page=page)
            else:
                raise NotImplementedError("No proper locator found for type action")

        case ActionTypes.PAGE_FOCUS:
            page = browser_ctx.pages[action["page_number"]]
            page.bring_to_front()
        case ActionTypes.NEW_TAB:
            page = browser_ctx.new_page()
            page.client = page.context.new_cdp_session(page)  # type: ignore[attr-defined]
        case ActionTypes.GO_BACK:
            page.go_back()
            # page.reload()
        case ActionTypes.GO_FORWARD:
            page.go_forward()
            # page.reload()
        case ActionTypes.GOTO_URL:
            page.goto(map_url_to_local(action["url"]))
        case ActionTypes.PAGE_CLOSE:
            page.close()
            if len(browser_ctx.pages) > 0:
                page = browser_ctx.pages[-1]
            else:
                page = browser_ctx.new_page()

        case ActionTypes.SELECT_OPTION:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                execute_playwright_select_option(locator_code, page)
            elif action["element_id"]:
                # NOTE[mandrade]: Enable select id action
                element_id = action["element_id"]
                element_center = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                option = "".join([_id2key[key] for key in action["text"]]).strip()
                safe_select_option(element_center=element_center, option=option, page=page)
            else:
                raise NotImplementedError("No proper locator found for select option action")

        case ActionTypes.CHECK:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                execute_playwright_check(locator_code, page)
            else:
                raise NotImplementedError("No proper locator found for select option action")

        case ActionTypes.UPLOAD:
            if action["element_id"]:
                element_id = action["element_id"]
                element_center = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
                path = "".join([_id2key[key] for key in action["text"]]).strip()
                execute_upload(element_center[0], element_center[1], path, page)
            else:
                raise NotImplementedError("Upload only supported for id based actions")

        case ActionTypes.REFINE_INPUT:
            # TODO[mandrade]: re-integrate code to re-compute SoM based on the ids to keep/remove.
            observation_processor.set_plot_ids(ids_keep=action.get("ids_keep", None), ids_skip=action.get("ids_remove", None))  # type: ignore[attr-defined]
            action.update({"wait_for": 0})

        case _:
            raise ValueError(f"Unknown action type: {action_type}")

    return page


async def aexecute_action(action: Action, page: APage, browser_ctx: ABrowserContext) -> APage:
    """Execute the async action on the ChromeDriver."""
    # NOTE[mandrade]: this function was not updated.
    action_type = action["action_type"]
    match action_type:
        case ActionTypes.NONE:
            pass
        case ActionTypes.SCROLL:
            direction = "up" if "up" in action["direction"] else "down"
            await aexecute_scroll(direction, page)
        case ActionTypes.KEY_PRESS:
            keys = action["key_comb"]
            await aexecute_key_press(keys, page)

        case ActionTypes.MOUSE_CLICK:
            await aexecute_mouse_click(action["coords"][0], action["coords"][1], page)
        case ActionTypes.CLEAR:
            element_id = action["element_id"]
            element_center = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            await execute_mouse_click(element_center[0], element_center[1], page)
            await execute_key_press("Meta+A", page)
            await execute_key_press("Backspace", page)
        case ActionTypes.MOUSE_HOVER:
            await aexecute_mouse_hover(action["coords"][0], action["coords"][1], page)
        case ActionTypes.KEYBOARD_TYPE:
            await aexecute_type(action["text"], page)

        case ActionTypes.CLICK:
            # check each kind of locator in order
            # TODO[shuyanzh]: order is temp now
            if action["element_id"]:
                raise NotImplementedError
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                await aexecute_focus(element_role, element_name, nth, page)
                await aexecute_click_current(page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                await aexecute_playwright_click(locator_code=locator_code, page=page)
            else:
                raise ValueError("No proper locator found for click action")
        case ActionTypes.HOVER:
            if action["element_id"]:
                raise NotImplementedError
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                await aexecute_focus(element_role, element_name, nth, page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                # [shuyanzh], don't support action args and kwargs now
                await aexecute_playwright_hover(locator_code=locator_code, page=page)
            else:
                raise NotImplementedError("No proper locator found for hover action")
        case ActionTypes.TYPE:
            if action["element_id"]:
                raise NotImplementedError
            elif action["element_role"] and action["element_name"]:
                element_role = int(action["element_role"])
                element_name = action["element_name"]
                nth = action["nth"]
                await aexecute_focus(element_role, element_name, nth, page)
                await aexecute_type(action["text"], page)
            elif action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                text = parsed_code[-1]["arguments"][0]
                # [shuyanzh], don't support action args and kwargs now
                await aexecute_playwright_type(text=text, locator_code=locator_code, page=page)
            else:
                raise NotImplementedError("No proper locator found for type action")

        case ActionTypes.PAGE_FOCUS:
            page = browser_ctx.pages[action["page_number"]]
            await page.bring_to_front()
        case ActionTypes.NEW_TAB:
            page = await browser_ctx.new_page()
        case ActionTypes.GO_BACK:
            await page.go_back()
        case ActionTypes.GO_FORWARD:
            await page.go_forward()
        case ActionTypes.GOTO_URL:
            await page.goto(action["url"])
        case ActionTypes.PAGE_CLOSE:
            await page.close()
            if len(browser_ctx.pages) > 0:
                page = browser_ctx.pages[-1]
            else:
                page = await browser_ctx.new_page()

        case ActionTypes.SELECT_OPTION:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                await aexecute_playwright_select_option(locator_code, page)
            else:
                raise NotImplementedError("No proper locator found for select option action")
        case ActionTypes.CHECK:
            if action["pw_code"]:
                parsed_code = parse_playwright_code(action["pw_code"])
                locator_code = parsed_code[:-1]
                await aexecute_playwright_check(locator_code, page)
            else:
                raise NotImplementedError("No proper locator found for select option action")
        case ActionTypes.UPLOAD:
            element_id = action["element_id"]
            element_center = observation_processor.get_element_center(element_id)  # type: ignore[attr-defined]
            await aexecute_upload(element_center[0], element_center[1], action["text"], page)
        case _:
            raise ValueError(f"Unknown action type: {action_type}")

    return page


def parse_playwright_code(code: str) -> list[ParsedPlaywrightCode]:
    # extract function calls
    if not code.startswith("page."):
        raise ValueError(f'Playwright action must start with "page.", but got {code}')

    regex = r"\.(?![^\(\)]*\))"
    chain = re.split(regex, code)[1:]

    parsed_chain = []

    for item in chain:
        tree = ast.parse(item)
        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function_name = node.func.id  # type: ignore[attr-defined]
                arguments = [ast.literal_eval(arg) if isinstance(arg, ast.Str) else arg for arg in node.args]
                keywords = {str(kw.arg): ast.literal_eval(kw.value) for kw in node.keywords}
                funcs.append(
                    ParsedPlaywrightCode(
                        {
                            "function_name": function_name,
                            "arguments": arguments,  # type: ignore
                            "keywords": keywords,
                        }
                    )
                )

        if len(funcs) != 1:
            raise ValueError(f"Fail to parse {item} in {code}")

        if funcs[0]["function_name"] not in PLAYWRIGHT_LOCATORS + PLAYWRIGHT_ACTIONS:
            raise ValueError(
                f"Invalid playwright code {item}, ",
                f"the function needs to be one of {PLAYWRIGHT_LOCATORS + PLAYWRIGHT_ACTIONS}",
            )

        parsed_chain.append(funcs[0])

    last_action = parsed_chain[-1]
    if last_action["function_name"] not in PLAYWRIGHT_ACTIONS:
        raise ValueError(
            f"Invalid playwright action {last_action},",
            f"the action needs to be one of {PLAYWRIGHT_ACTIONS}",
        )

    return parsed_chain
