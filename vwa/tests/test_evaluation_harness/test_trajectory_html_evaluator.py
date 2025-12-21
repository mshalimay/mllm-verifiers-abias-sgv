import json
from pathlib import Path
from typing import cast

import pytest
from browser_env.actions import Action
from evaluation_harness.evaluators import TrajectoryURLExactEvaluator
from evaluation_harness.helper_functions import PseudoPage
from playwright.sync_api import Page

browser_env.env_utils import StateInfo


class PageLike:
    def __init__(self, url: str):
        self.url = url


def make_trajectory(urls: list[str]) -> list[StateInfo | Action]:
    trajectory: list[StateInfo | Action] = []
    for url in urls:
        state: StateInfo = {"observation": {}, "info": {"page": PageLike(url)}}
        trajectory.append(state)
        # Minimal dummy action; evaluator does not inspect actions
        dummy_action: Action = cast(Action, {"answer": ""})
        trajectory.append(dummy_action)
    return trajectory


def make_pseudo_page(url: str) -> PseudoPage:
    class _FakePWPage:
        pass

    fake = cast(Page, _FakePWPage())
    return PseudoPage(fake, url)


def write_config(tmp_path: Path, reference_url: str, url_note: str = "EXACT") -> Path:
    config = {"eval": {"reference_url": reference_url, "url_note": url_note}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))
    return cfg_path


def test_exact_or_only_matches_any(tmp_path: Path):
    evaluator = TrajectoryURLExactEvaluator()
    cfg_path = write_config(tmp_path, "http://a.com/x |OR| http://a.com/y", url_note="EXACT")

    # Matches because one OR alternative is present
    traj = make_trajectory(["http://a.com/x"])  # includes x
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 1.0

    # Does not match when none of the OR alternatives are present
    traj = make_trajectory(["http://a.com/z"])  # neither x nor y
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 0.0


def test_exact_and_or_groups_require_all_groups(tmp_path: Path):
    evaluator = TrajectoryURLExactEvaluator()
    cfg_path = write_config(
        tmp_path,
        "http://a.com/x |OR| http://a.com/y |AND| http://b.com/z |OR| http://b.com/w",
        url_note="EXACT",
    )

    # Matches: one from first group (y) and one from second group (w)
    traj = make_trajectory(["http://a.com/y", "http://b.com/w"])
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 1.0

    # Fails: only first group satisfied
    traj = make_trajectory(["http://a.com/x"])  # missing any from second group
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 0.0


def test_and_exact_requires_all_groups(tmp_path: Path):
    evaluator = TrajectoryURLExactEvaluator()
    cfg_path = write_config(
        tmp_path,
        "http://a.com/x |AND| http://b.com/z",
        url_note="EXACT",
    )

    # Pass: both groups satisfied by trajectory URLs
    traj = make_trajectory(["http://a.com/x", "http://b.com/z"])
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 1.0

    # Fail: only first group satisfied
    traj = make_trajectory(["http://a.com/x"])
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 0.0


def test_gold_in_pred_substring_match(tmp_path: Path):
    evaluator = TrajectoryURLExactEvaluator()
    cfg_path = write_config(
        tmp_path,
        "https://site.com/products |AND| https://site.com/cart",
        url_note="GOLD in PRED",
    )

    # Matches by substring: product detail URL contains products, and exact cart URL
    traj = make_trajectory(["https://site.com/products/123", "https://site.com/cart"])
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 1.0

    # Fails: only one group satisfied
    traj = make_trajectory(["https://site.com/products/123"])  # missing cart group
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 0.0


def test_trailing_slash_normalization(tmp_path: Path):
    evaluator = TrajectoryURLExactEvaluator()
    cfg_path = write_config(tmp_path, "http://a.com/x/", url_note="EXACT")

    # clean_url should normalize trailing slash, so these should match
    traj = make_trajectory(["http://a.com/x"])
    assert evaluator(traj, cfg_path, make_pseudo_page("http://unused")) == 1.0
