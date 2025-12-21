"""Script to automatically login each website"""
# NOTE:[mandrade]: substantially refactored to support environment parallelization, check for expired cookies, etc.

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from benchmark_config import AUTH_DIR
from playwright.sync_api import sync_playwright

from browser_env.env_config import NO_COOKIE_RESET_SITES, Website, get_username, get_username_and_password
from browser_env.env_utils import check_websites, get_running_websites, get_url_from_file, wait_for_page_to_stabilize
from core_utils.logger_utils import logger

HEADLESS = True
SLOW_MO = 0
TIMEOUT = 5000 * 100

URL_SUFFIXES = {
    Website.SHOPPING: "/wishlist/",
    Website.CLASSIFIEDS: "/index.php?page=user&action=items",
    Website.REDDIT: f"/user/{get_username(Website.REDDIT)}/account",
    Website.SHOPPING_ADMIN: "/dashboard",
    Website.GITLAB: "/-/profile",
}
KEYWORDS = {
    Website.SHOPPING: "",
    Website.CLASSIFIEDS: "My listings",
    Website.REDDIT: "Delete",
    Website.SHOPPING_ADMIN: "Dashboard",
    Website.GITLAB: "",
}
EXACT_MATCH = {
    Website.SHOPPING: True,
    Website.REDDIT: True,
    Website.CLASSIFIEDS: True,
    Website.SHOPPING_ADMIN: True,
    Website.GITLAB: True,
}


def is_expired(storage_state: Path, url: str, keyword: str, url_exact: bool = True) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url, timeout=TIMEOUT)
    wait_for_page_to_stabilize(
        page=page,
        logger=logger,
        min_num_trues=3,
        return_after=3,
    )
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


def get_websites_from_cookies_file_path(cookies_file_path: str) -> tuple[list[Website], int]:
    cookies_file_name = Path(cookies_file_path).name
    websites = [Website(website_name) for website_name in cookies_file_name.split("-")[0].split(".")]
    instance_id = int(cookies_file_name.split("-")[1].split("_")[0])
    return websites, instance_id


def create_reddit_account(page, url, username: str, password: str):
    if page.url != f"{url}/login":
        page.goto(f"{url}/login", timeout=TIMEOUT)
    page.locator('a.button.button--secondary[href="/registration"]').click()
    page.locator("#user_username").fill(username)
    page.locator("#user_password_first").fill(password)
    page.locator("#user_password_second").fill(password)
    page.get_by_role("button", name="Sign up").click()


def login_reddit_account(page, url, username: str, password: str):
    page.goto(f"{url}/login", timeout=TIMEOUT)
    page.get_by_label("Username").fill(username)
    page.get_by_label("Password").fill(password)
    page.get_by_role("button", name="Log in").click()


def renew_comb(websites: list[Website], instance_id: int, auth_folder: str = AUTH_DIR) -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    for website in websites:
        url = get_url_from_file(website, instance_id)
        username, password = get_username_and_password(website)

        if website == Website.SHOPPING:
            page.goto(f"{url}/customer/account/login/", timeout=TIMEOUT)
            page.get_by_label("Email", exact=True).fill(username)
            page.get_by_label("Password", exact=True).fill(password)
            page.get_by_role("button", name="Sign In").click()

        if website == Website.REDDIT:
            login_reddit_account(page, url, username, password)
            wait_for_page_to_stabilize(page=page, logger=logger, min_num_trues=3, return_after=3)
            if "Invalid credentials" in page.content():
                create_reddit_account(page, url, username, password)
                wait_for_page_to_stabilize(page=page, logger=logger, min_num_trues=3, return_after=3)
                login_reddit_account(page, url, username, password)

        if website == Website.CLASSIFIEDS:
            page.goto(f"{url}/index.php?page=login", timeout=TIMEOUT)
            page.locator("#email").fill(username)
            page.locator("#password").fill(password)
            page.get_by_role("button", name="Log in").click()

        if website == Website.SHOPPING_ADMIN:
            page.goto(f"{url}", timeout=TIMEOUT)
            page.get_by_placeholder("user name").fill(username)
            page.get_by_placeholder("password").fill(password)
            page.get_by_role("button", name="Sign in").click()

        if website == Website.GITLAB:
            page.goto(f"{url}/users/sign_in", timeout=TIMEOUT)
            page.get_by_test_id("username-field").click()
            page.get_by_test_id("username-field").fill(username)
            page.get_by_test_id("username-field").press("Tab")
            page.get_by_test_id("password-field").fill(password)
            page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"{auth_folder}/{'.'.join(websites)}-{instance_id}_state.json")
    context_manager.__exit__()


def get_cookie_paths_for_site(
    site_name: str,
    docker_instance_id: str,
    all_sites: list[str] = [],
    auth_folder: str = AUTH_DIR,
    exc_comb: bool = False,
) -> list[str]:
    website = Website(site_name)
    if website == Website.WIKIPEDIA or website == Website.MAP or website == Website.HOMEPAGE:
        return []

    all_cookies = []

    websites: list[Website] = []
    if not all_sites:
        websites = get_running_websites(process_id=docker_instance_id, sites_to_exclude=[Website.WIKIPEDIA, Website.MAP, Website.HOMEPAGE])
    else:
        websites = [Website(site) for site in all_sites]

    # Add site cookie path
    all_cookies.append(f"{auth_folder}/{site_name}-{docker_instance_id}_state.json")

    # If not including site combinations, return
    if exc_comb:
        return all_cookies

    # Add site combinations cookie paths
    all_pairs = list(combinations(websites, 2))
    # Keep only the pairs whose one of the sites is in the sites list
    pairs = [pair for pair in all_pairs if site_name in pair]
    for pair in pairs:
        if Website.REDDIT in pair and (Website.SHOPPING in pair or Website.SHOPPING_ADMIN in pair):
            continue
        all_cookies.append(f"{auth_folder}/{'.'.join(sorted(pair))}-{docker_instance_id}_state.json")
    return all_cookies


def is_expired_for_sites(
    docker_instance_id: str,
    site_names: list[str],
    auth_folder: str = AUTH_DIR,
    exc_comb=False,
) -> dict[str, list[str]]:
    if not isinstance(site_names, list):
        site_names = [site_names]

    site_to_cookies = {
        site: get_cookie_paths_for_site(
            site_name=site,
            exc_comb=exc_comb,
            docker_instance_id=docker_instance_id,
            auth_folder=auth_folder,
        )
        for site in site_names
    }

    expired_site_to_cookies = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Map each submitted future to its (site, cookie file) information
        future_to_info = {}
        for site_name, cookies in site_to_cookies.items():
            for c_file in cookies:
                site_list, instance_id = get_websites_from_cookies_file_path(c_file)
                for cur_site in site_list:
                    url_suffix = URL_SUFFIXES[cur_site]
                    url = get_url_from_file(cur_site, instance_id) + url_suffix
                    keyword = KEYWORDS[cur_site]
                    match = EXACT_MATCH[cur_site]
                    future = executor.submit(
                        is_expired,
                        Path(c_file),
                        url,
                        keyword,
                        match,
                    )
                    future_to_info[future] = (site_name, c_file)

        for future in future_to_info:
            site_name, c_file = future_to_info[future]
            if future.result():
                expired_site_to_cookies.setdefault(site_name, [])
                if c_file not in expired_site_to_cookies[site_name]:
                    expired_site_to_cookies[site_name].append(c_file)
    return expired_site_to_cookies


def get_websites_to_reset(args):
    websites: dict[str, list[Website]] = {}
    all_sites_data = check_websites(process_id=args.p)

    if args.exc_comb and args.site_list:
        all_sites_data = [site_data for site_data in all_sites_data if site_data["name"] in args.site_list]

    sites_down = []
    for site_data in all_sites_data:
        website = Website(site_data["name"])
        if website in NO_COOKIE_RESET_SITES:
            continue

        if site_data["status"] == 1:
            instance_id = site_data["instance_id"]
            if instance_id not in websites:
                websites[instance_id] = []

            websites[instance_id].append(Website(site_data["name"]))
        else:
            sites_down.append(site_data["name"])

    if len(sites_down) > 0:
        raise Exception(f"Websites are down: {sites_down}. Docker instance ID: {args.p}")

    return websites


def main(sites: dict[str, list[Website]], auth_folder: str = AUTH_DIR, exc_comb: bool = False) -> None:
    os.makedirs(auth_folder, exist_ok=True)

    pairs_per_instance = {}
    if not exc_comb:
        print("[INFO] Cookie creation: Including combinations of sites.")
        # All possible combinations:
        for instance_id, websites in sites.items():
            pairs_per_instance[instance_id] = list(combinations(websites, 2))
    else:
        print("[INFO] Cookie creation: Excluding combinations of sites.")

    all_pairs_flat = [(instance_id, site_pair) for instance_id in pairs_per_instance.keys() for site_pair in pairs_per_instance[instance_id]]
    all_sites_flat = [(instance_id, site) for instance_id in sites.keys() for site in sites[instance_id]]

    with ThreadPoolExecutor(max_workers=8) as executor:
        for instance_id, site_pair in all_pairs_flat:
            # Auth doesn't work on this pair as they share the same cookie
            if Website.REDDIT in site_pair and (Website.SHOPPING in site_pair or Website.SHOPPING_ADMIN in site_pair):
                continue
            executor.submit(renew_comb, list(sorted(site_pair)), instance_id=instance_id, auth_folder=auth_folder)

        for instance_id, site in all_sites_flat:
            executor.submit(renew_comb, [site], instance_id=int(instance_id), auth_folder=auth_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exc_comb", action="store_true", default=False)
    parser.add_argument("--site_list", nargs="+", default=[])
    parser.add_argument("--auth_folder", type=str, default=AUTH_DIR)
    parser.add_argument("--p", type=str, default="")
    args = parser.parse_args()

    main(auth_folder=args.auth_folder, sites=get_websites_to_reset(args), exc_comb=args.exc_comb)
