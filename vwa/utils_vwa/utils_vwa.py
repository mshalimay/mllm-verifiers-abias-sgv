import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from benchmark_config import AUTH_DIR, CLASSIFIEDS_RESET_TOKEN, CLEAN_ENV_SCRIPT, RESET_ENV_SCRIPT
from browser_env import Action, ActionTypes, Trajectory
from browser_env.actions import is_equivalent
from browser_env.auto_login import is_expired_for_sites
from browser_env.env_config import NO_COOKIE_RESET_SITES, VWA_DOMAINS, WA_DOMAINS, Website
from browser_env.env_utils import get_running_websites, get_url_from_file

from core_utils.logger_utils import logger


# ===============================================================================
# Helpers for evaluation loop
# ===============================================================================
def early_stop(trajectory: Trajectory, thresholds: dict[str, int]) -> tuple[bool, str]:
    """Check whether need to stop early"""
    try:
        # reach the max step
        num_steps = (len(trajectory) - 1) / 2
        if num_steps >= thresholds["max_steps"]:
            return True, f"Unable to achieve this task after {thresholds['max_steps']} steps"

        last_k_actions: list[Action]
        action_seq: list[Action]

        # Case: parsing failure for k times
        k = thresholds["parsing_failure"]
        last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]

        if len(last_k_actions) >= k:
            if all([action["action_type"] == ActionTypes.NONE for action in last_k_actions]):
                return True, f"Failed to parse actions for {k} times"

        # Case: same action for k times
        k = thresholds["repeating_action"]
        last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
        action_seq = trajectory[1::2]  # type: ignore[assignment]

        if len(action_seq) == 0:
            return False, ""

        last_action: Action = action_seq[-1]

        if last_action["action_type"] != ActionTypes.TYPE:
            if len(last_k_actions) >= k:
                if all([is_equivalent(action, last_action) for action in last_k_actions]):
                    return True, f"Same action for {k} times"
        else:
            # check the action sequence
            if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
                return True, f"Same typing action for {k} times"
        return False, ""
    except Exception as e:
        logger.warning(f"Error in early_stop computation: {e}")
        return False, ""


# ===============================================================================
# Helpers for env setup, start, reset
# ===============================================================================
def build_site_list(sites: str | list[str], exc_sites: list[Website] = []) -> list[str]:
    if isinstance(sites, str):
        sites = [sites]

    if "all_vwa" in sites:
        sites = [website.value for website in VWA_DOMAINS if website not in exc_sites]
    elif "all_wa" in sites:
        sites = [website.value for website in WA_DOMAINS if website not in exc_sites]

    else:
        strings_to_check = [website.value for website in VWA_DOMAINS] + [website.value for website in WA_DOMAINS]
        invalid_sites = [site for site in sites if site not in strings_to_check]

        if len(invalid_sites) > 0:
            raise ValueError(f"Invalid site: {invalid_sites}. Valid sites: {strings_to_check}")
        return sites

    return sites


def get_websites_up_down(instance_id: int, env: str = "vwa") -> tuple[list[str], list[str]]:
    websites_up = get_running_websites(process_id=str(instance_id))
    websites_down = [website for website in VWA_DOMAINS if website not in websites_up]
    if env == "vwa":
        return [website.value for website in VWA_DOMAINS if website in websites_up], [website.value for website in websites_down]
    elif env == "wa":
        return [website.value for website in WA_DOMAINS if website in websites_up], [website.value for website in websites_down]
    return [], []


def clean_envs(instance_ids: list[int]) -> None:
    """Clean docker containers"""
    cmd = [CLEAN_ENV_SCRIPT, *[str(instance_id) for instance_id in instance_ids]]
    subprocess.run(cmd, check=True)


def _reset_environments(
    instance_id: int,
    domains: list[str],
    reset_env_script: str = RESET_ENV_SCRIPT,
    wait_for_reset: bool = True,
    env: str = "vwa",
    parallel: bool = False,
) -> None:
    # TODO: clean logic dependent on strings for websites.
    # TODO: reset environments available via python script now; can simplify further.
    domains = domains.copy()
    if env == "vwa":
        if "homepage" in domains:
            domains.remove("homepage")
            domains.append("homepagevwa")
    elif env == "wa":
        if "homepage" in domains:
            domains.remove("homepage")
            domains.append("homepagewa")

    else:
        raise ValueError(f"Invalid environment: {env}")

    # Build base command - use module format for .py scripts, direct execution otherwise
    if reset_env_script.endswith(".py"):
        base_cmd = ["python", reset_env_script]
    else:
        base_cmd = [reset_env_script]

    if not parallel:
        cmd_args = [*base_cmd, "-p", str(instance_id), *domains]
        try:
            logger.info(f"Resetting {domains}. Command: {cmd_args}. Parallel: {parallel}")
            if wait_for_reset:
                subprocess.run(cmd_args, check=True)
            else:
                subprocess.Popen(cmd_args)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to reset environments: {e.stderr}")

    else:
        # Run non-homepage domains in parallel using Popen, then run homepage last
        homepage_variants = {"homepage", "homepagevwa", "homepagewa"}
        homepage_domain = next((d for d in domains if d in homepage_variants), None)
        other_domains = [d for d in domains if d != homepage_domain]

        procs: list[tuple[subprocess.Popen[bytes], str]] = []
        for domain in other_domains:
            domain_cmd = [*base_cmd, "-p", str(instance_id), domain]
            logger.info(f"Resetting {domain}. Command: {domain_cmd}")
            try:
                proc = subprocess.Popen(domain_cmd)
                procs.append((proc, domain))
            except Exception as e:
                logger.error(f"Failed to start reset for {domain}: {e}")

        failures: list[tuple[str, int]] = []
        if wait_for_reset and procs:
            for proc, domain in procs:
                ret = proc.wait()
                if ret != 0:
                    failures.append((domain, ret))

        # Homepage runs last to ensure all links in it are correct
        if homepage_domain:
            homepage_cmd = [*base_cmd, "-p", str(instance_id), homepage_domain]
            logger.info(f"Resetting {homepage_domain} last. Command: {homepage_cmd}")
            try:
                if wait_for_reset:
                    ret = subprocess.run(homepage_cmd, check=False).returncode
                    if ret != 0:
                        failures.append((homepage_domain, ret))
                else:
                    subprocess.Popen(homepage_cmd)
            except Exception as e:
                logger.error(f"Failed to start reset for {homepage_domain}: {e}")

        if failures:
            failed_str = ", ".join([f"{d} (code {rc})" for d, rc in failures])
            logger.error(f"One or more resets failed: {failed_str}")


def reset_envs_with_retry(
    instance_id: int,
    reset_env_script: str = RESET_ENV_SCRIPT,
    domains: list[str] | str = "all_vwa",
    wait_for_reset: bool = True,
    env: str = "vwa",
    force_homepage: bool = True,
    max_attempts: int = 3,
    parallel: bool = False,
) -> tuple[bool, list[str]]:
    """
    Reset environments with retry.

    Args:
        instance_id (int): The instance ID of the Docker container.
        reset_env_script (str, optional): The script to reset the environments. Defaults to RESET_ENV_SCRIPT.
        domains (list[str] | str, optional): The domains to reset. Defaults to "all_vwa".
        wait_for_reset (bool, optional): Whether to wait for the reset to complete. Defaults to True.
        env (str, optional): String identifying the benchmark. Defaults to "vwa". Options: "vwa", "wa".
        force_homepage (bool, optional): Whether to force the reset of the homepage. Defaults to True.
        max_attempts (int, optional): The maximum number of attempts to reset the environments. Defaults to 3.

    Returns:
        tuple[int, list[str]]:
            - int: 1 if success, 0 if failed
            - list[str]: list of domains that failed to reset
    """
    sites_to_reset = build_site_list(domains)
    sites_to_retry = []

    if not sites_to_reset:
        return True, []

    # If force_homepage is True, add homepage to domains
    homepage_added = False
    if force_homepage and "homepage" not in sites_to_reset:
        homepage_added = True
        sites_to_reset.append("homepage")

    # Secure that all domains are valid for the environment
    strings_to_check = [website.value for website in VWA_DOMAINS] if env == "vwa" else [website.value for website in WA_DOMAINS]
    for domain in sites_to_reset:
        assert domain in strings_to_check, f"Domain {domain} not found in {strings_to_check} for env {env}"

    if Website.CLASSIFIEDS.value in sites_to_reset:
        classifieds_reset_success = try_token_reset_classifieds(instance_id)
        if classifieds_reset_success:
            sites_to_reset.remove(Website.CLASSIFIEDS.value)

            if len(sites_to_reset) == 1 and sites_to_reset[0] == "homepage" and homepage_added:
                _, websites_down = get_websites_up_down(instance_id=instance_id, env=env)
                if "homepage" not in websites_down:
                    logger.info("Classifieds is the only site to reset, token reset successful, and only homepage is left. Skipping homepage.")
                    return True, []

    # Initial start/reset
    _reset_environments(
        instance_id=instance_id,
        reset_env_script=reset_env_script,
        domains=sites_to_reset,
        wait_for_reset=wait_for_reset,
        parallel=parallel,
    )

    # If some domains failed, retry up to max_attempts times.
    must_reset_homepage = False
    for i in range(max_attempts):
        sites_to_retry = []
        _, websites_down = get_websites_up_down(instance_id=instance_id, env=env)
        for req_site in sites_to_reset:
            if req_site in websites_down:
                sites_to_retry.append(req_site)

        if not sites_to_retry:
            break

        print(f"{__file__}: Start/Reset of for instance {instance_id} failed for {sites_to_retry}. Retrying...")
        print(f"[INFO] Resetting {sites_to_retry}. Attempt {i + 1} of {max_attempts}.")
        _reset_environments(
            instance_id=instance_id,
            reset_env_script=reset_env_script,
            domains=sites_to_retry,
            wait_for_reset=wait_for_reset,
            parallel=parallel,
        )
        must_reset_homepage = True

    # If some envs were retried, force reset homepage to ensure all links in it are correct
    if must_reset_homepage:
        sites_to_retry.append("homepage")
        for i in range(max_attempts):
            _reset_environments(
                instance_id=instance_id,
                reset_env_script=reset_env_script,
                domains=["homepage"],
                wait_for_reset=wait_for_reset,
                parallel=False,
            )
            _, websites_down = get_websites_up_down(instance_id=instance_id, env=env)
            if "homepage" not in websites_down:
                sites_to_retry.remove("homepage")
                break

    if not sites_to_retry:
        return True, []
    else:
        return False, sites_to_retry


def maybe_reset_environments(
    instance_id: int,
    env: str,
    config_dict: dict[str, Any],
    require_all_sites_up: bool = True,
    force_reset: bool = False,
    force_reset_site_list: list[str] = [],
    force_homepage: bool = True,
    skip_reset: bool = False,
) -> tuple[bool, list[str]]:
    """Reset environments for each task before its execution.

    Args:
        instance_id (int): Docker instance ID
        env (str): Environment to run the test on.
        config_dict (dict[str, Any] | None, optional): Task configuration dictionary. Defaults to None.
        require_all_sites_up (bool, optional): Whether to require all sites to be up. Defaults to True.
        force_reset (bool, optional): Whether to force reset the websites for current task. Defaults to False.
    """
    sites_to_reset = set()
    sites_failed_reset = []
    websites_up, websites_down = get_websites_up_down(instance_id=instance_id, env=env)

    # Add down websites to start/reset list
    if require_all_sites_up and len(websites_down) > 0:
        # If require all sites up, add down sites to reset list
        sites_to_reset.update(websites_down)
    else:
        # Else, add only required sites that are down to the reset list
        for req_site in config_dict["sites"]:
            if req_site not in websites_up:
                logger.info(f"{__file__}: Required site {req_site} is down. Adding to reset list.")
                sites_to_reset.add(req_site)

    # If skip reset, only reset (start) required sites that are down
    if skip_reset and len(sites_to_reset) == 0:
        return True, []

    # If the task requires reset, add the sites to the reset list
    if config_dict["require_reset"] and not skip_reset:
        logger.info(f"{__file__}: Task requires reset. Adding {config_dict['sites']} to reset list.")
        [sites_to_reset.add(site) for site in config_dict["sites"]]

    # If force_reset, add sites in the config file to reset list or use the provided list
    if force_reset and not skip_reset:
        if not force_reset_site_list:
            force_reset_site_list = config_dict["sites"]
            logger.info(f"{__file__}: Force reset enabled but site list not provided. Using {config_dict['sites']}.")
        else:
            logger.info(f"{__file__}: Force reset enabled. Adding {force_reset_site_list} to reset list.")
            [sites_to_reset.add(site) for site in force_reset_site_list]

    # Reset websites
    if sites_to_reset:
        logger.info(f"{__file__}: Resetting environments for {sites_to_reset}")
        reset_success, sites_failed_reset = reset_envs_with_retry(
            instance_id=instance_id,
            domains=list(sites_to_reset),
            wait_for_reset=True,
            max_attempts=3,
            force_homepage=force_homepage,
            env=env,
        )
        return reset_success, sites_failed_reset
    else:
        return True, []


def try_token_reset_classifieds(instance_id: int) -> bool:
    logger.info(f"{__file__}: Trying to reset classifieds with token {CLASSIFIEDS_RESET_TOKEN}.")
    try:
        classifieds_url = get_url_from_file(Website.CLASSIFIEDS, instance_id)
        reset_success = False

        if not classifieds_url:
            return reset_success

        if classifieds_url:
            response = requests.post(
                f"{classifieds_url}/index.php?page=reset",
                data={"token": CLASSIFIEDS_RESET_TOKEN},
            )

            # Check if the request was successful
            if response.status_code == 200:
                reset_success = True
                logger.info(f"{__file__}: Classifieds token reset successful.")
            else:
                reset_success = False
                logger.info(f"{__file__}: Classifieds token reset failed. Retrying with docker.")
        return reset_success
    except Exception as _:
        logger.info(f"{__file__}: Classifieds token reset failed. Retrying with docker.")
        return False


def reset_cookies_with_retry(
    docker_instance_id: str,
    sites: list[str] | str = [],
    exc_comb: bool = False,
    expired_only: bool = False,
    auth_folder: str = AUTH_DIR,
    wait_for_cookies_reset: bool = True,
    max_attempts: int = 3,
    env: str = "vwa",
) -> tuple[bool, list[str]]:
    """
    Reset cookies with retry.

    Args:
        docker_instance_id (str): The instance ID of the Docker container.
        sites (list[str] | str, optional): The sites to reset. Defaults to [] = all sites.
        exc_comb (bool, optional): Whether to exclude combinations of sites. Defaults to False.
        expired_only (bool, optional): Whether to reset only expired cookies. Defaults to False.
        auth_folder (str, optional): The folder to store the auth state. Defaults to AUTH_DIR.
    """
    if not sites:
        sites = "all_vwa" if env == "vwa" else "all_wa"
    initial_sites = build_site_list(sites, exc_sites=NO_COOKIE_RESET_SITES)
    sites_to_reset = initial_sites.copy()

    if expired_only:
        expired_sites = is_expired_for_sites(docker_instance_id=docker_instance_id, site_names=initial_sites, auth_folder=auth_folder, exc_comb=exc_comb)
        sites_to_reset = [site for site in initial_sites if site in expired_sites]

        if not sites_to_reset:
            # print(print_template.format(msg="[INFO] Expired cookies set to true and no expired cookies to reset."))
            return True, []
        else:
            logger.info(f"Expired sites: {expired_sites}. Resetting cookies for {sites_to_reset}.")

    try:
        # First attempt to reset cookies
        code, msg = _reset_cookies(
            docker_instance_id=docker_instance_id,
            sites=sites_to_reset,
            exc_comb=exc_comb,
            auth_folder=auth_folder,
            wait_for_cookies_reset=wait_for_cookies_reset,
        )
        # If errors during reset_cookies script execution, log and return
        if code == -1:
            logger.error(f"Failed to reset cookies: {msg}", exc_info=True)
            return False, initial_sites

        expired_sites = is_expired_for_sites(docker_instance_id, sites_to_reset, auth_folder, exc_comb)
        if not expired_sites:
            return True, []

        # If no error but cookies still expired, Ignores expired_only and full reset of cookies for all sites.
        sites_to_reset = initial_sites
        logger.info(f"Cookies expired for {expired_sites}. Retrying. All: {sites_to_reset}")
        for i in range(max_attempts):
            logger.info(f"Retrying cookies reset for env {docker_instance_id}, sites {sites_to_reset}. Attempt {i + 1} of {max_attempts}.")
            try:
                _reset_cookies(
                    docker_instance_id=docker_instance_id,
                    sites=sites_to_reset,
                    exc_comb=exc_comb,
                    auth_folder=auth_folder,
                    wait_for_cookies_reset=wait_for_cookies_reset,
                )
            except Exception as e:
                logger.error(f"Failed to reset cookies: {e}. Retrying...", exc_info=True)
                time.sleep(1)

            expired_sites = is_expired_for_sites(docker_instance_id, sites_to_reset, auth_folder, exc_comb)
            if not expired_sites:
                logger.info(f"Cookies reset successful for {sites_to_reset}.")
                return True, []

        logger.info(f"Failed to reset cookies for {sites_to_reset} after {max_attempts} attempts.")
        return False, sites_to_reset

    except Exception as e:
        logger.error(f"Failed to reset cookies: {e}", exc_info=True)
        return False, sites_to_reset


def _reset_cookies(
    docker_instance_id: str,
    sites: list[str] | str,
    exc_comb: bool = False,
    auth_folder: str = AUTH_DIR,
    wait_for_cookies_reset: bool = True,
) -> tuple[bool, str]:
    """
    Reset cookies by calling the auto_login module.
    """
    if isinstance(sites, str):
        sites_to_reset = [sites]
    elif isinstance(sites, list):
        sites_to_reset = sites.copy()
    if not sites_to_reset:
        return True, ""

    logger.info(f"Resetting cookies for {sites_to_reset}, auth folder {auth_folder}, docker instance id {docker_instance_id}")
    cmd = ["python", "-m", "browser_env.auto_login", "--p", str(docker_instance_id), "--auth_folder", auth_folder]
    cmd.extend(["--site_list", *sites_to_reset]) if sites_to_reset else None
    cmd.append("--exc_comb") if exc_comb else None
    try:
        if wait_for_cookies_reset:
            subprocess.run(cmd, check=True)
            return True, ""
        else:
            subprocess.Popen(cmd)
            return True, ""
    except subprocess.CalledProcessError as e:
        return False, str(e.stderr)
    except Exception as e:
        return False, str(e)


# ===============================================================================
# File helpers
# ===============================================================================


def get_uids_from_txt(txt_path: str) -> set[str]:
    with open(txt_path, "r") as f:
        lines = f.read().splitlines()
        uids = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".json"):
                continue
            else:
                uids.add(line.strip())
        return uids


def get_uids_from_csv(csv_path: str | Path) -> set[str]:
    try:
        if not os.path.exists(csv_path):
            return set()
        df = pd.read_csv(csv_path)
        if df.empty:
            return set()
        # uid is domain_task_id_env
        if "domain_task_id_env" not in df.columns:
            # Create domain_task_id_env column
            df["domain_task_id_env"] = df["domain"].astype(str) + "_" + df["task_id"].astype(str)
            df = df.dropna(subset=["score", "domain_task_id_env"])
        return set(df["domain_task_id_env"].astype(str))
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return set()


class TrajectoryView:
    def __init__(self, trajectory: list[Any]):
        self.trajectory = trajectory
        self.task_config: dict[str, Any] | None = None

    @property
    def states(self) -> list[Any]:
        return self.trajectory[::2]  # Even indices are state infos

    @property
    def actions(self) -> list[Any]:
        return self.trajectory[1::2]  # Odd indices are actions

    @property
    def valid_actions(self) -> list[Any]:
        return [action for action in self.actions if "invalid" not in action]
