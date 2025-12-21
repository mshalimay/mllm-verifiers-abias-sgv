#!/usr/bin/env python3

import argparse
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

CHECK_SITES_MAX_ATTEMPTS = 30
MIN_PORT = 10000
MAX_PORT = 20000
# ---------------------------
# Config loading (script-relative)
# ---------------------------


def _vwa_root_from_this_file() -> Path:
    # vwa/scripts/environments/start_reset_envs.py -> vwa/
    return Path(__file__).resolve().parents[2]


def load_env_constants() -> Dict[str, str]:
    """Load constants directly from `benchmark_config.constants`.

    We keep returning a dict because it's convenient to pass to subprocesses as an
    environment overlay (mirrors what the bash script achieved via `export`).
    """

    vwa_root = _vwa_root_from_this_file()
    if str(vwa_root) not in sys.path:
        sys.path.insert(0, str(vwa_root))

    from benchmark_config.constants import (
        BASE_URL,
        CLASSIFIEDS_DOCKER_COMPOSE_DIR,
        CLASSIFIEDS_RESET_TOKEN,
        DOCKER_IMGS_PATH,
        HOMEPAGE_PATH_VWA,
        HOMEPAGE_PATH_WA,
        URLS_DIR,
        URLS_FILE_TEMPLATE,
    )

    return {
        "URLS_DIR": str(URLS_DIR),
        "URLS_FILE_TEMPLATE": str(URLS_FILE_TEMPLATE),
        "CLASSIFIEDS_DOCKER_COMPOSE_DIR": str(CLASSIFIEDS_DOCKER_COMPOSE_DIR),
        "DOCKER_IMGS_PATH": str(DOCKER_IMGS_PATH),
        "HOMEPAGE_PATH_WA": str(HOMEPAGE_PATH_WA),
        "HOMEPAGE_PATH_VWA": str(HOMEPAGE_PATH_VWA),
        "BASE_URL": str(BASE_URL),
        "CLASSIFIEDS_RESET_TOKEN": str(CLASSIFIEDS_RESET_TOKEN),
    }


# ---------------------------
# Subprocess helpers
# ---------------------------


def run(
    cmd: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    return subprocess.run(
        list(cmd),
        check=check,
        capture_output=capture,
        text=text,
        env=env,
    )


# ---------------------------
# HTTP health checks
# ---------------------------


def _read_urls_file_lines(urls_file: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for raw in Path(urls_file).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        # expected: "site-name url"
        parts = line.split()
        if len(parts) < 2:
            continue
        pairs.append((parts[0], parts[1]))
    return pairs


def _site_matches(site: str, *, process_id: Optional[str], site_names: Sequence[str]) -> bool:
    if process_id and not site_names:
        return bool(re.search(rf"-{re.escape(process_id)}$", site))

    if not site_names:
        return True

    for s in site_names:
        if process_id:
            if site == f"{s}-{process_id}":
                return True
        else:
            if site == s or site.startswith(f"{s}-"):
                return True
    return False


def _http_is_up(url: str, *, timeout_s: float = 5.0) -> bool:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "start_reset_envs.py"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            # Accept any 2xx/3xx response (curl -f would fail for 400+)
            status = getattr(resp, "status", None)
            if status is None:
                # Fall back: if we got a response object, treat as up
                return True
            return 200 <= int(status) < 400
    except Exception:
        return False


def check_websites_python(
    *,
    urls_file: str,
    process_id: Optional[str] = None,
    site_names: Optional[Sequence[str]] = None,
    timeout_s: float = 5.0,
) -> List[str]:
    """Checks the status of websites listed in a URLs file.

    "reddit-1         UP         URL: http://127.0.0.1:12345"
    """

    site_names = list(site_names or [])
    results: List[str] = []
    for site, url in _read_urls_file_lines(urls_file):
        if not _site_matches(site, process_id=process_id, site_names=site_names):
            continue

        up = _http_is_up(url, timeout_s=timeout_s)
        status = "UP" if up else "DOWN"
        # Keep formatting compatible with the bash printf:
        # printf "%-15s %-10s %s\n" "$site" "UP" "URL: $url"
        results.append(f"{site:<15} {status:<10} URL: {url}")

    return results


def which_or_raise(prog: str) -> None:
    if shutil.which(prog) is None:
        raise RuntimeError(f"Required program not found on PATH: {prog}")


# ---------------------------
# Port helpers
# ---------------------------


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def get_free_port(
    *,
    min_port: int = MIN_PORT,
    max_port: int = MAX_PORT,
    attempts: int = 100,
) -> int:
    for _ in range(attempts):
        port = random.randint(min_port, max_port)
        if is_port_free(port):
            return port

    # fall back to ephemeral
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------
# Docker helpers (CLI-based)
# ---------------------------


def does_container_exist(name: str, *, env: Dict[str, str]) -> bool:
    # docker ps -a --format '{{.Names}}'
    cp = run(["docker", "ps", "-a", "--format", "{{.Names}}"], env=env, capture=True)
    names = set(cp.stdout.split())
    return name in names


def remove_container_and_volumes(name: str, *, keep_volumes: bool = False, env: Dict[str, str]) -> None:
    if not does_container_exist(name, env=env):
        return

    print(f"Removing container {name}...")

    # volumes attached to container (ignore bind mounts)
    cp = run(
        [
            "docker",
            "inspect",
            "-f",
            '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}} {{end}}{{end}}',
            name,
        ],
        env=env,
        capture=True,
        check=False,
    )
    volumes = [v for v in cp.stdout.strip().split() if v]

    run(["docker", "rm", "-f", name], env=env, check=False)

    if keep_volumes:
        print(f"Keeping volumes for {name}.")
        return

    for v in volumes:
        print(f"Removing associated volume {v}...")
        run(["docker", "volume", "rm", "-f", v], env=env, check=False)


def docker_run_http80(
    *,
    container_name: str,
    image: str,
    let_os_decide_port: bool,
    env: Dict[str, str],
) -> int:
    # Equivalent of:
    #   docker run --name NAME -p :80 -d IMAGE
    # or
    #   docker run --name NAME -p HOSTPORT:80 -d IMAGE
    port = None if let_os_decide_port else get_free_port()
    port_arg = ":80" if let_os_decide_port else f"{port}:80"
    run(["docker", "run", "--name", container_name, "-p", port_arg, "-d", image], env=env)

    if not port:
        port = int(get_docker_mapped_port(container_name, max_wait_s=10, env=env))

    return port


def get_docker_mapped_port(container_name: str, *, max_wait_s: int = 10, env: Dict[str, str]) -> str:
    waited = 0
    sleep_s = 1
    while waited < max_wait_s:
        cp = run(["docker", "port", container_name], env=env, capture=True, check=False)
        out = (cp.stdout or "").strip()
        if out:
            # Example line: "80/tcp -> 0.0.0.0:32768" or "80/tcp -> :::32768"
            m = re.search(r":(\d+)", out.splitlines()[0])
            if m:
                return m.group(1)
        time.sleep(sleep_s)
        waited += sleep_s

    raise TimeoutError(f"Timed out waiting for docker port mapping for {container_name}")


def wait_for_mysql(container_name: str, *, max_wait_s: int = 60, env: Dict[str, str]) -> bool:
    print(f"Waiting for MySQL to be ready in {container_name}...")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        cp = run(
            [
                "docker",
                "exec",
                container_name,
                "mysql",
                "-u",
                "magentouser",
                "-pMyPassword",
                "-e",
                "SELECT 1;",
            ],
            env=env,
            check=False,
            capture=True,
        )
        if cp.returncode == 0:
            print(f"MySQL is ready in {container_name}.")
            return True
        print("MySQL not ready yet, waiting...")
        time.sleep(2)

    print(f"Warning: MySQL in {container_name} did not become ready within {max_wait_s} seconds.")
    return False


def disable_shopping_indexing(container_name: str, *, env: Dict[str, str]) -> None:
    cmds = [
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalogrule_product"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalogrule_rule"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalogsearch_fulltext"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalog_category_product"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "customer_grid"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "design_config_grid"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "inventory"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalog_product_category"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalog_product_attribute"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "catalog_product_price"],
        ["/var/www/magento2/bin/magento", "indexer:set-mode", "schedule", "cataloginventory_stock"],
    ]
    for cmd in cmds:
        run(["docker", "exec", container_name, *cmd], env=env, check=False)


def docker_compose_up(
    *,
    project: str,
    compose_file: str,
    services: Sequence[str],
    env: Dict[str, str],
    extra_args: Optional[Sequence[str]] = None,
) -> None:
    cmd = ["docker", "compose", "-p", project, "-f", compose_file, "up", "-d", *services]
    if extra_args:
        cmd.extend(list(extra_args))
    run(cmd, env=env)


def wait_container_healthy(container_name: str, *, max_wait_s: int = 600, env: Dict[str, str]) -> bool:
    print(f"Waiting for {container_name} to become healthy...")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        cp = run(
            [
                "docker",
                "inspect",
                "-f",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}",
                container_name,
            ],
            env=env,
            capture=True,
            check=False,
        )
        health = (cp.stdout or "").strip() if cp.returncode == 0 else "unknown"
        if health == "healthy":
            print(f"{container_name} is healthy.")
            return True
        time.sleep(5)
    return False


# ---------------------------
# Homepage process helpers
# ---------------------------


def _pgrep(pattern: str) -> List[int]:
    cp = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if cp.returncode != 0:
        return []
    return [int(x) for x in cp.stdout.split() if x.strip().isdigit()]


def _ps_command(pid: int) -> str:
    cp = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return (cp.stdout or "").strip()


def kill_homepage_processes(homepage_path: str, process_id: str) -> None:
    # Ported from bash kill_homepage_processes
    for pid in _pgrep("app.py"):
        cmd = _ps_command(pid)

        port: Optional[str] = None
        m = re.search(r"--port\s+(\d+)", cmd)
        if m:
            port = m.group(1)

        # Fallback: query ss
        if port is None:
            try:
                cp = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False)
                # find line containing pid=PID,
                for line in (cp.stdout or "").splitlines():
                    if f"pid={pid}," in line:
                        m2 = re.search(r":(\d+)\b", line)
                        if m2:
                            port = m2.group(1)
                            break
            except FileNotFoundError:
                port = None

        if port is None:
            print(f"PID {pid}: could not determine port; skipping")
            continue

        # Query /process_id
        try:
            cp = subprocess.run(
                ["curl", "-sS", "--max-time", "1", f"http://127.0.0.1:{port}/process_id"],
                capture_output=True,
                text=True,
                check=False,
            )
            instance_id = (cp.stdout or "").strip()
        except FileNotFoundError:
            # If curl not available, fall back to killing by command match
            instance_id = ""

        if instance_id == process_id:
            print(f"Killing homepage Flask process with ID {instance_id}, PID {pid} on port {port}...")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            print(f"Skipping PID {pid} on port {port} (instance_id={instance_id})")


# ---------------------------
# URL file helpers
# ---------------------------


def write_all_instance_urls(
    *,
    process_id: str,
    urls_file: str,
    base_url: str,
    env: Dict[str, str],
) -> None:
    "+Write all UP websites for this instance to the URL file (mirrors bash)."  # noqa: D400

    Path(urls_file).parent.mkdir(parents=True, exist_ok=True)

    temp_urls_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    temp_urls_path = temp_urls_file.name
    temp_urls_file.close()

    # Get all containers and their port mappings
    cp = run(["docker", "ps", "-a", "--format", "{{.Names}} {{.Ports}}"], env=env, capture=True)
    for line in (cp.stdout or "").splitlines():
        # match site-name-ID first token
        m = re.match(
            r"^(reddit-\d+|shopping-\d+|shopping_admin-\d+|gitlab-\d+|classifieds-\d+|wikipedia-\d+|homepage-\d+)",
            line,
        )
        if not m:
            continue
        container_name = m.group(1)
        if not container_name.endswith(f"-{process_id}"):
            continue

        # Extract host port from Ports column
        # Example: "0.0.0.0:32768->80/tcp"
        m2 = re.search(r":(\d+)->", line)
        if m2:
            port = m2.group(1)
            with open(temp_urls_path, "a") as f:
                f.write(f"{container_name} http://{base_url}:{port}\n")

    # Add homepage flask processes
    for pid in _pgrep(r"app.py [0-9]+ --port"):
        cmd = _ps_command(pid)
        m = re.search(r"app\.py\s+(\d+)\s+--port\s+(\d+)", cmd)
        if not m:
            continue
        instance_id, port = m.group(1), m.group(2)
        if instance_id != process_id:
            continue
        with open(temp_urls_path, "a") as f:
            f.write(f"homepage-{instance_id} http://{base_url}:{port}\n")

    # Determine UP/DOWN using Python (replacement for check_websites.sh)
    status_lines = check_websites_python(
        urls_file=temp_urls_path,
        process_id=process_id,
        site_names=None,
    )

    # Mimic bash filtering for UP
    up_lines: List[str] = []
    for line in status_lines:
        if " UP " in line:
            # format: site  UP  URL: url
            parts = line.split()
            if len(parts) >= 4:
                site = parts[0]
                url = parts[3]
                up_lines.append(f"{site} {url}")

    Path(urls_file).write_text("\n".join(up_lines) + ("\n" if up_lines else ""))

    # Cleanup
    try:
        os.unlink(temp_urls_path)
    except OSError:
        pass


# ---------------------------
# Main orchestration
# ---------------------------


ALL_VWA_SITES = ["shopping", "classifieds", "reddit", "wikipedia", "homepagevwa"]
ALL_WA_SITES = ["shopping", "shopping_admin", "reddit", "gitlab", "homepagewa"]


def find_available_id(*, env: Dict[str, str]) -> str:
    # Mirrors find_available_id() in bash
    used_ids: set[int] = set()
    cp = run(["docker", "ps", "-a", "--format", "{{.Names}}"], env=env, capture=True, check=False)
    for name in (cp.stdout or "").splitlines():
        m = re.search(r"[-_]?([0-9]+)$", name.strip())
        if m:
            used_ids.add(int(m.group(1)))

    i = 1
    while i in used_ids:
        i += 1
    return str(i)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start/reset WebArena / VisualWebArena sites (Python port of start_reset_envs.sh).")
    p.add_argument(
        "-p",
        dest="process_id",
        default=None,
        help="Instance ID for docker instances. If omitted, the script finds the next available ID.",
    )
    p.add_argument(
        "sites",
        nargs="+",
        help="Sites to start/reset. Special: all_vwa, all_wa",
    )
    p.add_argument(
        "--manual-port",
        action="store_true",
        default=False,
        help="If provided, script tries to get a free port; otherwise, let Docker decide.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    which_or_raise("docker")
    # docker compose is a subcommand; ensure docker exists and hope plugin exists.

    env_constants = load_env_constants()
    # Provide a baseline environment for subprocesses
    child_env = os.environ.copy()
    child_env.update(env_constants)

    args = parse_args(argv)

    sites = list(args.sites)
    if "all_vwa" in sites:
        sites = ALL_VWA_SITES
        print(f"Starting/resetting all VisualWebArena websites: {sites}.")

    if "all_wa" in sites:
        sites = ALL_WA_SITES
        print(f"Starting/resetting all WebArena websites: {sites}.")

    if "homepagewa" in sites and "homepagevwa" in sites:
        print("Error: both 'homepagewa' and 'homepagevwa' cannot be specified together.")
        return 1

    process_id = args.process_id or find_available_id(env=child_env)
    if args.process_id is None:
        print(f"No instance ID provided. Using next available ID: {process_id}")

    base_url = env_constants["BASE_URL"]

    print(f"{time.ctime()} Starting/resetting websites...")

    temp_url_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    temp_url_path = temp_url_file.name
    temp_url_file.close()

    def append_temp(name: str, url: str) -> None:
        with open(temp_url_path, "a") as f:
            f.write(f"{name} {url}\n")

    # Track sites that failed to start (docker errors)
    failed_sites: List[str] = []

    # --- Start sites ---

    # Reddit
    if "reddit" in sites:
        container_name = f"reddit-{process_id}"
        try:
            remove_container_and_volumes(container_name, env=child_env)
            port = docker_run_http80(
                container_name=container_name,
                image="postmill-populated-exposed-withimg",
                let_os_decide_port=not args.manual_port,
                env=child_env,
            )
            print(f"Started Reddit instance {process_id} at http://{base_url}:{port}.")
            append_temp(container_name, f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting Reddit: {e}")
            failed_sites.append("reddit")

    # Shopping
    if "shopping" in sites:
        container_name = f"shopping-{process_id}"
        try:
            remove_container_and_volumes(container_name, env=child_env)
            port = docker_run_http80(
                container_name=container_name,
                image="shopping_final_0712",
                let_os_decide_port=not args.manual_port,
                env=child_env,
            )

            if wait_for_mysql(container_name, max_wait_s=60, env=child_env):
                print(f"Configuring shopping instance {process_id}...")
                run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "/var/www/magento2/bin/magento",
                        "setup:store-config:set",
                        f"--base-url=http://{base_url}:{port}",
                    ],
                    env=child_env,
                    check=False,
                )
                run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "mysql",
                        "-u",
                        "magentouser",
                        "-pMyPassword",
                        "magentodb",
                        "-e",
                        f"UPDATE core_config_data SET value='http://{base_url}:{port}/' WHERE path = 'web/secure/base_url';",
                    ],
                    env=child_env,
                    check=False,
                )
                run(
                    ["docker", "exec", container_name, "/var/www/magento2/bin/magento", "cache:flush"],
                    env=child_env,
                    check=False,
                )
                disable_shopping_indexing(container_name, env=child_env)
            else:
                print(f"Error: Failed to configure shopping instance {process_id} due to MySQL connection issues.")

            print(f"Started shopping instance {process_id} at http://{base_url}:{port}.")
            append_temp(container_name, f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting shopping: {e}")
            failed_sites.append("shopping")

    # Shopping admin
    if "shopping_admin" in sites:
        container_name = f"shopping_admin-{process_id}"
        try:
            remove_container_and_volumes(container_name, env=child_env)
            port = docker_run_http80(
                container_name=container_name,
                image="shopping_admin_final_0719",
                let_os_decide_port=not args.manual_port,
                env=child_env,
            )

            if wait_for_mysql(container_name, max_wait_s=60, env=child_env):
                run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "/var/www/magento2/bin/magento",
                        "setup:store-config:set",
                        f"--base-url=http://{base_url}:{port}",
                    ],
                    env=child_env,
                    check=False,
                )
                run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "mysql",
                        "-u",
                        "magentouser",
                        "-pMyPassword",
                        "magentodb",
                        "-e",
                        f"UPDATE core_config_data SET value='http://{base_url}:{port}/' WHERE path = 'web/secure/base_url';",
                    ],
                    env=child_env,
                    check=False,
                )
                run(
                    ["docker", "exec", container_name, "/var/www/magento2/bin/magento", "cache:flush"],
                    env=child_env,
                    check=False,
                )
                print(f"Started shopping_admin instance {process_id} at http://{base_url}:{port}.")
            else:
                print(f"Error: Failed to configure shopping_admin instance {process_id} due to MySQL connection issues.")
                print(f"Started shopping_admin instance {process_id} at http://{base_url}:{port} (configuration may be incomplete).")

            append_temp(container_name, f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting shopping_admin: {e}")
            failed_sites.append("shopping_admin")

    # GitLab
    if "gitlab" in sites:
        container_name = f"gitlab-{process_id}"
        try:
            remove_container_and_volumes(container_name, env=child_env)

            port = get_free_port() if args.manual_port else None
            port_arg = f"{port}:{port}" if port else ":80"
            run(
                [
                    "docker",
                    "run",
                    "--name",
                    container_name,
                    "-d",
                    "-p",
                    port_arg,
                    "gitlab-populated-final-port8023",
                    "/opt/gitlab/embedded/bin/runsvdir-start",
                ],
                env=child_env,
            )
            if not port:
                port = int(get_docker_mapped_port(container_name, max_wait_s=10, env=child_env))

            run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sed",
                    "-i",
                    f"s|^external_url.*|external_url 'http://{base_url}:{port}'|",
                    "/etc/gitlab/gitlab.rb",
                ],
                env=child_env,
                check=False,
            )
            run(["docker", "exec", container_name, "gitlab-ctl", "reconfigure"], env=child_env, check=False)

            print(f"Started GitLab instance {process_id} at http://{base_url}:{port}.")
            append_temp(container_name, f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting GitLab: {e}")
            failed_sites.append("gitlab")

    # Classifieds
    if "classifieds" in sites:
        container_name = f"classifieds-{process_id}"
        db_container_name = f"classifieds_db-{process_id}"
        try:
            port = get_free_port()  # fixed port needed for templating

            compose_dir = env_constants["CLASSIFIEDS_DOCKER_COMPOSE_DIR"]
            reset_token = env_constants["CLASSIFIEDS_RESET_TOKEN"]
            template_file = Path(compose_dir) / "docker-compose-raw.yml"
            output_file = Path(compose_dir) / f"docker-compose-{process_id}.yml"

            remove_container_and_volumes(container_name, env=child_env)
            remove_container_and_volumes(db_container_name, env=child_env)

            # Template compose file (Python version of sed)
            content = template_file.read_text()
            content = content.replace("<classifieds_port>", str(port))
            content = content.replace("<your-server-hostname>", base_url)
            content = content.replace("<classifieds_reset_token>", reset_token)
            content = re.sub(
                r"container_name:\s*classifieds_db\b",
                f"container_name: classifieds_db-{process_id}",
                content,
            )
            content = re.sub(
                r"container_name:\s*classifieds\s*$",
                f"container_name: classifieds-{process_id}",
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(r"\bdb_data\b", f"db_data-{process_id}", content)
            output_file.write_text(content)

            project = f"classifieds-{process_id}"

            print(f"Starting {db_container_name}...")
            docker_compose_up(
                project=project,
                compose_file=output_file.as_posix(),
                services=["db"],
                env=child_env,
            )

            if not wait_container_healthy(db_container_name, max_wait_s=600, env=child_env):
                print(f"Warning: {db_container_name} did not become healthy within timeout.")

            # Populate classifieds database
            run(
                [
                    "docker",
                    "exec",
                    db_container_name,
                    "mysql",
                    "-u",
                    "root",
                    "-ppassword",
                    "osclass",
                    "-e",
                    "source /docker-entrypoint-initdb.d/osclass_craigslist.sql",
                ],
                env=child_env,
                check=False,
            )

            print(f"Starting {container_name}...")
            docker_compose_up(
                project=project,
                compose_file=output_file.as_posix(),
                services=["web"],
                env=child_env,
                extra_args=["--wait"],
            )

            print(f"Started classifieds instance {process_id} at http://{base_url}:{port}.")
            append_temp(container_name, f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting classifieds: {e}")
            failed_sites.append("classifieds")

    # Wikipedia
    if "wikipedia" in sites:
        container_name = f"wikipedia-{process_id}"
        try:
            wiki_path = env_constants["DOCKER_IMGS_PATH"]
            image = "ghcr.io/kiwix/kiwix-serve:3.3.0"
            zim = "wikipedia_en_all_maxi_2022-05.zim"

            remove_container_and_volumes(container_name, env=child_env)

            # Build docker run command with conditional port mapping
            port = get_free_port() if args.manual_port else None
            port_arg = f"{port}:80" if port else ":80"
            run(
                ["docker", "run", "-d", "--name", container_name, f"--volume={wiki_path}/:/data", "-p", port_arg, image, zim],
                env=child_env,
            )
            if not port:
                port = int(get_docker_mapped_port(container_name, max_wait_s=10, env=child_env))

            print(f"Started Wikipedia instance {process_id} at http://{base_url}:{port}.")
            append_temp(container_name, f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting Wikipedia: {e}")
            failed_sites.append("wikipedia")

    # Homepage
    if "homepagewa" in sites or "homepagevwa" in sites:
        try:
            if "homepagewa" in sites:
                homepage_path = env_constants["HOMEPAGE_PATH_WA"]
                hp_env = "wa"
                hp_site_name = "homepagewa"
            else:
                homepage_path = env_constants["HOMEPAGE_PATH_VWA"]
                hp_env = "vwa"
                hp_site_name = "homepagevwa"

            kill_homepage_processes(homepage_path, process_id)
            port = get_free_port()
            print(f"Starting homepage instance {process_id} at http://{base_url}:{port} ...")

            # run update_homepage.py
            script_root = _vwa_root_from_this_file()
            update_homepage = (script_root / "scripts" / "environments" / "update_homepage.py").as_posix()

            # Ensure PYTHONPATH includes vwa root so update_homepage can import browser_env
            run(
                [
                    sys.executable,
                    update_homepage,
                    process_id,
                    f"{homepage_path}/templates/index_raw.html",
                    f"{homepage_path}/templates/index-{process_id}.html",
                    "--base_url",
                    base_url,
                    "--env",
                    hp_env,
                ],
                env=child_env | {"PYTHONPATH": f"{script_root}{os.pathsep}{child_env.get('PYTHONPATH', '')}"},
                check=False,
            )

            # Start flask app in background (nohup-like)
            log_path = f"{homepage_path}/flask-{process_id}.log"
            with open(log_path, "a") as logf:
                subprocess.Popen(
                    [sys.executable, f"{homepage_path}/app.py", process_id, "--port", str(port)],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
            time.sleep(2)

            # Find PID
            pids = _pgrep(f"{re.escape(homepage_path)}/app.py {process_id}")
            pid = pids[0] if pids else None

            print(f"Started homepage instance {process_id} at http://{base_url}:{port}. Process ID: {pid}.")
            append_temp(f"homepage-{process_id}", f"http://{base_url}:{port}")
        except Exception as e:
            print(f"Error starting homepage: {e}")
            failed_sites.append(hp_site_name)

    # --- Wait for all websites to be up ---

    url_file = env_constants["URLS_FILE_TEMPLATE"].replace("{PROCESS_ID}", process_id)
    print(f"Waiting until {sites} are up...")

    attempt = 1
    while attempt <= CHECK_SITES_MAX_ATTEMPTS:
        print(f"\nAttempt {attempt}/{CHECK_SITES_MAX_ATTEMPTS}:")
        status_lines = check_websites_python(
            urls_file=temp_url_path,
            process_id=process_id,
            site_names=None,
        )
        out = "\n".join(status_lines)
        print(out.strip())

        are_all_up = True
        for line in out.splitlines():
            if " DOWN " in line:
                are_all_up = False
                break

        if are_all_up:
            print("All requested websites are up!")
            break

        if attempt == CHECK_SITES_MAX_ATTEMPTS:
            print(f"Warning: Some websites failed to start after {CHECK_SITES_MAX_ATTEMPTS} attempts.")
            print("\nFinal websites status:")
            final_status = check_websites_python(
                urls_file=temp_url_path,
                process_id=process_id,
                site_names=sites,
            )
            print("\n".join(final_status).strip())
            break

        time.sleep(5)
        attempt += 1

    print("Writing all UP websites for this instance to the URL file...")
    write_all_instance_urls(process_id=process_id, urls_file=url_file, base_url=base_url, env=child_env)

    print(f"{time.ctime()} Finished starting/resetting websites.")

    # Report any sites that failed during docker operations
    if failed_sites:
        print(f"\nWarning: The following sites failed during startup: {', '.join(failed_sites)}")
        print("These sites were skipped and may not appear in the final URL file.")

    # Cleanup temp
    try:
        os.unlink(temp_url_path)
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
