#!/usr/bin/env python3
"""
Idempotently ensure required (Visual)WebArena docker images are available.

- Downloads missing local .tar image archives (shopping/admin/reddit/gitlab)
- Loads images into Docker only if not already loaded
- Pulls remote images for classifieds (jykoh/classifieds, mysql)
- Downloads Wikipedia `.zim` if missing and pulls kiwix-serve image

Examples:
  python -m scripts.environments.ensure_docker_imgs all_vwa
  python -m scripts.environments.ensure_docker_imgs shopping reddit
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------
# Config loading (script-relative)
# ---------------------------


def _vwa_root_from_this_file() -> Path:
    # vwa/scripts/environments/ensure_docker_imgs.py -> vwa/
    return Path(__file__).resolve().parents[2]


def load_env_constants() -> Dict[str, str]:
    """Load constants directly from `benchmark_config.constants`."""
    vwa_root = _vwa_root_from_this_file()
    if str(vwa_root) not in sys.path:
        sys.path.insert(0, str(vwa_root))

    from benchmark_config.constants import (  # noqa: WPS433 (local import)
        CLASSIFIEDS_DOCKER_COMPOSE_DIR,
        DOCKER_IMGS_PATH,
    )

    return {
        "DOCKER_IMGS_PATH": str(DOCKER_IMGS_PATH),
        "CLASSIFIEDS_DOCKER_COMPOSE_DIR": str(CLASSIFIEDS_DOCKER_COMPOSE_DIR),
    }


CLASSIFIEDS_COMPOSE_ZIP_URL = "https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"


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
    return subprocess.run(
        list(cmd),
        check=check,
        capture_output=capture,
        text=text,
        env=env,
    )


def which_or_raise(prog: str) -> None:
    if shutil.which(prog) is None:
        raise RuntimeError(f"Required program not found on PATH: {prog}")


def log(msg: str) -> None:
    print(f"[ensure_docker_imgs] {msg}")


def ensure_writable_dir(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Expected directory but found file: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise PermissionError(
            "DOCKER_IMGS_PATH is not writable.\n"
            f"Path: {path}\n"
            "Fix by changing ownership/permissions, e.g.:\n"
            f"  sudo chown -R $USER:$USER {path}\n"
            f"  sudo chmod -R u+rwX {path}\n"
        )


def ensure_classifieds_sql_present(*, classifieds_compose_dir: Path) -> None:
    """Ensure classifieds DB seed SQL exists by unzipping it if needed.

    The docker-compose mounts `classifieds_docker_compose/mysql/` into the MySQL
    container at `/docker-entrypoint-initdb.d/`. Both `init_db.sh` and
    `start_reset_envs.*` expect `osclass_craigslist.sql` to exist there.
    """
    mysql_dir = classifieds_compose_dir / "mysql"
    sql_path = mysql_dir / "osclass_craigslist.sql"
    zip_path = mysql_dir / "osclass_craigslist.zip"

    if sql_path.exists():
        return

    # Ensure we can write the extracted SQL next to the zip (this directory is mounted into MySQL).
    mysql_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(mysql_dir, os.W_OK):
        raise PermissionError(
            "Classifieds seed directory is not writable; cannot extract SQL.\n"
            f"Path: {mysql_dir}\n"
            "Fix by changing ownership/permissions, e.g.:\n"
            f"  sudo chown -R $USER:$USER {mysql_dir}\n"
            f"  sudo chmod -R u+rwX {mysql_dir}\n"
        )

    if not zip_path.exists():
        log(
            "Warning: classifieds SQL seed is missing.\n"
            f"Expected either:\n  - {sql_path}\n  - {zip_path}\n"
            "Classifieds may fail to initialize/reset without this file."
        )
        return

    log(f"Extracting classifieds SQL seed: {zip_path} -> {sql_path}")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            # Find a member that ends with osclass_craigslist.sql (zip may contain nested paths)
            members = [m for m in zf.namelist() if m.endswith("osclass_craigslist.sql")]
            if not members:
                raise RuntimeError(f"No osclass_craigslist.sql found inside {zip_path}")
            member = members[0]
            with zf.open(member, "r") as src, open(sql_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    except Exception as e:  # noqa: BLE001
        log(f"Warning: failed to extract {zip_path}: {e}")
        return

    if sql_path.exists():
        log(f"OK: extracted {sql_path}")


def resolve_classifieds_compose_dir(*, docker_imgs_path: Path, configured: Path) -> Path:
    """Pick the classifieds compose directory to use.

    Some setups keep the compose bundle under DOCKER_IMGS_PATH (downloaded), e.g.
    `.../docker_imgs/classifieds_docker_compose/`.
    """
    if configured.exists():
        return configured
    candidate = docker_imgs_path / "classifieds_docker_compose"
    if candidate.exists():
        return candidate
    # Default to candidate so callers can create/populate it.
    return candidate


def ensure_classifieds_compose_bundle_present(
    *,
    classifieds_compose_dir: Path,
    docker_imgs_path: Path,
    env: Dict[str, str],
) -> None:
    """Ensure classifieds docker-compose bundle exists by downloading from archive.org if needed."""
    required = [
        classifieds_compose_dir / "docker-compose-raw.yml",
        classifieds_compose_dir / "mysql" / "init_db.sh",
        classifieds_compose_dir / "mysql" / "classifieds_restore.sql",
        classifieds_compose_dir / "mysql" / "osclass_craigslist.sql",
    ]
    if all(p.exists() for p in required):
        return

    # Need to download bundle
    zip_out = docker_imgs_path / "classifieds_docker_compose.zip"
    if not zip_out.exists():
        log(f"Downloading classifieds docker-compose bundle: {CLASSIFIEDS_COMPOSE_ZIP_URL}")
        download_file(CLASSIFIEDS_COMPOSE_ZIP_URL, zip_out, env=env)
    else:
        log(f"Found classifieds docker-compose bundle archive: {zip_out}")

    # Ensure we can write into the target directory (or its parent if it doesn't exist yet)
    target_parent = classifieds_compose_dir if classifieds_compose_dir.exists() else classifieds_compose_dir.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    if not os.access(target_parent, os.W_OK):
        raise PermissionError(
            "Classifieds compose directory is not writable; cannot extract bundle.\n"
            f"Path: {target_parent}\n"
            "Fix by changing ownership/permissions, e.g.:\n"
            f"  sudo chown -R $USER:$USER {target_parent}\n"
            f"  sudo chmod -R u+rwX {target_parent}\n"
        )

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(zip_out) as zf:
            zf.extractall(tmp)

        # Find the root folder inside the archive (expected: classifieds_docker_compose/)
        inner = tmp / "classifieds_docker_compose"
        if not inner.exists():
            # Fallback: locate by docker-compose-raw.yml
            matches = list(tmp.rglob("docker-compose-raw.yml"))
            if not matches:
                raise RuntimeError(f"Could not find docker-compose-raw.yml inside {zip_out}")
            inner = matches[0].parent

        log(f"Extracting classifieds compose bundle to: {classifieds_compose_dir}")
        shutil.copytree(inner, classifieds_compose_dir, dirs_exist_ok=True)

    # The upstream bundle may include a deprecated docker-compose.yml; we use docker-compose-raw.yml.
    legacy = classifieds_compose_dir / "docker-compose.yml"
    if legacy.exists() and (classifieds_compose_dir / "docker-compose-raw.yml").exists():
        try:
            legacy.unlink()
            log(f"Removed deprecated file: {legacy}")
        except Exception as e:  # noqa: BLE001
            log(f"Warning: failed to remove deprecated file {legacy}: {e}")

    # Re-check minimal required files
    missing = [p for p in required if not p.exists()]
    if missing:
        log(f"Warning: classifieds compose bundle still missing expected files: {missing}")


# ---------------------------
# Docker helpers
# ---------------------------


def docker_image_exists(image: str, *, env: Dict[str, str]) -> bool:
    cp = run(["docker", "image", "inspect", image], env=env, check=False, capture=True)
    return cp.returncode == 0


def docker_load_tar(tar_path: Path, *, env: Dict[str, str]) -> None:
    run(["docker", "load", "--input", str(tar_path)], env=env)


def docker_pull(image: str, *, env: Dict[str, str]) -> None:
    run(["docker", "pull", image], env=env)


# ---------------------------
# Download helpers
# ---------------------------


def download_file(url: str, out: Path, *, env: Dict[str, str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    aria2c = shutil.which("aria2c")
    curl = shutil.which("curl")
    wget = shutil.which("wget")

    # Prefer aria2c for faster segmented downloads if available.
    # Tunables (env vars):
    #   ARIA2C_CONNECTIONS (default: 16)
    #   ARIA2C_SPLIT (default: 16)
    #   ARIA2C_MIN_SPLIT_SIZE (default: 1M)
    if aria2c:
        connections = env.get("ARIA2C_CONNECTIONS", "16")
        split = env.get("ARIA2C_SPLIT", "16")
        min_split_size = env.get("ARIA2C_MIN_SPLIT_SIZE", "1M")
        log(f"Downloading (aria2c) {url} -> {out} (connections={connections} split={split} min_split_size={min_split_size})")
        run(
            [
                aria2c,
                "--continue=true",
                "--max-tries=5",
                "--retry-wait=2",
                "--check-certificate=true",
                "--file-allocation=none",
                f"--max-connection-per-server={connections}",
                f"--split={split}",
                f"--min-split-size={min_split_size}",
                f"--dir={out.parent.as_posix()}",
                f"--out={out.name}",
                url,
            ],
            env=env,
        )
        return

    if curl:
        # -L follow redirects; --fail for non-2xx; -C - resume; --retry for transient failures.
        log(f"Downloading (curl) {url} -> {out}")
        run(
            [curl, "-L", "--fail", "--retry", "5", "--retry-delay", "2", "-C", "-", "-o", str(out), url],
            env=env,
        )
        return

    if wget:
        log(f"Downloading (wget) {url} -> {out}")
        run([wget, "-c", "-O", str(out), url], env=env)
        return

    raise RuntimeError("Need curl or wget to download files (not found on PATH).")


# ---------------------------
# Site definitions
# ---------------------------


TarSpec = Tuple[str, str, str]  # (docker_image_name, tar_filename, url)


TAR_SPECS: Dict[str, TarSpec] = {
    "shopping": (
        "shopping_final_0712",
        "shopping_final_0712.tar",
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar",
    ),
    # keep both names for convenience/compatibility with start_reset_envs.py
    "admin": (
        "shopping_admin_final_0719",
        "shopping_admin_final_0719.tar",
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar",
    ),
    "shopping_admin": (
        "shopping_admin_final_0719",
        "shopping_admin_final_0719.tar",
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar",
    ),
    "reddit": (
        "postmill-populated-exposed-withimg",
        "postmill-populated-exposed-withimg.tar",
        "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar",
    ),
    "gitlab": (
        "gitlab-populated-final-port8023",
        "gitlab-populated-final-port8023.tar",
        "http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar",
    ),
}


PULL_SPECS: Dict[str, List[str]] = {
    "classifieds": ["jykoh/classifieds:latest", "mysql:8.1"],
}

WIKIPEDIA_ZIM = (
    "wikipedia_en_all_maxi_2022-05.zim",
    "http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim",
)
WIKIPEDIA_IMAGE = "ghcr.io/kiwix/kiwix-serve:3.3.0"


def expand_sites(sites: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for s in sites:
        if s == "all_vwa":
            expanded.extend(["shopping", "reddit", "classifieds", "wikipedia"])
        elif s == "all_wa":
            expanded.extend(["shopping", "admin", "reddit", "gitlab", "wikipedia"])
        else:
            expanded.append(s)

    # de-dupe while preserving order
    out: List[str] = []
    seen = set()
    for s in expanded:
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


# ---------------------------
# Main logic
# ---------------------------


def ensure_tar_image_loaded(site: str, *, docker_imgs_path: Path, env: Dict[str, str]) -> None:
    image, tar_name, url = TAR_SPECS[site]
    tar_path = docker_imgs_path / tar_name

    if docker_image_exists(image, env=env):
        log(f"OK: Docker image already present: {image}")
        return

    if not tar_path.exists():
        log(f"Missing archive: {tar_path}")
        download_file(url, tar_path, env=env)
    else:
        log(f"Found archive: {tar_path}")

    log(f"Loading into Docker: {tar_path}")
    docker_load_tar(tar_path, env=env)

    if docker_image_exists(image, env=env):
        log(f"OK: Loaded image: {image}")
    else:
        log(f"Warning: docker load finished but image '{image}' is still not inspectable.")
        log("         Run: docker images | head -50")


def ensure_pulled_images(site: str, *, env: Dict[str, str]) -> None:
    for image in PULL_SPECS[site]:
        if docker_image_exists(image, env=env):
            log(f"OK: Docker image already present: {image}")
            continue
        log(f"Pulling image: {image}")
        docker_pull(image, env=env)


def ensure_wikipedia(*, docker_imgs_path: Path, env: Dict[str, str]) -> None:
    zim_name, zim_url = WIKIPEDIA_ZIM
    zim_path = docker_imgs_path / zim_name

    if not zim_path.exists():
        log(f"Missing Wikipedia ZIM: {zim_path}")
        download_file(zim_url, zim_path, env=env)
    else:
        log(f"OK: Found Wikipedia ZIM: {zim_path}")

    if docker_image_exists(WIKIPEDIA_IMAGE, env=env):
        log(f"OK: Docker image already present: {WIKIPEDIA_IMAGE}")
    else:
        log(f"Pulling image: {WIKIPEDIA_IMAGE}")
        docker_pull(WIKIPEDIA_IMAGE, env=env)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/load required WebArena/VWebArena docker images (idempotent).")
    p.add_argument(
        "sites",
        nargs="+",
        help="Sites to prepare. Options: shopping, admin, shopping_admin, reddit, gitlab, classifieds, wikipedia, all_vwa, all_wa",
    )
    p.add_argument(
        "--delete-tars",
        action="store_true",
        default=False,
        help="After successfully loading images, delete downloaded .tar archives under DOCKER_IMGS_PATH to reclaim disk space.",
    )
    return p.parse_args(argv)


def maybe_prompt_delete_tars(*, tar_paths: List[Path], assume_yes: bool) -> None:
    # De-dupe and keep only existing files
    uniq: List[Path] = []
    seen = set()
    for p in tar_paths:
        if not p.exists() or not p.is_file():
            continue
        if p in seen:
            continue
        uniq.append(p)
        seen.add(p)

    if not uniq:
        return

    if assume_yes:
        do_delete = True
    else:
        # Only prompt in interactive terminals; otherwise default to "no".
        if not sys.stdin.isatty():
            log("Note: downloaded .tar archives are no longer needed after docker load; keeping them (non-interactive).")
            return

        total_gb = sum(p.stat().st_size for p in uniq) / (1024**3)
        log(
            f"Downloaded .tar archives are no longer needed after 'docker load'.\n"
            f"Found {len(uniq)} archive(s) (~{total_gb:.1f} GB) in DOCKER_IMGS_PATH.\n"
            "Delete them now to reclaim disk space? [y/N] "
        )
        ans = input().strip().lower()
        do_delete = ans in {"y", "yes"}

    if not do_delete:
        return

    for p in uniq:
        try:
            p.unlink()
            log(f"Deleted: {p}")
        except Exception as e:  # noqa: BLE001
            log(f"Warning: failed to delete {p}: {e}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    which_or_raise("docker")

    env_constants = load_env_constants()
    child_env = os.environ.copy()
    child_env.update(env_constants)

    docker_imgs_path = Path(env_constants["DOCKER_IMGS_PATH"])
    ensure_writable_dir(docker_imgs_path)
    classifieds_compose_dir_cfg = Path(env_constants["CLASSIFIEDS_DOCKER_COMPOSE_DIR"])
    classifieds_compose_dir = resolve_classifieds_compose_dir(
        docker_imgs_path=docker_imgs_path,
        configured=classifieds_compose_dir_cfg,
    )

    args = parse_args(argv)
    sites = expand_sites(args.sites)

    valid = set(TAR_SPECS.keys()) | set(PULL_SPECS.keys()) | {"wikipedia", "all_vwa", "all_wa"}
    for s in args.sites:
        if s not in valid:
            raise SystemExit(f"Unknown site '{s}'. Valid: {sorted(valid)}")

    log(f"DOCKER_IMGS_PATH={docker_imgs_path}")

    # Track candidate archives to delete after a successful run (only .tar archives).
    tar_candidates: List[Path] = []

    for site in sites:
        if site in TAR_SPECS:
            # Candidate archive for optional deletion prompt at the end
            _, tar_name, _ = TAR_SPECS[site]
            tar_candidates.append(docker_imgs_path / tar_name)
            ensure_tar_image_loaded(site, docker_imgs_path=docker_imgs_path, env=child_env)
        elif site in PULL_SPECS:
            if site == "classifieds":
                ensure_classifieds_compose_bundle_present(
                    classifieds_compose_dir=classifieds_compose_dir,
                    docker_imgs_path=docker_imgs_path,
                    env=child_env,
                )
                ensure_classifieds_sql_present(classifieds_compose_dir=classifieds_compose_dir)
            ensure_pulled_images(site, env=child_env)
        elif site == "wikipedia":
            ensure_wikipedia(docker_imgs_path=docker_imgs_path, env=child_env)
        else:
            raise SystemExit(f"Unhandled site '{site}' (internal error)")

    maybe_prompt_delete_tars(tar_paths=tar_candidates, assume_yes=args.delete_tars)
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
