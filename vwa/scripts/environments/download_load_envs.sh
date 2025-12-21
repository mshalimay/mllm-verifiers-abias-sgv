#!/usr/bin/env bash
# Ensure required (Visual)WebArena docker images are present:
# - Downloads missing local .tar image archives (shopping/admin/reddit/gitlab)
# - Loads images into Docker only if not already loaded
# - Pulls remote images for classifieds (jykoh/classifieds, mysql)
# - Downloads Wikipedia `.zim` if missing and pulls kiwix-serve image
#
# Usage:
#   ./scripts/environments/download_load_envs.sh shopping admin reddit gitlab classifieds
#   ./scripts/environments/download_load_envs.sh all_vwa
#   ./scripts/environments/download_load_envs.sh all_wa
#   ./scripts/environments/download_load_envs.sh --delete-tars all_wa
#
# Notes:
# - Uses DOCKER_IMGS_PATH from `benchmark_config.constants` (same as load_docker_imgs.sh)
# - Downloads use the direct links documented in `environment_docker/README.MD`
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "Error: $*" >&2; exit 1; }
log() { echo "[ensure_docker_imgs] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_writable_dir() {
  local dir="$1"
  mkdir -p "$dir"
  if [ ! -w "$dir" ]; then
    die "DOCKER_IMGS_PATH is not writable: $dir
Fix by changing ownership/permissions, e.g.:
  sudo chown -R \$USER:\$USER \"$dir\"
  sudo chmod -R u+rwX \"$dir\""
  fi
}

ensure_classifieds_sql_present() {
  # Ensure `osclass_craigslist.sql` exists next to the docker-compose mysql seed files.
  # If only a zip is present, extract it using Python (no unzip dependency).
  local compose_dir="$1" # CLASSIFIEDS_DOCKER_COMPOSE_DIR
  local mysql_dir="${compose_dir}/mysql"
  local sql_path="${mysql_dir}/osclass_craigslist.sql"
  local zip_path="${mysql_dir}/osclass_craigslist.zip"

  if [ -f "$sql_path" ]; then
    return 0
  fi

  if [ ! -f "$zip_path" ]; then
    log "Warning: classifieds SQL seed missing. Expected $sql_path or $zip_path"
    return 0
  fi

  mkdir -p "$mysql_dir"
  if [ ! -w "$mysql_dir" ]; then
    die "Classifieds seed directory is not writable; cannot extract SQL:
  $mysql_dir
Fix by changing ownership/permissions, e.g.:
  sudo chown -R \$USER:\$USER \"$mysql_dir\"
  sudo chmod -R u+rwX \"$mysql_dir\""
  fi

  log "Extracting classifieds SQL seed: $zip_path -> $sql_path"
  python3 - <<PY
import zipfile, shutil, os, sys
zip_path = ${zip_path@Q}
sql_path = ${sql_path@Q}
with zipfile.ZipFile(zip_path) as zf:
    members = [m for m in zf.namelist() if m.endswith("osclass_craigslist.sql")]
    if not members:
        raise RuntimeError(f"No osclass_craigslist.sql found inside {zip_path}")
    member = members[0]
    with zf.open(member, "r") as src, open(sql_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
print(f"OK: extracted {sql_path}")
PY
}

CLASSIFIEDS_COMPOSE_ZIP_URL="https://archive.org/download/classifieds_docker_compose/classifieds_docker_compose.zip"

resolve_classifieds_compose_dir() {
  # Prefer configured CLASSIFIEDS_DOCKER_COMPOSE_DIR if it exists, otherwise fall back to DOCKER_IMGS_PATH/classifieds_docker_compose
  local configured="$1"
  if [ -d "$configured" ]; then
    echo "$configured"
    return 0
  fi
  echo "$DOCKER_IMGS_PATH/classifieds_docker_compose"
}

ensure_classifieds_compose_bundle_present() {
  local compose_dir="$1"
  local required1="$compose_dir/docker-compose-raw.yml"
  local required2="$compose_dir/mysql/init_db.sh"
  local required3="$compose_dir/mysql/classifieds_restore.sql"
  local required4="$compose_dir/mysql/osclass_craigslist.sql"

  if [ -f "$required1" ] && [ -f "$required2" ] && [ -f "$required3" ] && [ -f "$required4" ]; then
    return 0
  fi

  # Download archive (to DOCKER_IMGS_PATH) if needed
  local zip_out="$DOCKER_IMGS_PATH/classifieds_docker_compose.zip"
  if [ ! -f "$zip_out" ]; then
    log "Downloading classifieds docker-compose bundle: $CLASSIFIEDS_COMPOSE_ZIP_URL"
    download_file "$CLASSIFIEDS_COMPOSE_ZIP_URL" "$zip_out"
  else
    log "Found classifieds docker-compose bundle archive: $zip_out"
  fi

  mkdir -p "$compose_dir"
  if [ ! -w "$compose_dir" ]; then
    die "Classifieds compose directory is not writable; cannot extract bundle:
  $compose_dir
Fix by changing ownership/permissions, e.g.:
  sudo chown -R \$USER:\$USER \"$compose_dir\"
  sudo chmod -R u+rwX \"$compose_dir\""
  fi

  log "Extracting classifieds docker-compose bundle into: $compose_dir"
  python3 - <<PY
import zipfile, tempfile, shutil
from pathlib import Path

zip_path = Path(${zip_out@Q})
target = Path(${compose_dir@Q})

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    inner = tmp / "classifieds_docker_compose"
    if not inner.exists():
        matches = list(tmp.rglob("docker-compose-raw.yml"))
        if not matches:
            raise RuntimeError(f"Could not find docker-compose-raw.yml inside {zip_path}")
        inner = matches[0].parent
    shutil.copytree(inner, target, dirs_exist_ok=True)
print("OK: extracted classifieds compose bundle")
PY

  # The upstream bundle may include a deprecated docker-compose.yml; we use docker-compose-raw.yml.
  if [ -f "$compose_dir/docker-compose.yml" ] && [ -f "$compose_dir/docker-compose-raw.yml" ]; then
    rm -f "$compose_dir/docker-compose.yml" || true
    log "Removed deprecated file: $compose_dir/docker-compose.yml"
  fi
}

download_file() {
  local url="$1"
  local out="$2"

  mkdir -p "$(dirname "$out")"

  # Prefer aria2c for faster segmented downloads if available.
  # Tunables (env vars):
  #   ARIA2C_CONNECTIONS (default: 16)
  #   ARIA2C_SPLIT (default: 16)
  #   ARIA2C_MIN_SPLIT_SIZE (default: 1M)
  if command -v aria2c >/dev/null 2>&1; then
    local connections="${ARIA2C_CONNECTIONS:-16}"
    local split="${ARIA2C_SPLIT:-16}"
    local min_split_size="${ARIA2C_MIN_SPLIT_SIZE:-1M}"
    log "Downloading (aria2c) $url -> $out (connections=$connections split=$split min_split_size=$min_split_size)"
    aria2c \
      --continue=true \
      --max-tries=5 \
      --retry-wait=2 \
      --check-certificate=true \
      --file-allocation=none \
      --max-connection-per-server="$connections" \
      --split="$split" \
      --min-split-size="$min_split_size" \
      --dir="$(dirname "$out")" \
      --out="$(basename "$out")" \
      "$url"
  elif command -v curl >/dev/null 2>&1; then
    # -L follow redirects; --fail for non-2xx; -C - resume; --retry for transient failures.
    log "Downloading (curl) $url -> $out"
    curl -L --fail --retry 5 --retry-delay 2 -C - -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    log "Downloading (wget) $url -> $out"
    wget -c -O "$out" "$url"
  else
    die "Need curl or wget to download files"
  fi
}

docker_image_exists() {
  local image="$1"
  docker image inspect "$image" >/dev/null 2>&1
}

ensure_tar_image_loaded() {
  local image="$1"     # e.g. shopping_final_0712
  local tar_path="$2"  # e.g. $DOCKER_IMGS_PATH/shopping_final_0712.tar
  local url="$3"

  if docker_image_exists "$image"; then
    log "OK: Docker image already present: $image"
    return 0
  fi

  if [ ! -f "$tar_path" ]; then
    log "Missing archive: $tar_path"
    download_file "$url" "$tar_path"
  else
    log "Found archive: $tar_path"
  fi

  log "Loading into Docker: $tar_path"
  docker load --input "$tar_path"

  if docker_image_exists "$image"; then
    log "OK: Loaded image: $image"
  else
    log "Warning: docker load finished but image '$image' is still not inspectable."
    log "         Run: docker images | head -50"
  fi
}

ensure_pulled_image() {
  local image="$1" # e.g. mysql:8.1
  if docker_image_exists "$image"; then
    log "OK: Docker image already present: $image"
    return 0
  fi
  log "Pulling image: $image"
  docker pull "$image"
}

ensure_file_present() {
  local out="$1"
  local url="$2"
  if [ -f "$out" ]; then
    log "OK: Found file: $out"
    return 0
  fi
  log "Missing file: $out"
  download_file "$url" "$out"
}

#===============================================================================
# Load environment variables from Python constants (script-relative)
# (kept consistent with scripts/environments/load_docker_imgs.sh)
#===============================================================================
find_repo_root() {
  local start_dir="$1"
  local marker_dir="$2"
  local current_dir="$start_dir"
  while [ "$current_dir" != "/" ]; do
    if [ -d "$current_dir/$marker_dir" ]; then
      echo "$current_dir"
      return 0
    fi
    current_dir="$(dirname "$current_dir")"
  done
  return 1
}

VWA_ROOT="$(find_repo_root "$SCRIPT_DIR" "benchmark_config" || true)"
if [ -z "${VWA_ROOT:-}" ]; then
  log "Warning: Could not find benchmark_config directory. Falling back to relative path."
  VWA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

eval "$(
  PYTHONPATH="${VWA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from benchmark_config.export_to_shell import export_env_vars
for k, v in export_env_vars().items():
    v_str = str(v).replace('"', '\\"')
    print(f'export {k}="{v_str}"')
PY
)"
#===============================================================================

require_cmd docker

delete_tars=false
raw_args=("$@")
sites=()
for a in "${raw_args[@]}"; do
  case "$a" in
    --delete-tars)
      delete_tars=true
      ;;
    -h|--help)
      # handled below
      sites+=("$a")
      ;;
    *)
      sites+=("$a")
      ;;
  esac
done

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ ${#sites[@]} -eq 0 ]; then
  cat <<'USAGE'
Usage:
  ./scripts/environments/download_load_envs.sh [--delete-tars] <site_name...>

Where site_name in:
  shopping admin reddit gitlab classifieds wikipedia all_vwa all_wa

Examples:
  ./scripts/environments/download_load_envs.sh all_vwa
  ./scripts/environments/download_load_envs.sh shopping reddit
  ./scripts/environments/download_load_envs.sh --delete-tars all_wa

This script uses DOCKER_IMGS_PATH from benchmark_config and will download large archives
for some sites (tens of GB). Make sure you have sufficient disk space.
USAGE
  exit 0
fi

expanded_sites=()
for s in "${sites[@]}"; do
  case "$s" in
    all_vwa)
      expanded_sites+=(shopping reddit classifieds wikipedia)
      ;;
    all_wa)
      expanded_sites+=(shopping admin reddit gitlab wikipedia)
      ;;
    shopping|admin|reddit|gitlab|classifieds|wikipedia)
      expanded_sites+=("$s")
      ;;
    *)
      die "Unknown site '$s'. Use --help for valid options."
      ;;
  esac
done

# De-duplicate while preserving order
unique_sites=()
for s in "${expanded_sites[@]}"; do
  skip=0
  for u in "${unique_sites[@]}"; do
    if [ "$u" = "$s" ]; then skip=1; break; fi
  done
  if [ "$skip" -eq 0 ]; then unique_sites+=("$s"); fi
done

log "DOCKER_IMGS_PATH=$DOCKER_IMGS_PATH"
ensure_writable_dir "$DOCKER_IMGS_PATH"

tar_candidates=()

for site in "${unique_sites[@]}"; do
  case "$site" in
    shopping)
      tar_candidates+=("$DOCKER_IMGS_PATH/shopping_final_0712.tar")
      ensure_tar_image_loaded \
        "shopping_final_0712" \
        "$DOCKER_IMGS_PATH/shopping_final_0712.tar" \
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar"
      ;;
    admin)
      tar_candidates+=("$DOCKER_IMGS_PATH/shopping_admin_final_0719.tar")
      ensure_tar_image_loaded \
        "shopping_admin_final_0719" \
        "$DOCKER_IMGS_PATH/shopping_admin_final_0719.tar" \
        "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar"
      ;;
    reddit)
      tar_candidates+=("$DOCKER_IMGS_PATH/postmill-populated-exposed-withimg.tar")
      ensure_tar_image_loaded \
        "postmill-populated-exposed-withimg" \
        "$DOCKER_IMGS_PATH/postmill-populated-exposed-withimg.tar" \
        "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar"
      ;;
    gitlab)
      tar_candidates+=("$DOCKER_IMGS_PATH/gitlab-populated-final-port8023.tar")
      ensure_tar_image_loaded \
        "gitlab-populated-final-port8023" \
        "$DOCKER_IMGS_PATH/gitlab-populated-final-port8023.tar" \
        "http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar"
      ;;
    classifieds)
      # Some setups keep the compose bundle under DOCKER_IMGS_PATH; fall back if configured dir doesn't exist.
      CLASSIFIEDS_DOCKER_COMPOSE_DIR="$(resolve_classifieds_compose_dir "${CLASSIFIEDS_DOCKER_COMPOSE_DIR:-}")"
      ensure_classifieds_compose_bundle_present "$CLASSIFIEDS_DOCKER_COMPOSE_DIR"
      ensure_classifieds_sql_present "$CLASSIFIEDS_DOCKER_COMPOSE_DIR"
      ensure_pulled_image "jykoh/classifieds:latest"
      ensure_pulled_image "mysql:8.1"
      ;;
    wikipedia)
      ensure_file_present \
        "$DOCKER_IMGS_PATH/wikipedia_en_all_maxi_2022-05.zim" \
        "http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim"
      ensure_pulled_image "ghcr.io/kiwix/kiwix-serve:3.3.0"
      ;;
  esac
done

existing_tars=()
for p in "${tar_candidates[@]}"; do
  if [ -f "$p" ]; then
    existing_tars+=("$p")
  fi
done

if [ ${#existing_tars[@]} -gt 0 ]; then
  if $delete_tars; then
    log "Deleting downloaded .tar archives (requested via --delete-tars)..."
    rm -f "${existing_tars[@]}" || true
  else
    # Only prompt on interactive terminals; otherwise keep the files.
    if [ -t 0 ]; then
      echo "[ensure_docker_imgs] Downloaded .tar archives are no longer needed after 'docker load'."
      echo -n "[ensure_docker_imgs] Delete them now to reclaim disk space? [y/N] "
      read -r ans || ans=""
      case "${ans,,}" in
        y|yes)
          rm -f "${existing_tars[@]}" || true
          log "Deleted .tar archives."
          ;;
        *)
          log "Keeping .tar archives."
          ;;
      esac
    else
      log "Note: downloaded .tar archives are no longer needed after docker load; keeping them (non-interactive)."
    fi
  fi
fi

log "Done."


