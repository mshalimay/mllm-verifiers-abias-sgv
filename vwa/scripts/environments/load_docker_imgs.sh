#!/bin/bash
# This script loads the environment images into Docker. Needs to run only once for each image.

# Function to check if a site is in the command input list
is_site_in_list() {
  local site="$1"
  for s in "${sites[@]}"; do
    if [[ "$s" == "$site" ]]; then
      return 0
    fi
  done
  return 1
}


#===============================================================================
# Load environment variables from Python constants (script-relative)
#===============================================================================

# Find repository root by searching upward for a marker directory
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
# Search upward for benchmark_config directory
VWA_ROOT=$(find_repo_root "$SCRIPT_DIR" "benchmark_config")
# Fallback if not found
if [ -z "$VWA_ROOT" ]; then
  echo "Warning: Could not find benchmark_config directory. Falling back to relative path."
  VWA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

# Use Python to emit export commands; set PYTHONPATH so benchmark_config can be imported reliably.
eval "$(
  PYTHONPATH="${VWA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from benchmark_config.export_to_shell import export_env_vars

for k, v in export_env_vars().items():
    # Safely escape any double quotes in values for bash export
    v_str = str(v).replace('"', '\\"')
    print(f'export {k}="{v_str}"')
PY
)"
#===============================================================================


sites=("$@")  # sites=('shopping' 'admin' 'reddit' 'gitlab' 'classifieds')
# Usage: scripts/load_docker_images.sh shopping admin reddit gitlab

# Load Image for Shopping Website (OneStopShop)  [WebArena & VWebArena]
if is_site_in_list "shopping"; then
    docker load --input "$DOCKER_IMGS_PATH/shopping_final_0712.tar"
fi

# Load Image for Shopping Admin Website [WebArena]
if is_site_in_list "admin"; then
    docker load --input "$DOCKER_IMGS_PATH/shopping_admin_final_0719.tar"
fi

# Load Social Forum Website (Reddit) [WebArena & VWebArena]
if is_site_in_list "reddit"; then
    docker load --input "$DOCKER_IMGS_PATH/postmill-populated-exposed-withimg.tar"
fi

# Gitlab Website [WebArena]
if is_site_in_list "gitlab"; then
    docker load --input "$DOCKER_IMGS_PATH/gitlab-populated-final-port8023.tar"
fi

# Classifieds [VWebArena] 
# Can take long time if running for the first time; it will download image before loading into Docker.
if is_site_in_list "classifieds"; then
    docker pull jykoh/classifieds:latest
    docker pull mysql:8.1
fi
