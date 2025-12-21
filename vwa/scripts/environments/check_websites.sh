#!/bin/bash


SITE_NAMES=()
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Default values
# Help function
print_usage() {
    echo "Usage: $0 [-p process_id] [-s \"site1 site2 ...\"] [-f url_file]"
    echo "Options:"
    echo "  -p    Process ID - if only process ID is given, checks all sites for that ID"
    echo "  -s    Space-separated list of site names (e.g., \"shopping reddit wikipedia\")"
    echo "  -f    File containing URLs to check (format: 'site-name url' per line)"
    echo "  -h    Show this help message"
    exit 1
}

# Add -f to getopts
while getopts "p:s:f:h" opt; do
  case $opt in
    p)
      PROCESS_ID="$OPTARG"
      ;;
    s)
      # Convert space-separated string to array
      read -ra SITE_NAMES <<< "$OPTARG"
      ;;
    f)
      URLS_FILE="$OPTARG"
      ;;
    h)
      print_usage
      ;;
    *)
      print_usage
      ;;
  esac
done

# If no URLS_FILE provided, use default logic
if [[ -z $URLS_FILE ]]; then
    if [[ -n $PROCESS_ID ]]; then
        URLS_FILE="${URLS_FILE_TEMPLATE//\{PROCESS_ID\}/$PROCESS_ID}"
    else
        URLS_FILE=$(mktemp)
        sort $URLS_DIR/*.txt | uniq > $URLS_FILE
    fi
fi
echo URLs file: $URLS_FILE

# Generate site identifiers based on process ID and site types
SITES_TO_CHECK=()
if [[ -n $PROCESS_ID ]]; then
    if [[ ${#SITE_NAMES[@]} -gt 0 ]]; then
        # If both process ID and site names are provided
        for site_name in "${SITE_NAMES[@]}"; do
            SITES_TO_CHECK+=("$site_name-$PROCESS_ID")
        done
    else
        # If only process ID is provided, we'll filter by ID in the loop
        echo "Checking all sites for process ID: $PROCESS_ID"
    fi
fi

if [[ ${#SITE_NAMES[@]} -gt 0 ]]; then
    echo "Checking sites: ${SITE_NAMES[*]}"
fi

check_url() {
    local url=$1
    # Try to connect with a 5 second timeout
    if curl -s -f --max-time 5 "$url" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

while IFS= read -r line; do
    site=$(echo "$line" | cut -d ' ' -f 1)
    url=$(echo "$line" | cut -d ' ' -f 2)

    # If process ID is provided but no specific sites, filter by process ID
    if [[ -n $PROCESS_ID && ${#SITE_NAMES[@]} -eq 0 ]]; then
        if [[ ! "$site" =~ -$PROCESS_ID$ ]]; then
            continue
        fi
    fi

    # If specific sites were provided, filter by them (with or without process ID)
    if [[ ${#SITE_NAMES[@]} -gt 0 ]]; then
        match=false
        for s in "${SITE_NAMES[@]}"; do
            if [[ -n $PROCESS_ID ]]; then
                # Require exact match with process id
                if [[ "$site" == "$s-$PROCESS_ID" ]]; then
                    match=true
                    break
                fi
            else
                # Allow any process id
                if [[ "$site" == "$s" || "$site" == "$s-"* ]]; then
                    match=true
                    break
                fi
            fi
        done
        if ! $match; then
            continue
        fi
    fi

    if check_url "$url"; then
        printf "%-15s %-10s %s\n" "$site" "UP" "URL: $url"
    else
        printf "%-15s %-10s %s\n" "$site" "DOWN" "URL: $url"
    fi

done < "$URLS_FILE"
echo "Check complete."
