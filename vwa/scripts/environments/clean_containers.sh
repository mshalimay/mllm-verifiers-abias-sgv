#!/bin/bash

# Clean up all Docker containers and homepage Flask processes for the given instance IDs.
# Usage:
#   ./clean_containers.sh <instance_id1> [instance_id2 ...]
#   ./clean_containers.sh            # cleans ALL discovered instance IDs (containers + homepage Flask)
#   ./clean_containers.sh --force    # same as above, but skips confirmation

set -o pipefail

show_usage() {
  echo "Usage: $0 <instance_id1> [instance_id2 ...]"
  echo "  Deletes Docker containers whose names end with -<ID> or _<ID>."
  echo "  Also kills homepage Flask processes started as: app.py <ID> --port <PORT>."
  echo ""
  echo "If no IDs are provided, the script will discover IDs from Docker container names"
  echo "and running homepage Flask processes and clean ALL of them."
  echo "Use --force to skip the confirmation prompt."
  exit 1
}

is_numeric() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

extract_ids_from_container_names() {
  # Extract numeric tokens that appear as -<ID>- / _<ID>_ / at start/end.
  # We intentionally keep this conservative: only tokens separated by '-' or '_' are considered IDs.
  docker ps -a --format '{{.Names}}' 2>/dev/null \
    | sed -E 's/[^0-9_-]/_/g' \
    | grep -Eo '(^|[-_])[0-9]+([-_]|$)' \
    | grep -Eo '[0-9]+' \
    | sort -u || true
}

extract_ids_from_homepage_flask_processes() {
  # Matches processes containing: app.py <ID> --port
  # Use ps (portable) instead of relying on pgrep output formatting.
  ps -eo args 2>/dev/null \
    | grep -E 'app\.py [0-9]+ --port' \
    | grep -Eo 'app\.py [0-9]+' \
    | awk '{print $2}' \
    | sort -u || true
}

discover_all_ids() {
  {
    extract_ids_from_container_names
    extract_ids_from_homepage_flask_processes
  } | sort -u || true
}

confirm_cleanup_all() {
  local ids="$1"
  echo "No instance IDs provided. I found the following IDs to clean:"
  printf '%s\n' "$ids" | sed 's/^/  - /'
  echo ""
  echo "This will REMOVE Docker containers (and their named volumes) and KILL homepage Flask processes for ALL listed IDs."

  local reply
  read -r -p "Proceed? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]]
}

kill_flask_processes_for_id() {
  local id="$1"
  local pattern="app\\.py ${id} --port"
  # Find and kill matching Flask processes if any
  local pids
  pids=$(pgrep -f "$pattern" || true)
  if [[ -n "$pids" ]]; then
    echo "Killing homepage Flask processes for ID $id: $pids"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    # Give them a moment to exit, then force kill if needed
    sleep 0.5
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      echo "Forcing kill for remaining Flask processes for ID $id"
      pkill -9 -f "$pattern" 2>/dev/null || true
    fi
  else
    echo "No homepage Flask processes found for ID $id"
  fi
}

remove_containers_for_id() {
  local id="$1"

  # Match names that have -ID- or _ID- (or start/end with it), not just at the end.
  local containers
  containers=$(docker ps -a --format '{{.Names}}' | grep -E '(^|[-_])'"${id}"'([-_]|$)' || true)

  if [[ -z "$containers" ]]; then
    echo "No Docker containers found for ID $id"
  else
    echo "Removing Docker containers for ID $id:"
    while IFS= read -r name; do
      [[ -z "$name" ]] && continue
      echo "  - $name"

      # Get volumes used by this container before removing it
      local volumes
      volumes=$(docker inspect "$name" --format '{{range .Mounts}}{{.Name}}{{println}}{{end}}' 2>/dev/null || true)

      # Remove the container
      docker rm -f "$name" || true

      # Remove associated volumes if any
      if [[ -n "$volumes" ]]; then
        echo "    Removing volumes:"
        while IFS= read -r volume; do
          [[ -z "$volume" ]] && continue
          echo "      - $volume"
          docker volume rm "$volume" 2>/dev/null || echo "    Warning: Could not remove volume $volume"
        done <<< "$volumes"
      fi

    done <<< "$containers"
  fi
}

main() {
  local force=false

  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_usage
  fi

  if [[ "${1:-}" == "--force" ]]; then
    force=true
    shift
  fi

  # If no IDs, clean all discovered IDs.
  if [[ $# -eq 0 ]]; then
    local ids
    ids=$(discover_all_ids)
    if [[ -z "$ids" ]]; then
      echo "No instance IDs discovered from Docker containers or homepage Flask processes. Nothing to clean."
      echo "Pruning unused networks..."
      docker network prune -f
      echo "Cleanup complete."
      exit 0
    fi

    if [[ "$force" != true ]]; then
      if ! confirm_cleanup_all "$ids"; then
        echo "Aborted."
        exit 1
      fi
    fi

    # Convert newline-separated IDs into positional parameters (bash-compatible).
    local -a _ids
    readarray -t _ids <<< "$ids"
    set -- "${_ids[@]}"
  fi

  for id in "$@"; do
    if ! is_numeric "$id"; then
      echo "Skipping invalid instance ID: $id (must be numeric)" >&2
      continue
    fi

    printf '\n=== Cleaning resources for instance ID: %s ===\n' "$id"
    remove_containers_for_id "$id"
    kill_flask_processes_for_id "$id"
  done

  # Prune unused networks
  echo "Pruning unused networks..."
  docker network prune -f

  printf '\nCleanup complete.\n'
}

main "$@"
