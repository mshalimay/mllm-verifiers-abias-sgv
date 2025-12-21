#!/bin/bash
# Script to start/reset all websites. Assumes images are loaded into Docker (see `load_docker_imgs.sh`).
# For Docker-based environments, containers are force-removed and re-initialized.

# Echo time
echo "$(date) Starting/resetting websites..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#===============================================================================
# Load environment variables from Python constants (script-relative)
#===============================================================================

# Resolve the VisualWebArena repo root by searching upward for benchmark_config directory
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
echo VWA_ROOT: $VWA_ROOT

# Get script used to check website statuses
CHECK_WEBSITES_SCRIPT=""
if [ -x "${SCRIPT_DIR}/check_websites.sh" ]; then
  CHECK_WEBSITES_SCRIPT="${SCRIPT_DIR}/check_websites.sh"
elif [ -r "${VWA_ROOT}/scripts/environments/check_websites.sh" ]; then
  CHECK_WEBSITES_SCRIPT="${VWA_ROOT}/scripts/environments/check_websites.sh"
else
  echo "Warning: check_websites.sh not found in expected locations."
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

show_usage() {
  echo "Usage: $0 [-p_id <ID>] <site1> <site2> ..."
  echo "  -p: The ID to assign to the docker instances. If not provided, the script will create a new ID."
  echo "  <site1>, <site2>, ...: List of websites to start/reset"
  echo "    Options: all_vwa, all_wa, shopping, "shopping_admin", reddit, gitlab, classifieds, wikipedia, homepagewa, homepagevwa"
  echo "    Choose 'all_vwa' to start/reset all websites for VisualWebArena"
  exit 1
}

# Pick a random starting port in the range 20000-40000
ID_SEPARATOR="-"
MIN_PORT=10000
MAX_PORT=20000
CURRENT_PORT=$((RANDOM % (MAX_PORT - MIN_PORT + 1) + MIN_PORT))
LET_OS_DECIDE_PORT=true
# Array to store the ports for each website instance.
temp_url_file=$(mktemp)

#===============================================================================
# Auxiliary functions
#===============================================================================

# Helper: Wait for Docker to assign a mapped host port for a container
get_docker_mapped_port() {
  local container_name="$1"
  local max_wait="${3:-10}"
  local port=""
  local waited=0
  local sleep_seconds=1
  while (( $(echo "$waited < $max_wait" | bc -l) )); do
    port=$(docker port "$container_name" 2>/dev/null | head -n1 | awk -F: '{print $2}' | tr -d ' ')
    if [[ -n "$port" ]]; then
      echo "$port"
      return 0
    fi
    sleep $sleep_seconds
    waited=$(echo "$waited + $sleep_seconds" | bc)
  done
  return 1
}


# Check if a site is in the list of input sites.
is_site_in_list() {
  local site="$1"
  for s in "${sites[@]}"; do
    if [[ "$s" == "$site" ]]; then
      return 0
    fi
  done
  return 1
}

# Check if a Docker container exists.
does_container_exist() {
  local container_name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qw "$container_name"; then
    return 0
  fi
  return 1
}

remove_container_and_volumes() {
  local container_name="$1"
  local keep="$2"

  if does_container_exist "$container_name"; then
    echo "Removing container $container_name..."

    # Get all volumes attached to this container (ignore bind mounts)
    local volumes
    volumes=$(docker inspect -f '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}} {{end}}{{end}}' "$container_name")

    # Remove the container
    docker rm -f "$container_name"

    # Remove the volumes unless "keep" was specified
    if [ "$keep" != "keep" ]; then
      for v in $volumes; do
        if [ -n "$v" ]; then
          echo "Removing associated volume $v..."
          docker volume rm -f "$v"
        fi
      done
    else
      echo "Keeping volumes for $container_name."
    fi
  fi
}

# Find the next port that is not being used.
get_free_port() {
  local attempts=0
  local max_attempts=100
  local port
  while [ $attempts -lt $max_attempts ]; do
    port=$((RANDOM % (MAX_PORT - MIN_PORT + 1) + MIN_PORT))
    if ! ss -ltn | awk '{print $4}' | grep -q ":$port$"; then
      CURRENT_PORT=$((port+1))
      echo "$port"
      return
    fi
    attempts=$((attempts+1))
  done
  # Fallback: increment from CURRENT_PORT
  port=$CURRENT_PORT
  while ss -ltn | awk '{print $4}' | grep -q ":$port$"; do
    port=$((port+1))
  done
  CURRENT_PORT=$((port+1))
  echo "$port"
}

# Find the next available instance ID when one is not provided.
find_available_id() {
  local id=1
  while true; do
    # If any Docker container name ends with -<id> or _<id>, consider the ID used.
    if docker ps -a --format '{{.Names}}' | grep -qE '[-_]?'"$id"'$'; then
    # Check if container is running, if not, we use this id; else, create one
      id=$((id+1))
    else
      echo "$id"
      return
    fi
  done
}

# Wait for MySQL to be ready inside a container
wait_for_mysql() {
  local container_name="$1"
  local max_wait="${2:-60}"
  local waited=0
  local sleep_seconds=2
  
  echo "Waiting for MySQL to be ready in $container_name..."
  while (( $(echo "$waited < $max_wait" | bc -l) )); do
    # Try to connect to MySQL and run a simple query
    if docker exec "$container_name" mysql -u magentouser -pMyPassword -e "SELECT 1;" > /dev/null 2>&1; then
      echo "MySQL is ready in $container_name."
      return 0
    fi
    echo "MySQL not ready yet, waiting..."
    sleep $sleep_seconds
    waited=$(echo "$waited + $sleep_seconds" | bc)
  done
  
  echo "Warning: MySQL in $container_name did not become ready within $max_wait seconds."
  return 1
}

# Disable re-indexing of products on the shopping website.
disable_shopping_indexing() {
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_product
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_rule
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalogsearch_fulltext
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalog_category_product
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule customer_grid
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule design_config_grid
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule inventory
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_category
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_attribute
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_price
  docker exec "$1" /var/www/magento2/bin/magento indexer:set-mode schedule cataloginventory_stock
}


# Update the classifieds `docker-compose.yml` file for a given instance.
update_classifieds_docker_compose() {
  local base_url="$1"
  local port="$2"
  local instance_id="$3"
  local template_file="$4"
  local output_file="$5"

  sed -e "s|<classifieds_port>|$port|g" \
      -e "s|<your-server-hostname>|$base_url|g" \
      -e "s|<classifieds_reset_token>|$CLASSIFIEDS_RESET_TOKEN|g" \
      -e "s|container_name:[[:space:]]*classifieds_db|container_name: classifieds_db-${instance_id}|g" \
      -e "s|container_name:[[:space:]]*classifieds$|container_name: classifieds-${instance_id}|g" \
      -e "s|\bdb_data\b|db_data-${instance_id}|g" "$template_file" > "$output_file"
}

kill_homepage_processes() {
  local homepage_path="$1"
  local process_id="$2"

  # Get all running app.py processes
  pgrep -f "app.py" | while read -r pid; do
    # Determine the port this process is listening on
    cmd=$(ps -p "$pid" -o command=)
    port=""
    if [[ $cmd =~ --port[[:space:]]+([0-9]+) ]]; then
      port="${BASH_REMATCH[1]}"
    fi

    # Fallback: try to infer port from socket listing
    if [[ -z "$port" ]]; then
      port=$(ss -ltnp 2>/dev/null | awk -v pid="$pid" '$0 ~ ("pid=" pid ",") {print $4}' | sed -E 's/.*:([0-9]+)$/\1/' | head -n 1)
    fi

    if [[ -z "$port" ]]; then
      echo "PID $pid: could not determine port; skipping"
      continue
    fi

    # Query the /process_id endpoint to retrieve the instance id
    instance_id=$(curl -sS --max-time 1 "http://127.0.0.1:$port/process_id" || true)

    if [[ "$instance_id" == "$process_id" ]]; then
      echo "Killing homepage Flask process with ID $instance_id, PID $pid on port $port..."
      kill -9 "$pid" || true
    else
      echo "Skipping PID $pid on port $port (instance_id=$instance_id)"
    fi
  done
}

get_url_site_by_id() {
  local site="$1"
  local target_id="$2"
  local container_name="${site}-${target_id}"
  local port=$(get_docker_mapped_port "$container_name" 10)
  if [[ -n "$port" ]]; then
    echo "http://$BASE_URL:$port"
  fi
}

get_port_site_by_id() {
  local site="$1"
  local target_id="$2"
  local container_name="${site}-${target_id}"
  get_docker_mapped_port "$container_name" 10
}

# Execute the check_websites.sh script
check_websites() {
  local target_id="$1"
  local temp_urls_file="$2"
  local temp_status_file="$3"
  
  if [ -n "$CHECK_WEBSITES_SCRIPT" ] && [ -r "$CHECK_WEBSITES_SCRIPT" ]; then
    bash "$CHECK_WEBSITES_SCRIPT" -p "$target_id" -f "$temp_urls_file" > "$temp_status_file"
    return 0
  else
    echo "Error: check_websites.sh not found" > "$temp_status_file"
    return 1
  fi
}

write_all_instance_urls() {
  target_id="$1"
  urls_file="$2"
  temp_urls_file=$(mktemp)

  # Create url_file dir if it doesn't exist
  mkdir -p "$(dirname "$urls_file")"

  # Get all running containers with their ports
  docker ps -a --format '{{.Names}} {{.Ports}}' | while read -r line; do
    if [[ $line =~ ^(reddit-[0-9]+|shopping-[0-9]+|"shopping_admin"-[0-9]+|gitlab-[0-9]+|classifieds-[0-9]+|wikipedia-[0-9]+|homepage-[0-9]+) ]]; then
      container_name=${BASH_REMATCH[1]}
      # If container name does not end with instance_id, skip
      if [[ ! "$container_name" =~ -$target_id$ ]]; then
        continue
      fi

      # Extract the host port from the Ports column
      if [[ $line =~ :([0-9]+)- ]]; then
        port=${BASH_REMATCH[1]}
        echo "$container_name http://$BASE_URL:$port" >> "$temp_urls_file"
      fi
    fi
  done
  
  # Also check for Flask homepage processes
  pgrep -f "app.py [0-9]+ --port" | while read -r pid; do
    if [[ $(ps -p $pid -o command=) =~ app\.py\ ([0-9]+)\ --port\ ([0-9]+) ]]; then
      instance_id=${BASH_REMATCH[1]}
      if [[ $instance_id != $target_id ]]; then
        continue
      fi
      port=${BASH_REMATCH[2]}
      echo "homepage-$instance_id http://$BASE_URL:$port" >> "$temp_urls_file"
    fi
  done

  # Check which sites are up
  temp_status_file=$(mktemp)
  temp_up_urls_file=$(mktemp)

  # Find and run check_websites.sh
  check_websites "$target_id" "$temp_urls_file" "$temp_status_file"

  while IFS= read -r line; do
    if [[ "$line" == *" UP "* ]]; then
      # Extract site and URL
      site=$(echo "$line" | awk '{print $1}')
      url=$(echo "$line" | awk '{print $4}')
      echo "$site $url" >> "$temp_up_urls_file"
    fi
  done < "$temp_status_file"

  mv "$temp_up_urls_file" "$urls_file"
}



#===============================================================================
# Parse command-line arguments
#===============================================================================
process_id=""

# Parse command-line options
while getopts "p:" opt; do
  case $opt in
    p)
      process_id="$OPTARG"
      ;;
    *)
      show_usage
      ;;
  esac
done

# Shift to remove processed options, leaving only site arguments
shift $((OPTIND-1))

# Automatically determine an instance ID if none was supplied.
if [ -z "$process_id" ]; then
  process_id=$(find_available_id)
  echo "No instance ID provided. Using next available ID: $process_id"
fi

# # Prune unused networks
# echo "Pruning unused networks..."
# docker network prune -f

sites=("$@")
if [ ${#sites[@]} -eq 0 ]; then
  show_usage
fi
# If 'all_vwa' is in the list, start/reset all VisualWebArena websites.
if [[ " ${sites[@]} " =~ " all_vwa " ]]; then
  sites=("shopping" "classifieds" "reddit" "wikipedia" "homepagevwa")
  echo "Starting/resetting all VisualWebArena websites: ${sites[@]}."
fi
# If 'all_wa' is in the list, start/reset all WebArena websites.
if [[ " ${sites[@]} " =~ " all_wa " ]]; then
  sites=("shopping" ""shopping_admin"" "reddit" "gitlab" "homepagewa")
  echo "Starting/resetting all WebArena websites: ${sites[@]}."
fi
# If both 'homepagewa' and 'homepagevwa' are in the list, show an error message.
if [[ " ${sites[@]} " =~ " homepagewa " ]] && [[ " ${sites[@]} " =~ " homepagevwa " ]]; then
  echo "Error: both 'homepagewa' and 'homepagevwa' cannot be specified together."
  exit 1
fi

#===============================================================================
# Launch websites
#===============================================================================

#------------------------------
# Launch social forum website (Reddit) [WebArena & VisualWebArena]
#------------------------------
if is_site_in_list "reddit"; then
    container_name="reddit-$process_id"
    remove_container_and_volumes "$container_name"
    if $LET_OS_DECIDE_PORT; then
        docker run --name "$container_name" -p :80 -d postmill-populated-exposed-withimg
        port=$(get_docker_mapped_port "$container_name" 10)
        echo "Reddit port: $port"
    else
        port=$(get_free_port)
        docker run --name "$container_name" -p "$port":80 -d postmill-populated-exposed-withimg
    fi

    echo "Started Reddit instance $process_id at http://$BASE_URL:$port."
    echo "reddit-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#------------------------------
# Launch shopping website (OneStopShop) [WebArena & VisualWebArena]
#------------------------------
if is_site_in_list "shopping"; then
  
  container_name="shopping-$process_id"
  remove_container_and_volumes "$container_name"

  if $LET_OS_DECIDE_PORT; then
    docker run --name "$container_name" -p :80 -d shopping_final_0712
    port=$(get_docker_mapped_port "$container_name" 10)
  else
    port=$(get_free_port)
    docker run --name "$container_name" -p "$port":80 -d shopping_final_0712
  fi

  echo "Started shopping instance $process_id at http://$BASE_URL:$port."

  # Wait for MySQL to be ready before running Magento commands
  container_name="shopping-$process_id"
  if wait_for_mysql "$container_name" 60; then
    echo "Configuring shopping instance $process_id..."
    docker exec "$container_name" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$BASE_URL:$port"
    docker exec "$container_name" mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://$BASE_URL:$port/' WHERE path = 'web/secure/base_url';"
    docker exec "$container_name" /var/www/magento2/bin/magento cache:flush
    disable_shopping_indexing "$container_name"
  else
    echo "Error: Failed to configure shopping instance $process_id due to MySQL connection issues."
  fi
  echo "shopping-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#------------------------------
# Launch shopping "shopping_admin" website (e-commerce content management system (CMS)) [WebArena]
#------------------------------
if is_site_in_list ""shopping_admin""; then
    container_name=""shopping_admin"-$process_id"
    remove_container_and_volumes "$container_name"
    if $LET_OS_DECIDE_PORT; then
      docker run --name "$container_name" -p :80 -d shopping_admin_final_0719
      port=$(get_docker_mapped_port "$container_name" 10)
    else
      port=$(get_free_port)
      docker run --name "$container_name" -p "$port":80 -d shopping_admin_final_0719
    fi

    # Wait for MySQL to be ready before running Magento commands
    if wait_for_mysql "$container_name" 60; then
      docker exec "$container_name" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$BASE_URL:$port"
      docker exec "$container_name" mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://$BASE_URL:$port/' WHERE path = 'web/secure/base_url';"
      docker exec "$container_name" /var/www/magento2/bin/magento cache:flush
      echo "Started shopping_admin instance $process_id at http://$BASE_URL:$port."
    else
      echo "Error: Failed to configure shopping_admin instance $process_id due to MySQL connection issues."
      echo "Started shopping_admin instance $process_id at http://$BASE_URL:$port (configuration may be incomplete)."
    fi
    echo "shopping_admin-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#------------------------------
# Launch GitLab Website [WebArena]
#------------------------------
if is_site_in_list "gitlab"; then
    container_name="gitlab-$process_id"
    remove_container_and_volumes "$container_name"
    if $LET_OS_DECIDE_PORT; then
      docker run --name "$container_name" -d -p :80 gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
      port=$(get_docker_mapped_port "$container_name" 10)
    else
      port=$(get_free_port)
      docker run --name "$container_name" -d -p "$port":"$port" gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
    fi

    docker exec "$container_name" sed -i "s|^external_url.*|external_url 'http://$BASE_URL:$port'|" /etc/gitlab/gitlab.rb
    docker exec "$container_name" gitlab-ctl reconfigure
    echo "Started GitLab instance $process_id at http://$BASE_URL:$port."
    echo "gitlab-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#------------------------------
# Launch classifieds website [VisualWebArena]
#------------------------------
if is_site_in_list "classifieds"; then
    container_name="classifieds-$process_id"
    db_container_name="classifieds_db-$process_id"
    remove_container_and_volumes "$container_name"
    remove_container_and_volumes "$db_container_name"

    port=$(get_free_port) #TODO: Let OS decide port for classifieds requires more tweaking
    update_classifieds_docker_compose "$BASE_URL" "$port" "$process_id" \
      "$CLASSIFIEDS_DOCKER_COMPOSE_DIR/docker-compose-raw.yml" \
      "$CLASSIFIEDS_DOCKER_COMPOSE_DIR/docker-compose-$process_id.yml"

    # Bring up both services; 'web' waits on DB health via compose condition
    echo "Starting $db_container_name..."
    docker compose -p classifieds-"$process_id" -f "$CLASSIFIEDS_DOCKER_COMPOSE_DIR/docker-compose-$process_id.yml" up -d db
    
    # Wait for DB to report healthy
    echo "Waiting for $db_container_name to become healthy..."
    for i in {1..120}; do
      health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' "$db_container_name" 2>/dev/null || echo "unknown")
      if [ "$health" = "healthy" ]; then
        echo "$db_container_name is healthy."
        break
      fi
      sleep 5
    done
    # Populate classifieds database
    # echo "Populating classifieds database..."
    docker exec "$db_container_name" mysql -u root -ppassword osclass -e 'source /docker-entrypoint-initdb.d/osclass_craigslist.sql'
    
    echo "Starting $container_name..."
    docker compose -p classifieds-"$process_id" -f "$CLASSIFIEDS_DOCKER_COMPOSE_DIR/docker-compose-$process_id.yml" up -d web --wait

    echo "Started classifieds instance $process_id at http://$BASE_URL:$port."
    echo "classifieds-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#------------------------------
# Launch Wikipedia website [WebArena & VisualWebArena]
#------------------------------
if is_site_in_list "wikipedia"; then
    container_name="wikipedia-$process_id"
    remove_container_and_volumes "$container_name"
    if $LET_OS_DECIDE_PORT; then
      docker run -d --name "$container_name" --volume="$DOCKER_IMGS_PATH/:/data" -p :80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
      port=$(get_docker_mapped_port "$container_name" 10)
    else
      port=$(get_free_port)
      docker run -d --name "$container_name" --volume="$DOCKER_IMGS_PATH/:/data" -p "$port":80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim
    fi
    echo "Started Wikipedia instance $process_id at http://$BASE_URL:$port."
    echo "wikipedia-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#------------------------------
# Launch homepage [WebArena & VisualWebArena]
#------------------------------
if is_site_in_list "homepagewa" || is_site_in_list "homepagevwa"; then
  if is_site_in_list "homepagewa"; then
    homepage_path="$HOMEPAGE_PATH_WA"
    env="wa"
  else
    homepage_path="$HOMEPAGE_PATH_VWA"
    env="vwa"
  fi
  kill_homepage_processes "$homepage_path" "$process_id"
  port=$(get_free_port)
  
  echo "Starting homepage instance $process_id at http://$BASE_URL:$port ..."
#   update_homepage_html "$process_id" "$homepage_path/templates/index_raw.html" "$homepage_path/templates/index-$process_id.html"
#   python3 scripts/environments/update_homepage.py "$process_id" "$homepage_path/templates/index_raw.html" "$homepage_path/templates/index-$process_id.html" --base_url "$BASE_URL"
  PYTHONPATH=. python3 scripts/environments/update_homepage.py "$process_id" "$homepage_path/templates/index_raw.html" "$homepage_path/templates/index-$process_id.html" --base_url "$BASE_URL" --env "$env"

  
  nohup python3 "$homepage_path/app.py" "$process_id" --port "$port" > "$homepage_path/flask-$process_id.log" 2>&1 &
  sleep_seconds=2
  echo "Waiting $sleep_seconds seconds for the homepage to start..."
  sleep "$sleep_seconds"
    pid=$(pgrep -f "$homepage_path/app.py $process_id")
  echo "Started homepage instance $process_id at http://$BASE_URL:$port. Process ID: $pid."
  echo "homepage-$process_id http://$BASE_URL:$port" >> "$temp_url_file"
fi

#===============================================================================
# Wait for all websites to be up.
#===============================================================================
url_file="${URLS_FILE_TEMPLATE//\{PROCESS_ID\}/$process_id}"
echo "Waiting until ${sites[@]} are up..."
# Maximum number of attempts (5 minutes with 10 seconds sleep).
max_attempts=30
attempt_count=1
temp_status_file=$(mktemp)
while [ $attempt_count -le $max_attempts ]; do
  echo -e "\nAttempt $attempt_count/$max_attempts:"
  check_websites "$process_id" "$temp_url_file" "$temp_status_file"
  echo "$(cat $temp_status_file)"
  are_all_sites_up=true
  while IFS= read -r line; do
    if [[ "$line" == *" DOWN "* ]]; then
      are_all_sites_up=false
      break
    fi
  done < "$temp_status_file"
  if $are_all_sites_up; then
    echo "All requested websites are up!"
    break
  fi
  if [ $attempt_count -eq $max_attempts ]; then
    echo "Warning: Some websites failed to start after $max_attempts attempts."
    echo -e "\nFinal websites status:"
    if [ -n "$CHECK_WEBSITES_SCRIPT" ] && [ -r "$CHECK_WEBSITES_SCRIPT" ]; then
      bash "$CHECK_WEBSITES_SCRIPT" -p "$process_id" -s "${sites[*]}"
    fi
    break
  fi
  sleep_seconds=5
  echo "Waiting $sleep_seconds seconds before the next attempt..."
  sleep $sleep_seconds
  ((attempt_count++))
done

echo "Writing all UP websites for this instance to the URL file..."
# Write all UP websites for this instance to the URL file.

# Create url_file dir if it doesn't exist
write_all_instance_urls "$process_id" "$url_file"


# echo time
echo "$(date) Finished starting/resetting websites."