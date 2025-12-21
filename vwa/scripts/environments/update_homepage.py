# NOTE[mandrade]: added to support dynamic URLs, website display names, env parallelization

import re
import subprocess
import time
from argparse import ArgumentParser

from browser_env.env_config import SITE_DISPLAY_NAMES, Website


def get_docker_mapped_port(container_name: str, max_wait: int = 10) -> str:
    """
    Get the mapped port for a Docker container.

    Args:
        container_name: Name of the Docker container
        max_wait: Maximum time to wait for port mapping (seconds)

    Returns:
        Port number as string, or empty string if not found
    """
    waited = 0
    sleep_seconds = 1

    while waited < max_wait:
        try:
            result = subprocess.run(["docker", "port", container_name], capture_output=True, text=True, check=True)
            if result.stdout.strip():
                # Parse Docker port output more robustly
                lines = result.stdout.strip().split("\n")
                if lines:
                    port_line = lines[0]

                    # Extract port using regex to handle various formats
                    import re

                    # Look for patterns like :8080 or 8080/tcp
                    port_match = re.search(r":(\d+)", port_line)
                    if port_match:
                        return port_match.group(1)

        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"Error parsing port for {container_name}: {e}")
            pass

        time.sleep(sleep_seconds)
        waited += sleep_seconds

    return ""


def get_port_site_by_id(site: str, instance_id: str) -> str:
    """
    Get the port for a specific site and instance ID.

    Args:
        site: Site name (e.g., "shopping", "classifieds")
        instance_id: Instance identifier

    Returns:
        Port number as string
    """
    container_name = f"{site}-{instance_id}"
    return get_docker_mapped_port(container_name, 10)


def update_homepage_html_regex(instance_id: str, template_file: str, output_file: str, base_url: str = "localhost", env: str = "") -> None:
    """Alternative implementation using regex for replacements."""

    # Get ports for each site
    if env == "vwa":
        ports = {
            "shopping": get_port_site_by_id("shopping", instance_id),
            "classifieds": get_port_site_by_id("classifieds", instance_id),
            "reddit": get_port_site_by_id("reddit", instance_id),
            "wikipedia": get_port_site_by_id("wikipedia", instance_id),
        }
    elif env == "wa":
        ports = {
            "shopping": get_port_site_by_id("shopping", instance_id),
            "classifieds": get_port_site_by_id("classifieds", instance_id),
            "reddit": get_port_site_by_id("reddit", instance_id),
            "wikipedia": get_port_site_by_id("wikipedia", instance_id),
        }
    else:
        ports = {
            "shopping": get_port_site_by_id("shopping", instance_id),
            "classifieds": get_port_site_by_id("classifieds", instance_id),
            "reddit": get_port_site_by_id("reddit", instance_id),
            "wikipedia": get_port_site_by_id("wikipedia", instance_id),
            "shopping_admin": get_port_site_by_id("shopping_admin", instance_id),
            "gitlab": get_port_site_by_id("gitlab", instance_id),
        }

    # Read template
    with open(template_file, "r") as f:
        content = f.read()

    # Replace using regex
    content = re.sub(r"<your-server-hostname>", base_url, content)
    for site_id, port in ports.items():
        site_display_name = SITE_DISPLAY_NAMES[Website(site_id)]
        content = re.sub(f"<{site_id}-name>", site_display_name, content)
        content = re.sub(f"<{site_id}-port>", port, content)

    # Write output
    with open(output_file, "w") as f:
        f.write(content)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("instance_id", type=str, help="Docker instance ID")
    parser.add_argument("template_file", type=str, help="Template file")
    parser.add_argument("output_file", type=str, help="Output file")
    parser.add_argument("--base_url", type=str, help="Base URL", default="localhost")
    parser.add_argument("--env", type=str, help="Environment", default="")
    args = parser.parse_args()
    update_homepage_html_regex(args.instance_id, args.template_file, args.output_file, args.base_url, args.env)
