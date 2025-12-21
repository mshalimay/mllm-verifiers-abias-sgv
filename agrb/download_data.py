import os
import random
import time

from huggingface_hub import login, snapshot_download

# Set your Hugging Face token
HF_TOKEN = "" or os.environ.get("HF_TOKEN")

login(token=HF_TOKEN)

AGENT = "**"

VISUALWEBRENA_PATTERNS = [
    f"trajectories/cleaned/visualwebarena/{AGENT}/**",
    f"screenshots/visualwebarena/{AGENT}/**",
    # f"judgments/visualwebarena/{AGENT}/**",
]


def download_with_retry(max_retries=5):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            snapshot_download(
                repo_id="McGill-NLP/agent-reward-bench",
                repo_type="dataset",
                local_dir="./trajectories/",
                # max_workers=3,
                allow_patterns=VISUALWEBRENA_PATTERNS,
            )
            print("Download completed successfully!")
            break
        except Exception as e:
            print(e)
            wait_time = (2**attempt) + random.uniform(0, 1)
            print(f"Rate limited. Waiting {wait_time:.1f} seconds before retry...")
            time.sleep(wait_time)


download_with_retry()
