import os
import subprocess


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    command = ["vllm", "serve", "ByteDance-Seed/UI-TARS-1.5-7B"]
    command.extend(["--host", "127.0.0.1"])
    command.extend(["--max-model-len", "28000"])
    subprocess.run(command)


if __name__ == "__main__":
    main()
