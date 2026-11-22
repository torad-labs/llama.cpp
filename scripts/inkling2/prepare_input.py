#!/usr/bin/env python3
"""Start the stock server only to tokenize, write the replay input, and stop the server again.

Run it on the host that holds the model, after approval. It reuses the server plan of the recorded run and
replay_prepare.py, which rejects any prompt whose recorded hash differs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "plan", "template", "libs", "probe", "reference", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--build-sha256", required=True)
    p.add_argument("--model-sha256", required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_bytes())
    argv = list(plan["argv"])
    argv[0] = str(args.server.resolve())
    argv[argv.index("--chat-template-file") + 1] = str(args.template.resolve())
    url = "http://127.0.0.1:" + argv[argv.index("--port") + 1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "LLAMA_"))}
    env.update(plan["env"], LD_LIBRARY_PATH=str(args.libs.resolve()), LLAMA_INKLING2_EXPERT_SPLIT="0")
    log = args.out.with_suffix(".server.log")
    with log.open("wb") as f, subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT) as server:
        try:
            while True:
                if server.poll() is not None:
                    raise SystemExit("the tokenizer server exited before it was ready; see " + str(log))
                try:
                    with urllib.request.urlopen(url + "/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(5)
            subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("replay_prepare.py")),
                            "--probe", args.probe, "--reference", args.reference, "--tokenizer", url,
                            "--build-sha256", args.build_sha256, "--model-sha256", args.model_sha256,
                            "--out", args.out], check=True)
        finally:
            if server.poll() is None:
                server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
