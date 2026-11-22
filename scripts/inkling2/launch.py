#!/usr/bin/env python3
"""Start the profile's server as search.py starts a trial: words before the command, then -- and settings (NAME=VALUE)."""
import json
import os
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
# Beside search.py in the repo, or beside a kit's copy of it
kit = here / "repo/scripts/inkling2"
sys.path.insert(0, str(kit if kit.is_dir() else here))
import search
import serve

split = sys.argv.index("--")
wrap, settings = sys.argv[1:split], dict(word.rsplit("=", 1) for word in sys.argv[split + 1:])
base = ["--profile", "tensor-4", "--draft", "--server", os.environ["SERVER"], "--port", "8103", "--tuned", os.devnull,
        "--packs", os.environ["PACKS"]]
argv, env = search.plan(base, settings)
argv = serve.glm53.resolve_server_arguments(argv, env)
print(json.dumps({"argv": argv, "env": env}), file=sys.stderr, flush=True)
full = serve.execution_environment(env)
# Behind a wrapper the settings ride on the command too, so the server has them whatever the wrapper hands on
words = wrap + ["env"] + [f"{name}={value}" for name, value in env.items()] + argv if wrap else argv
os.execvpe(words[0], words, full)
