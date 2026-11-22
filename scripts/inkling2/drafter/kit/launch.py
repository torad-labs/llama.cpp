"""Start the recording tool with the serving launcher's own arguments and environment.

The launcher (scripts/inkling2/serve.py in the engine source) decides how Inkling is served: the split, the pool, the
micro-batch, the cache types and the engine's environment. This asks it for the draft layout without a draft and
without the projector, and runs the recording tool in place of the server with exactly that.

A probe may change the engine's environment with --set KEY=VALUE; a recording to train on never does.

Usage: launch.py --source DIR --tool PATH (--model GGUF | --packs DIR) [--slots N] [--set KEY=VALUE]... [--show]
           <prepare|capture> [tool options]
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="the engine source tree")
    parser.add_argument("--tool", required=True, type=Path, help="the recording tool, beside the engine's libraries")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--model", type=Path, help="a GGUF in place of the profile's pack (the rehearsal fixture)")
    where.add_argument("--packs", type=Path, help="the folder holding the profile's pack")
    parser.add_argument("--slots", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="one more variable of the engine's environment, over the launcher's own (probes only)")
    parser.add_argument("--show", action="store_true", help="print the command and the settings, start nothing")
    parser.add_argument("command", choices=["prepare", "capture"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("inkling2_serve", args.source / "scripts/inkling2/serve.py")
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    asked = ["--profile", "tensor-4", "--draft-layout", "--no-draft", "--text-only", "--server", str(args.tool)]
    asked += ["--model", str(args.model)] if args.model else ["--packs", str(args.packs)]
    if args.slots is not None:
        asked += ["--slots", str(args.slots)]
    argv, settings = serve.make_plan(serve.arguments(asked))
    serve.glm53.verify_assets({serve.TEMPLATE: serve.TEMPLATE_SHA256})
    full = [argv[0], args.command, *args.rest, "--", *argv[1:]]
    extra = {}
    for pair in args.set:
        key, eq, value = pair.partition("=")
        if not key or not eq:
            raise SystemExit("--set takes KEY=VALUE, not %r" % pair)
        extra[key] = value
    if args.show:
        print(json.dumps({"argv": full, "env": settings, "set": extra}, indent=2))
        return
    for label, path in (("tool", Path(full[0])), ("model", Path(argv[argv.index("-m") + 1]))):
        if not path.is_file():
            raise SystemExit("missing %s: %s" % (label, path))
    sys.stdout.flush()
    env = dict(serve.execution_environment(settings))
    env.update(extra)
    if extra:
        print("launch: the engine's environment is changed for this run: %s" % " ".join(sorted(args.set)), flush=True)
    sys.stdout.flush()
    os.execve(full[0], full, env)


if __name__ == "__main__":
    main()
