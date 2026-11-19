#!/usr/bin/env python3
"""Builds the read-only folder a session served by the rented machine may read.

Whatever such a session reads is sent to that machine as its prompt, so the folder holds only what may leave this
one: the fork's tracked files as HEAD has them, without the seats' instruction files, and the text records of past
runs with this machine's home path replaced. Binaries, archives and text over the size cap are left out and listed.
It ends with a credential scan of the folder; read its findings before any session opens it.

Usage: material.py DEST [--jobs DIR] [--research DIR]

  engine/            the fork at HEAD
  engine-commits.txt the commits since the build the measurements were taken on
  runs/jobs/         results of the drafter's and the engine seat's rig jobs (dspark-g*, speed-*, knobs-*)
  runs/<folder>/     the research folders named in RESEARCH
  MANIFEST.txt       every file with its size
  LEFT-OUT.txt       what was not copied
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HOME = os.path.expanduser("~")
MEASURED = "7b262d27f1c55a0f96842a2163823dbb620013a7"
JOB_PREFIXES = ("dspark-g", "speed-", "knobs-")
RESEARCH = ("inkling2-dspark-train-20261119", "inkling2-dspark-20261118", "inkling2-tp4-20261118")
TEXT = {".md", ".json", ".jsonl", ".log", ".txt", ".py", ".sh", ".csv", ".toml", ".cpp", ".c", ".h", ".patch",
        ".status", ".args", ".sha", ".sha256", ".exit"}
CAP = 1 << 20
JOB_CAP = 4 << 20
SEAT_FILES = {"CLAUDE.md", "AGENTS.md"}
SEAT_DIRS = (".campaign/", ".claude/")
SKIP_DIRS = {"portable-src", "portable-build", "portable-base", "portable-run", ".git", "__pycache__", ".venv",
             "node_modules", "jobs", ".venv-selftest", "build-venv", ".rig-job"}

left_out = []


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True, stdout=subprocess.PIPE).stdout


def put(dst, data):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)


def engine(dest):
    tracked = [p for p in git("ls-files", "-z").decode().split("\0") if p]
    changed = {p for p in git("diff", "--name-only", "-z", "HEAD").decode().split("\0") if p}
    n = 0
    for rel in tracked:
        if os.path.basename(rel) in SEAT_FILES or rel.startswith(SEAT_DIRS):
            continue
        src = os.path.join(REPO, rel)
        dst = os.path.join(dest, "engine", rel)
        if rel in changed:
            put(dst, git("show", "HEAD:" + rel))
        elif os.path.islink(src) or not os.path.isfile(src):
            continue
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        n += 1
    head = git("rev-parse", "HEAD").decode().strip()
    since = git("log", "--format=%h %ad %s", "--date=format:%Y-%m-%d %H:%M", MEASURED + "..HEAD").decode()
    put(os.path.join(dest, "engine-commits.txt"),
        ("engine/ is the fork at commit %s.\n"
         "The rented machine's build, and every measurement of 2026-11-19, is commit\n"
         "%s. The commits engine/ has after it, newest first:\n\n%s" % (head, MEASURED, since)).encode())
    print("engine: %d files at %s (%d taken from HEAD over an uncommitted edit)" % (n, head[:9], len(changed)))


def tree(dest, src_root, dst_root, cap=CAP):
    n = b = 0
    for base, dirs, files in os.walk(src_root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("src-"))
        for name in sorted(files):
            src = os.path.join(base, name)
            rel = os.path.relpath(src, src_root)
            if os.path.islink(src) or not os.path.isfile(src):
                continue
            size = os.path.getsize(src)
            out = os.path.join(os.path.relpath(dst_root, dest), rel)
            if os.path.splitext(name)[1] not in TEXT or size > cap:
                left_out.append((out, size))
                continue
            with open(src, "rb") as f:
                data = f.read()
            if b"\0" in data[:4096]:
                left_out.append((out, size))
                continue
            put(os.path.join(dst_root, rel), data.replace(HOME.encode(), b"/home/user"))
            n += 1
            b += size
    print("%s: %d files, %.1f MB" % (os.path.relpath(dst_root, dest), n, b / 1e6))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dest")
    p.add_argument("--jobs", default=os.path.join(HOME, ".local/share/rig/local/jobs"), help="rig's folder of jobs")
    p.add_argument("--research", default=os.path.join(HOME, "Documents/dev/infra/rig/local-nvme/research"),
                   help="the folder holding the research folders named in RESEARCH")
    args = p.parse_args()
    dest = os.path.abspath(args.dest)
    if os.path.exists(dest):
        sys.exit("refusing to write into an existing folder: " + dest)
    engine(dest)
    for job in sorted(os.listdir(args.jobs)) if os.path.isdir(args.jobs) else []:
        res = os.path.join(args.jobs, job, "results")
        if job.startswith(JOB_PREFIXES) and os.path.isdir(res):
            tree(dest, res, os.path.join(dest, "runs/jobs", job), cap=JOB_CAP)
    for name in RESEARCH:
        src = os.path.join(args.research, name)
        if os.path.isdir(src):
            tree(dest, src, os.path.join(dest, "runs", name))
        else:
            print("runs/%s: not on this machine" % name)
    lines = []
    for base, dirs, files in os.walk(dest):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(base, name)
            lines.append("%10d  %s" % (os.path.getsize(path), os.path.relpath(path, dest)))
    put(os.path.join(dest, "MANIFEST.txt"), ("\n".join(lines) + "\n").encode())
    left_out.sort(key=lambda r: -r[1])
    put(os.path.join(dest, "LEFT-OUT.txt"),
        ("Files of the run folders that are not here: binaries, archives, or text over the size cap.\n"
         "bytes  path\n" + "\n".join("%12d  %s" % (s, q) for q, s in left_out) + "\n").encode())
    print("files in all: %d; left out of the run folders: %d" % (len(lines), len(left_out)))
    report = dest + ".gitleaks.json"
    scan = subprocess.run(["gitleaks", "dir", dest, "--no-banner", "--redact", "--report-format", "json",
                           "--report-path", report])
    print("credential scan: exit %d, findings in %s" % (scan.returncode, report))
    print("A finding under engine/ that the fork's .gitleaksignore already names is known. Any other is read by the\n"
          "project's lead, or its file is removed from the folder, before a session opens it.")
    sys.exit(scan.returncode)


main()
