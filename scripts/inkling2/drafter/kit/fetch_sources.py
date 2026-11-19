"""Fetch the public Claude Code session datasets, anonymously and at pinned revisions, into a Hugging Face cache.

Usage: fetch_sources.py <hub folder>
"""
import os
import sys

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import snapshot_download  # noqa: E402

SOURCES = {
    "choucsan/mimo-claude-code-traces-1k": (
        "39cc3fc3ed608fff800220445bad9eb1738516f8", ["session/**/*.jsonl", "README.md"]),
    "armand0e/claude-fable-5-claude-code": (
        "7388ef96f3179f23b6b3618b96572f1aa9b84510", ["*.jsonl", "README.md"]),
    "AlinCiocan/fable-5-claude-code-traces": (
        "e33ebbca230ae258b2c28aeee9fe3429e7fbeab6", ["*.jsonl", "README.md"]),
    "trace-commons/agent-traces": (
        "112ebd4d03ce852b00e935d523107c3d0c9a65bf", ["sessions/claude_code/*.jsonl", "data/*.parquet", "README.md"]),
    "crispwisp/wisp-claude-code-sessions": (
        "c2c90b59174318ab0b163ec9c9ac82bb879288ce", ["transcripts/**/*.jsonl", "README.md"]),
    "liangzhidanta/claude-code-glm53-swesmith-trajectories": (
        "73041e4958062af2311d1ec24cced5f4a932f530",
        ["README.md", "RELEASE_MANIFEST.json", "data/sft/*.jsonl", "data/metadata/production_v2_summary.json"]),
}


def main() -> None:
    hub = sys.argv[1]
    os.makedirs(hub, exist_ok=True)
    for repo, (revision, patterns) in SOURCES.items():
        path = snapshot_download(repo, repo_type="dataset", revision=revision, cache_dir=hub, token=False,
                                 allow_patterns=patterns)
        files = sum(len(names) for _, _, names in os.walk(path))
        print("%s %s %d files" % (repo, os.path.basename(path)[:12], files), flush=True)


if __name__ == "__main__":
    main()
