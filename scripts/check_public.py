"""Fail if a release contains private references or unapproved binary files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
BINARY_SUFFIXES = {".pt", ".pth", ".safetensors", ".mp4", ".mov", ".mkv", ".png", ".jpg", ".pdf", ".parquet", ".jsonl", ".zip"}
RELEASE_ASSETS = {"assets/method.png", "assets/dsr_bench_table.png", "paper.pdf"}
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".md", ".txt", ".sh", ".json", ""}
FORBIDDEN = [
    re.compile(r"\b[Ee]\d{1,3}(?:[_-][A-Za-z0-9]+)?\b"),
    re.compile(r"/data/(?:yk|2/)"),
    re.compile(r"D:[\\/]yk[\\/]Study", re.IGNORECASE),
    re.compile(r"KindofCrazy/dsr\b", re.IGNORECASE),
    re.compile(r"sft-v1\.2", re.IGNORECASE),
    re.compile(r"infra-vgm-feature-extract", re.IGNORECASE),
    re.compile(r"(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False,
    )
    if result.returncode == 0 and result.stdout:
        return [ROOT / path.decode("utf-8") for path in result.stdout.split(b"\0") if path]
    return [
        path for path in ROOT.rglob("*") if path.is_file()
        and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
    ]


def main() -> None:
    errors = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            if relative.as_posix() not in RELEASE_ASSETS:
                errors.append(f"{relative}: unapproved binary file")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or relative.as_posix() == "scripts/check_public.py":
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in FORBIDDEN):
                errors.append(f"{relative}:{number}: forbidden release content")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Public content scan passed")


if __name__ == "__main__":
    main()
