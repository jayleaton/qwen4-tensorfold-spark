#!/usr/bin/env python3
"""Check that every relative Markdown link in the repo's tracked and untracked (not ignored) files names a file that
exists, and that a #fragment on a Markdown target names one of its headings. External links are not fetched.

  scripts/check_links.py          scan the repo; exit 1 and list each broken link as path:line: target
  scripts/check_links.py PATH...  scan only these Markdown files
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")
EXTERNAL = ("http://", "https://", "mailto:", "tel:")


def anchors(text: str) -> set[str]:
    """GitHub-style heading anchors: lower case, punctuation dropped, spaces to hyphens."""
    out, in_fence = set(), False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("#"):
            continue
        title = line.lstrip("#").strip().lower()
        out.add(re.sub(r"[^\w\- ]", "", title).replace(" ", "-"))
    return out


def broken(path: Path, root: Path = ROOT) -> list[str]:
    found, in_fence = [], False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for target in LINK.findall(re.sub(r"`[^`]*`", "", line)):
            if target.startswith(EXTERNAL):
                continue
            name, _, fragment = target.partition("#")
            dest = path if not name else (path.parent / name)
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            if not dest.exists():
                found.append(f"{rel}:{n}: {target} (no such file)")
            elif fragment and dest.suffix == ".md" and fragment not in anchors(dest.read_text(encoding="utf-8")):
                found.append(f"{rel}:{n}: {target} (no such heading)")
    return found


def markdown_files(root: Path = ROOT) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "-co", "--exclude-standard", "--", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [root / p for p in out.split("\0") if p and (root / p).is_file()]


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv] or markdown_files()
    problems = [p for f in files for p in broken(f)]
    for p in problems:
        print(p)
    print(f"check-links: {'FINDINGS' if problems else 'clean'} ({len(files)} Markdown files)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
