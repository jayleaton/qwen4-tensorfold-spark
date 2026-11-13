"""scripts/check_links.py passes good relative links and headings and reports a missing file or heading."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "scripts" / "check_links.py"


def scan(*paths):
    return subprocess.run([sys.executable, str(SCAN), *map(str, paths)], capture_output=True, text=True)


def test_good_links_pass(tmp_path):
    (tmp_path / "b.md").write_text("# Gates and risks\n")
    a = tmp_path / "a.md"
    a.write_text(
        "[b](b.md) [h](b.md#gates-and-risks) [x](https://example.com/nothing) `[c](c.md)`\n```\n[d](d.md)\n```\n"
    )
    assert scan(a).returncode == 0


def test_missing_file_and_heading_fail(tmp_path):
    (tmp_path / "b.md").write_text("# Gates\n")
    a = tmp_path / "a.md"
    a.write_text("[c](c.md)\n[h](b.md#risks)\n")
    r = scan(a)
    assert r.returncode == 1
    assert "a.md:1: c.md (no such file)" in r.stdout and "a.md:2: b.md#risks (no such heading)" in r.stdout
