#!/usr/bin/env python3
"""Fetch and verify the in-model runner assets (calc.tfpc and the graph-v7-powmod bank) for the runner image layer.

Usage: fetch_runner_assets.py RELEASE MANIFEST DEST
RELEASE names the archive (`url URL`, `sha256 HEX`: config/runner-assets.release); MANIFEST lists exactly the files it
must hold (`HEX  PATH`: config/runner-assets.sha256). The archive's sha256 is checked before it is opened; only
regular files at safe relative paths are extracted (no links, devices, absolute or `..` paths); DEST must then hold
exactly MANIFEST's files at MANIFEST's sha256s. Any miss fails with the file and URL named. DEST must not exist."""

import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath


def fail(text):
    print(f"runner assets: {text}", file=sys.stderr)
    sys.exit(1)


def main():
    release, manifest, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    pins = dict(line.split(None, 1) for line in release.read_text().splitlines() if line and not line.startswith("#"))
    url, want = pins["url"].strip(), pins["sha256"].strip()
    files = {}
    for line in manifest.read_text().splitlines():
        if line and not line.startswith("#"):
            digest, path = line.split(None, 1)
            files[path.strip()] = digest
    print(f"runner assets: downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        fail(f"sha256 mismatch for {url}: expected {want}, got {got}")
    print(f"runner assets: archive sha256 {got[:12]}... ok, {len(data)} bytes")
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            p = PurePosixPath(m.name)
            if m.isdir():
                continue
            if not m.isfile() or p.is_absolute() or ".." in p.parts:
                fail(f"refused archive member {m.name!r} (type {m.type!r}) from {url}")
            rel = PurePosixPath(*[x for x in p.parts if x != "."])
            if str(rel) not in files:
                fail(f"unexpected file {rel} in {url}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
    for path, digest in files.items():
        f = dest / path
        if not f.is_file():
            fail(f"{path} missing from {url}")
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != digest:
            fail(f"sha256 mismatch for {path} from {url}: expected {digest}, got {h}")
    extra = [str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file() and str(f.relative_to(dest)) not in files]
    if extra:
        fail(f"extra files {extra[:3]} from {url}")
    print(f"runner assets: {len(files)} files match {manifest.name}")


if __name__ == "__main__":
    main()
