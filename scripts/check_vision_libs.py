"""Check `tensorfold-native capabilities --json` for what TP1's --vision loads: cuBLASLt found, JPEG and WebP usable.

usage: python3 -I scripts/check_vision_libs.py CAPABILITIES.json
The engine loads the image tower's GEMMs from cuBLASLt and decodes JPEG and WebP only with the exact releases it pins
(libjpeg-turbo 3.1.4.1, libwebp 1.6.0); `usable` is its own check of a decoder's exports and version.
"""

import json
import sys

NEEDED = {"cuBLASLt": "found", "libjpeg-turbo": "usable", "libwebp": "usable"}


def problems(caps):
    """The --vision libraries that are missing or unusable, as lines; empty when all are there."""
    rows = {row.get("name"): row for row in caps.get("optional_libraries", [])}
    out = []
    for name, field in NEEDED.items():
        row = rows.get(name)
        if row is None:
            out.append(f"{name}: not listed")
        elif row.get(field) is not True:
            out.append(f"{name}: {field} is {row.get(field)!r} {row.get('problem') or ''}".rstrip())
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    with open(argv[1]) as f:
        caps = json.load(f)
    bad = problems(caps)
    for line in bad:
        print(f"check_vision_libs: {line}", file=sys.stderr)
    if not bad:
        print("check_vision_libs: cuBLASLt found; libjpeg-turbo and libwebp usable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
