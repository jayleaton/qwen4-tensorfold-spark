"""check_vision_libs.py passes only when cuBLASLt is found and both pinned decoders are usable."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_vision_libs", ROOT / "scripts/check_vision_libs.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def caps(**over):
    rows = {
        "cuBLASLt": {"name": "cuBLASLt", "found": True, "usable": None},
        "libjpeg-turbo": {"name": "libjpeg-turbo", "found": True, "usable": True},
        "libwebp": {"name": "libwebp", "found": True, "usable": True},
    }
    for name, row in over.items():
        rows[name.replace("_", "-")].update(row)
    return {"optional_libraries": list(rows.values())}


def test_all_present_is_clean():
    assert mod.problems(caps()) == []


def test_a_wrong_decoder_release_is_named_with_its_problem():
    bad = mod.problems(caps(libwebp={"usable": False, "problem": "another release"}))
    assert bad == ["libwebp: usable is False another release"]


def test_missing_cublaslt_and_unlisted_rows_fail():
    assert mod.problems(caps(cuBLASLt={"found": False})) == ["cuBLASLt: found is False"]
    assert mod.problems({"optional_libraries": []}) == [f"{n}: not listed" for n in mod.NEEDED]
