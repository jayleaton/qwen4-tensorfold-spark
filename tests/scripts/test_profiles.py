"""Provisional profile stubs stay stubs: one per size, marked provisional, no measured claim, no DeepSeek knob."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SIZES = ("28b", "125b", "256b")
REQUIRED = (
    "PROFILE",
    "RANKS",
    "IMAGE",
    "MODEL_DIR",
    "PREPARED_DIR",
    "STATE_DIR",
    "CHECKPOINT",
    "QUANT_FORMAT",
    "CONTEXT",
)


def values(size, name=None):
    out = {}
    path = ROOT / "profiles/provisional" / f"{name or f'qwen4-{size}'}.env.example"
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            out[key] = value.strip('"')
    return out


@pytest.mark.parametrize("size", SIZES)
def test_profile_has_required_keys_and_its_own_folders(size):
    v = values(size)
    assert all(k in v for k in REQUIRED), [k for k in REQUIRED if k not in v]
    assert v["PROFILE"] == f"qwen4-{size}"
    for k in ("MODEL_DIR", "PREPARED_DIR", "STATE_DIR"):
        assert v[k].startswith("<") and f"qwen4-{size}" in v[k]


@pytest.mark.parametrize("size", SIZES)
def test_profile_claims_nothing_unmeasured(size):
    text = (ROOT / "profiles/provisional" / f"qwen4-{size}.env.example").read_text()
    assert text.startswith("# PROVISIONAL")
    v = values(size)
    for k in ("CHECKPOINT", "QUANT_FORMAT", "CONTEXT"):
        assert v[k] == "TBD", (k, v[k])
    assert not any(k.startswith("TF_DSV41_") for k in v)
    assert v["RANKS"] in ("1", "2")
    if v["RANKS"] == "1":
        assert not any(k.startswith(("TF_TP_", "NCCL_")) for k in v)


def test_engine_pin_is_tbd_or_a_full_sha():
    pin = (ROOT / "config/tensorfold.rev").read_text().strip()
    assert pin == "TBD" or re.fullmatch(r"[0-9a-f]{40}", pin)
    assert (ROOT / "config/tensorfold.repo").read_text().strip() == "https://github.com/jayleaton/TensorFold.git"


def test_flash_next_control_is_tp1_with_its_measured_chunks():
    name = "qwen38-flash-next-tp1"
    text = (ROOT / "profiles/provisional" / f"{name}.env.example").read_text()
    assert text.startswith("# PROVISIONAL") and "not a Qwen4 size" in text
    v = values(None, name)
    assert all(k in v for k in REQUIRED), [k for k in REQUIRED if k not in v]
    assert v["PROFILE"] == name and v["RANKS"] == "1"
    assert not any(k.startswith(("TF_TP_", "NCCL_", "TF_DSV41_")) for k in v)
    for k in ("MODEL_DIR", "PREPARED_DIR", "STATE_DIR"):
        assert v[k].startswith("<") and name in v[k]
    assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", v["CHECKPOINT"])
    assert (v["TF_FLASHNEXT_CHUNK"], v["TF_FLASHNEXT_CHUNK_SOLO"]) == ("1536", "4096")


def test_flash_next_candidate_pins_its_engine_and_runs_compute_in_engine():
    name = "qwen38-flash-next-tp1-tool-full"
    text = (ROOT / "profiles/provisional" / f"{name}.env.example").read_text()
    assert text.startswith("# PROVISIONAL") and "not a Qwen4 size" in text
    v = values(None, name)
    control = values(None, "qwen38-flash-next-tp1")
    assert all(k in v for k in REQUIRED), [k for k in REQUIRED if k not in v]
    assert v["PROFILE"] == name and v["RANKS"] == "1"
    assert not any(k.startswith(("TF_TP_", "NCCL_", "TF_DSV41_")) for k in v)
    for k in ("MODEL_DIR", "PREPARED_DIR", "STATE_DIR"):
        assert v[k].startswith("<") and name in v[k]
    assert re.fullmatch(r"[0-9a-f]{40}", v["ENGINE_REV"]) and v["ENGINE_REV"][:12] in v["IMAGE"]
    assert v["ENGINE_REPO"] == (ROOT / "config/tensorfold.repo").read_text().strip()
    for k in ("CHECKPOINT", "QUANT_FORMAT", "CONTEXT", "TF_FLASHNEXT_CHUNK", "TF_FLASHNEXT_CHUNK_SOLO"):
        assert v[k] == control[k], k
    # compute runs in the engine for every request: the lane, its server-side default, the runner and S3 all on
    for k in ("TF_COMPUTE_LANE", "TF_COMPUTE_LANE_DEFAULT", "TF_RUNNER", "TF_COMPUTE_LANE_RUNNER"):
        assert v[k] == "1", k


def test_flash_next_tp2_candidate_pins_its_engine_and_credits_her_recipe():
    name = "qwen38-flash-next-tp2-mia-tool-full"
    text = (ROOT / "profiles/provisional" / f"{name}.env.example").read_text()
    assert text.startswith("# PROVISIONAL") and "not a Qwen4 size" in text
    assert "MiaAI-Lab" in text and "Apache License 2.0" in text
    assert re.search(r"github\.com/MiaAI-Lab/[\w.-]+ at [0-9a-f]{40}", text)
    v = values(None, name)
    assert all(k in v for k in REQUIRED), [k for k in REQUIRED if k not in v]
    assert v["PROFILE"] == name and v["RANKS"] == "2"
    assert not any(k.startswith("TF_DSV41_") for k in v)
    for k in ("MODEL_DIR", "PREPARED_DIR", "STATE_DIR", "COST_CACHE_DIR"):
        assert v[k].startswith("<") and name in v[k]
    assert re.fullmatch(r"[0-9a-f]{40}", v["ENGINE_REV"]) and v["ENGINE_REV"][:12] in v["IMAGE"]
    assert v["ENGINE_REPO"] == (ROOT / "config/tensorfold.repo").read_text().strip()
    assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", v["CHECKPOINT"])
    assert v["TF_FLASHNEXT_ENGINE"] == "mia"
    for k in ("TP_MASTER", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"):
        assert v[k].startswith("<"), k
    for k in ("TF_COMPUTE_LANE", "TF_COMPUTE_LANE_DEFAULT", "TF_RUNNER", "TF_COMPUTE_LANE_RUNNER"):
        assert v[k] == "1", k
