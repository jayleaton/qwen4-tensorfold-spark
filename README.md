Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# Qwen3.8-Flash-Next (and later Qwen4) on TensorFold, NVIDIA DGX Spark

**This repo is the recipe (profiles, patches, build script and docs), not the engine.** The engine is
[TensorFold](https://github.com/ashhart/TensorFold), written in Zig: the pinned, unmodified submodule
`vendor/TensorFold` plus the 84 patches in [`patches/`](patches), applied when you build the image. The Python files
here are only this repo's own checks.

The profiles serve Qwen3.8-Flash-Next, the released preview of the Qwen4 architecture, with the engine's compute lane
(exact computation in every thinking chat), the in-model runner on an exact circuit and graph bank, and the lane's
calls on that runner. The Qwen4 profiles are placeholders: no Qwen4 checkpoint has been published.

## Start here

| You have | Profile | Patches | Checkpoint |
| --- | --- | --- | --- |
| **one Spark** | **TP1**: [qwen38-flash-next-tp1-tool-full](profiles/provisional/qwen38-flash-next-tp1-tool-full.env.example) | `patches/series.tp1` (0001-0084) | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, MLX 4-bit, 65,536 context |
| **two linked Sparks** | **TP2**: [qwen38-flash-next-tp2-mia-tool-full](profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env.example) | `patches/series.tp2` (0001-0080) | `nvidia/Qwen3.8-Flash-Next-NVFP4`, NVFP4, 1,048,576 context |

Then follow the steps below: [build the image](#2-build-the-image), [serve on one Spark](#3-serve-on-one-spark-tp1)
or [on two](#4-serve-on-two-sparks-tp2), and [check it answers](#5-check-it-answers).

TP2 runs Mia's (MiaAI-Lab's) Flash-Next engine, forward-ported onto this base and selected with
`TF_FLASHNEXT_ENGINE=mia`. TP1 runs TensorFold's own `flashnext` family. Both carry the same compute lane and runner.
The control profile [qwen38-flash-next-tp1](profiles/provisional/qwen38-flash-next-tp1.env.example) is the plain TP1
boot with no lane or runner. It is there to compare against, not to serve. Each profile pins its checkpoint revision
in `CHECKPOINT` (`repo@revision`).

## The engine: upstream TensorFold plus patches

| | |
| --- | --- |
| Upstream | [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) at `aaac299d1d90c2625e54cb50f3f0d1494c1c33ea` (the `vendor/TensorFold` submodule, `config/tensorfold.rev`), unmodified |
| `patches/0001`-`0080` | the TP2 engine: Mia's dual-Spark Flash-Next engine as `flashnext_mia` (0002-0012: her own commits, authored MiaAI-Lab), then the compute lane, the in-model runner and its frontend, the call gates, `tool_choice`, the thinking-budget fixes, lane cost keeping and the runner's digit stop (0001-0076); then, for this base, three lean commits and one settings fix (0077-0080, below) |
| `patches/0081`-`0084` | four runner commits on top, for TP1: the digit hold after a stop, no word glued to the ` @engine` mark, the quoted rule (no engine value inside an open quotation), and the reply's runner receipt (`tensorfold.runner`) listing the digit stops that fired (`stops`) |
| `patches/series.tp1`, `series.tp2` | which patches each profile applies, in order, and the tree sha the result must have |

The patches are a `git format-patch` series; each keeps its original author. `scripts/build.sh` applies them with
`git am` and refuses to go on unless the tree is the one the series file pins.

**Base moved 2026-11-13** from `f212eeabd` to upstream main `aaac299d1` (786 upstream commits). The engine commits are
the same, moved over: the TP1 runner commits change line for line what they did on `f212eeabd`, and the conflict
resolutions keep upstream's code and ours side by side. Four commits are new on this base:
- 0077 and 0078 keep three files under the 600-line limit and two test comments under 120 columns;
- 0079: upstream's own Flash Next now reads four settings Mia's engine also reads (`TF_FLASHNEXT_CONFIDENCE`,
  `_KEEP_LATE`, `_PROFILE`, `_SHARED_SIDE`), and each now has one definition, upstream's. `TF_FLASHNEXT_SHARED_SIDE`
  takes `0` or `1` only (it used to take `on` and `off` too). No profile sets any of them;
- 0080 keeps `native/cuda.zig` under the limit on this base.

**Re-gated on this base 2026-11-13:** the images `scripts/build.sh` builds from this recipe ran the same release gates
as the `f212eeabd` builds (TP2 `d306dc2fe26e`, TP1 `b62876185195`), with the same boot lines and env files. On every
gate with fixed sampling the replies are the same, byte for byte or token count for token count. TP2 decode is equal
or up to 6% faster. An agent task run in 8 fresh sessions came out 7/8 twice on this base and 7/8 on `d306dc2fe`
(see the TP2 rows under [Measured results](#measured-results) and [Known limits](#known-limits)). The `f212eeabd`
trees remain buildable from the recipe at `e2aca8b`.

## Prerequisites

- **Hardware:** NVIDIA DGX Spark (GB10). The engine is built for `sm_121`. TP2 needs two Sparks.
- **Link (TP2 only):** a RoCE link between the two Sparks. The profile uses both RoCE rails (`NCCL_IB_HCA`), GID
  index 3 on both nodes and 4 NCCL channels. You need each node's link interface name, both rail device names and
  rank 0's address on the link.
- **Docker** with GPU access through the NVIDIA container runtime, on each Spark.
- **To build:** on a Spark, the CUDA toolkit (`nvcc` with `sm_121`, as DGX OS ships it), [Zig
  0.17.0](https://ziglang.org/download/) (the engine's pinned version), git, python3 and Docker. The build needs no
  GPU. The native compile needs about 5 GiB of memory at `JOBS=1`.
- **Driver:** the profiles were measured with the host's 580 kernel module. The images carry the CUDA 13.3 user-mode
  driver (libcuda 610.43.02, taken from Mia's NGC-based image) and use it only while the kernel module is older than
  610 and `cuInit` succeeds through it; otherwise the host driver. `TF_CUDA_COMPAT=0` forces the host driver. Each
  rank logs which one it uses.
- **Checkpoint:** download the exact `CHECKPOINT` revision from Hugging Face into `MODEL_DIR` (on both Sparks for
  TP2). The profiles set `HF_HUB_OFFLINE=1`.
- **Disk:** no disk figure has been measured for these profiles. Budget for the checkpoint, the TensorFold checkout,
  the build tree, the base images the build copies from (TP2: Mia's image), and the finished image.

## Quickstart

### 1. Clone and make the env file

```sh
git clone https://github.com/jayleaton/qwen4-tensorfold-spark
cd qwen4-tensorfold-spark
cp profiles/provisional/qwen38-flash-next-tp1-tool-full.env.example \
   profiles/provisional/qwen38-flash-next-tp1-tool-full.env        # TP1
cp profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env.example \
   profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env    # TP2, on both Sparks
```

The `.env` files are git-ignored. Fill every `<placeholder>`: `MODEL_DIR`, `PREPARED_DIR`, `STATE_DIR`, and for TP2
also `COST_CACHE_DIR`, `CUDA_CACHE_DIR`, `TP_MASTER`, `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`, each Spark with its own
values. Use absolute paths. Give each checkpoint its own prepared, state and cache folders; never reuse another pack's
prepared files, calibration or sessions. Leave the `TF_COMPUTE_LANE*` and `TF_RUNNER*` lines as they are.

Download the checkpoint at the profile's pinned revision, for example for TP1:

```sh
hf download TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --revision 2b170fa6309d5d1ee380b35636075fac7945f286 --local-dir "$MODEL_DIR"
```

### 2. Build the image

On a Spark (for TP2 build on one and copy the image to the other, or build on both):

```sh
scripts/build.sh tp1     # or: scripts/build.sh tp2
```

The script runs three steps; `scripts/build.sh tp1 source|native|image` runs one.

1. **source:** `git submodule update --init vendor/TensorFold`, a clean copy of it at the pinned commit in
   `build/PROFILE/src`, and every patch in `patches/series.PROFILE` applied with `git am`. It checks that the resulting
   tree is the measured one.
2. **native:** in that tree, with Zig 0.17.0 and no `-Dtarget` / `-Dcpu` (host code for the Spark's own CPU, as the
   measured images were built):

   ```sh
   zig build check-native-cuda-host -j1 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
   zig build native -j1 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast -Dstrip=true --prefix build/PROFILE/out
   zig build runner-frontend-check -j1 --prefix build/PROFILE/out
   zig build grammar -j1 -Dxgrammar=build/xgrammar --prefix build/PROFILE/grammar
   ```

   `build/xgrammar` is [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (`97787376`) with its submodules, cloned
   if missing.
3. **image:** the Docker layers in [`docker/`](docker), with no compile in Docker:
   - **Serving base** ([`docker/tp1.Dockerfile`](docker/tp1.Dockerfile) / [`docker/tp2.Dockerfile`](docker/tp2.Dockerfile)):
     NVIDIA's CUDA 13.0.1 base, NCCL, ibverbs, python3 (the compute lane's interpreter), `tensorfold-native` and
     `libtfgrammar`. For TP2 it also carries **Mia's captured Triton kernel set**: her engine replays these compiled
     kernels (`aot.json` and `cubins/`). They are copied byte for byte from her recipe's image (by default her published
     image `ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold`, by digest; `MIA_IMAGE=` names another) and
     checked against [`docker/mia-kernels.sha256`](docker/mia-kernels.sha256): 421 cubins and `aot.json`, the
     measured image's kernels (its copy came from her local rebuild and differs only in debug line bytes). Her
     LICENSE, NOTICE and CREDITS go to `/opt/mia-recipe`. The CUDA 13.3 compat driver comes from the same image.
   - **Runner layer** ([`docker/runner.Dockerfile`](docker/runner.Dockerfile)): clang-18 / lld-18 for the runner's
     frontend, the circuit `calc.tfpc` and the `graph-v7-powmod` bank (release `runner-assets-20261107`, downloaded by
     `scripts/fetch_runner_assets.py` and checked file by file against `config/runner-assets.sha256`). The build fails
     unless `tf-runner-frontend-check --require` computes the engine's probe cells.
   - **CUDA forward-compatibility layer** (TP1, [`docker/compat/`](docker/compat)): the driver files in
     `/opt/tensorfold/compat`, checked against `compat.sha256` and `compat.links`, and the entrypoint `tf-compat-entry`.
     The files come from `COMPAT_IMAGE` (default NVIDIA's PyTorch 26.07 container by digest, the base of Mia's image).

The finished image is `qwen4-tensorfold:tp1` or `qwen4-tensorfold:tp2`, the profile's `IMAGE`. On TP2 both Sparks
need the same image:

```sh
docker save qwen4-tensorfold:tp2 | ssh <second-spark> docker load
```

### 3. Serve on one Spark (TP1)

The profile gives the engine environment (the `TF_*` keys), the host folders and `SERVE_ARGS`. The measured boot
passed only the `TF_*` keys to the container and mounted the checkpoint read-only:

```sh
source profiles/provisional/qwen38-flash-next-tp1-tool-full.env
grep '^TF_' profiles/provisional/qwen38-flash-next-tp1-tool-full.env > tf.env
docker run -d --name qwen38fn --init --gpus all --ipc=host --network host --ulimit memlock=-1 --env-file tf.env \
  -v "$MODEL_DIR":/model:ro "$IMAGE" \
  serve /model --host 127.0.0.1 --port 8000 --context 65536 --thinking-budget 8192
```

The server listens on `127.0.0.1:8000`.

### 4. Serve on two Sparks (TP2)

On each Spark, with that Spark's own filled-in `.env`, make the engine env file (the profile's `TF_*`,
`TENSORFOLD_*` and `HF_HUB_OFFLINE` keys, as the measured boots passed them):

```sh
source profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env
grep -E '^(TF_|TENSORFOLD_|HF_HUB_OFFLINE=)' profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env > tf.env
mkdir -p "$COST_CACHE_DIR" "$CUDA_CACHE_DIR"
```

Start rank 1 on the second Spark first:

```sh
docker run -d --name qwen38fn-r1 --gpus all --ipc=host --network host \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" -e NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS" \
  -e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" \
  --env-file tf.env -v "$MODEL_DIR":/model:ro -v "$CUDA_CACHE_DIR":/cache \
  -v "$COST_CACHE_DIR":/root/.cache/tensorfold \
  "$IMAGE" serve /model --name Qwen3.8-Flash-Next --host 127.0.0.1 --port 8000 \
  --context 1048576 --parallel 16 --max-tokens 32768 --thinking --thinking-budget 8192 --kv-cache fp8 \
  --tp 2 --rank 1 --master "$TP_MASTER" --master-port "$TP_MASTER_PORT"
```

Then rank 0 on the first Spark, which serves on port 8000:

```sh
docker run -d --name qwen38fn-r0 --gpus all --ipc=host --network host \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" -e NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS" \
  -e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" \
  --env-file tf.env -v "$MODEL_DIR":/model:ro -v "$CUDA_CACHE_DIR":/cache \
  -v "$COST_CACHE_DIR":/root/.cache/tensorfold \
  "$IMAGE" serve /model --name Qwen3.8-Flash-Next --host 127.0.0.1 --port 8000 \
  --context 1048576 --parallel 16 --max-tokens 32768 --thinking --thinking-budget 8192 --kv-cache fp8 \
  --tp 2 --rank 0 --master "$TP_MASTER" --master-port "$TP_MASTER_PORT"
```

Each line reads that Spark's own `.env`: `NCCL_SOCKET_IFNAME` is its link interface, `NCCL_IB_HCA` its two RoCE rail
devices, and `TP_MASTER` the first Spark's address on the link (the same on both). The serve flags are the profile's
`SERVE_ARGS`. A rank with no kept lane-cost table in `COST_CACHE_DIR` times it
(about 40 s) and writes it; the boot log says `lanes costs (kept)` or `(measured)`. Delete the folder to time them
again. Startup (rank 1 launch to rank 0 ready) measured 69 s with the costs kept and 111 s without (on an
earlier pin, `4c4e966eb`). Follow it with
`docker logs -f qwen38fn-r0`.

### 5. Check it answers

```sh
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "What is 7^22 mod 1000003? Answer with the number."}],
  "max_tokens": 4096
}'
```

The first call lists the model. The second returns a chat completion: thinking is on, so the reasoning comes first
and the answer after it. On TP1 the served model name is the one `/v1/models` lists; use that name in `model`.

### Turn the lanes off

The compute lane and the runner are engine settings in the env file. To serve without them, leave their keys out of
`tf.env` and restart the containers, as the lanes-off measurements below did (the other keys unchanged):

```sh
grep -vE '^(TF_COMPUTE_LANE|TF_RUNNER)' tf.env > tf-lanes-off.env    # then --env-file tf-lanes-off.env
```

The same image serves either way.

## Measured results

DGX Spark; the TP2 runs had the thermal guard on (pause at 92 C). The check suite has 797 items; its lane categories
are arith (40), gsm8k (80), reason (40) and track (20). The computation-heavy subset is 120 tasks with a frozen grader,
thinking on.

### TP2: qwen38-flash-next-tp2-mia-tool-full

| Date | Engine | Result |
| --- | --- | --- |
| 2026-11-13 | `20c1fb1a3` (the tree `patches/series.tp2` makes on `aaac299d1`, recipe `f0002f9`) | release gates rerun on the image `scripts/build.sh tp2` builds: computation-heavy subset 92/120 at 731 tokens a task, median 12.1 s, every task's token count equal to `d306dc2fe`'s |
| 2026-11-13 | `20c1fb1a3` | long-integer reasoning set 9/9, no digit run past the exact value, every reply token-identical to `d306dc2fe`'s; thinking-off set 17/20, all 20 replies byte-identical to `d306dc2fe`'s; lane categories 180/180, all 180 replies byte-identical; thinking-budget probe 12/12; `tool_choice` required, named and any pass |
| 2026-11-13 | `20c1fb1a3` | the agent-client task (7^222 mod 1000003 and the sum of squares 1..2026 to a file, see `d306dc2fe` below) in 8 fresh opencode sessions, run twice: 7/8 and 7/8; the first 5 sessions (the form `d306dc2fe` ran) 10/10. Each miss is the model's own sum of squares written without a tool call; 7^222 mod 1000003 is exact in all 16. `d306dc2fe` on the same 8 sessions (2026-11-14): 7/8, its miss an empty reply |
| 2026-11-13 | `20c1fb1a3` | boot smoke, the image built with `scripts/build.sh tp2` and served with section 4's lines: ready 132 s after rank 0 started; section 5's curl answers 30584 (7^22 mod 1000003); thinking on, off, streamed and a client tool call pass; `tool_choice` required, named and any pass; both ranks on the CUDA 13.3 compat driver |
| 2026-11-12 | `d306dc2fe` (the tree `patches/series.tp2` made on `f212eeabd`, recipe `e2aca8b`) | an agent-client task writing 7^222 mod 1000003 and the sum of squares 1..2026 to a file, in opencode, 5 fresh sessions: 5/5 exact (22-66 s each); with `TF_RUNNER_INLINE_MARK=0` on the same image 4/5 |
| 2026-11-12 | `d306dc2fe` | long-integer reasoning set (7^222; 3^300 and 2^500; 13^150; three seeds each): 9/9 with no reasoning digit run past the exact value; with the mark off 8/9 (one reply ran away, 10,402 tokens) |
| 2026-11-12 | `d306dc2fe` | computation-heavy subset 92/120 at 731 tokens a task, median 12.2 s |
| 2026-11-12 | `d306dc2fe` | thinking-off set 17/20, all 20 replies byte-identical to the earlier pinned run, 0 runner cells |
| 2026-11-12 | `d306dc2fe` | lane categories 180/180; `tool_choice` required, named and any pass; a 12-prompt thinking-budget probe 12/12 |
| 2026-11-12 | `d306dc2fe` | a three-turn opencode build of a Python CLI package with tests: every turn exits 0, 73 tests pass, 99 requests all 200, slowest 89 s; boot 79 s |
| 2026-11-12 | `4c4e966eb` | startup (rank 1 launch to rank 0 ready) 111 s with lane costs timed, 69 s with them kept |
| 2026-11-11 | `188d9c63c` | computation-heavy subset 92/120 at 738 tokens a task, median 12.3 s; lanes off on the same recipe 58/120 at 6,038 tokens, median 141.0 s |
| 2026-11-11 | `188d9c63c` | thinking-off chats: 0 runner cells across 68 replies; decode equal to the same image without runner and lane (1.006x, paired) |
| 2026-11-10 | `2f77d333d` | Mia's decode protocol against her original image, same day |

Mia's decode protocol, tokens/s at 1 / 2 / 4 / 8 / 16 streams, 2026-11-10 and 11-11, compat driver on:

| Arm | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| her original recipe image (vision off) | 69.6 | 103.1 | 156.4 | 233.2 | 372.5 |
| the port, runner and lane off | 70.1 | 103.5 | 158.0 | 238.5 | 380.6 |
| this profile (rerun, both nodes at or below 86 C at each start) | 69.0 | 100.9 | 154.7 | 233.6 | 362.6 |
| this profile on `20c1fb1a3` (2026-11-13, the `f0002f9` image, fresh boot, both nodes at or below 86 C) | 69.8 | 102.1 | 158.9 | 235.9 | 383.2 |

Her published figures are 67.4 / - / 152.2 / - / 390.1. With the runner and lane, a decode round costs about 6% more
at 8 and 16 streams than the port without them. On `4c4e966eb` the profile decoded 69.97 / 155.68 / 362.16 at 1 / 4 /
16 streams.

### TP1: qwen38-flash-next-tp1-tool-full

| Date | Engine | Result |
| --- | --- | --- |
| 2026-11-11 | `c8e930375` | full check suite 736/797 (control 737), 487 tokens an item (control 586) |
| 2026-11-11 | `c8e930375` | computation-heavy subset 93/120 at 908 tokens a task (control 50/120 at 6,605); thinking-off set 18/20 |
| 2026-11-11 | `c8e930375` + compat | compat driver against host driver: speed score +5.5%; the same tokens on 116 paired requests, compat faster on all 116 |
| 2026-11-11 | `1a7ed13bb` | computation-heavy subset 93/120, 120/120 replies identical; thinking-off set 0 runner cells, 18/20 |
| 2026-11-12 | `501674d7f` | tool calls cut by `max_tokens` inside a value: 12/12 end cleanly; agent cells 40/40 token-identical to `1a7ed13bb`; check subset 68/72; thinking-off 18/20. The computation-heavy subset was not run on this commit. |
| 2026-11-13 | `b62876185` (the tree `patches/series.tp1` made on `f212eeabd`, recipe `e2aca8b`) | long-integer reasoning set (7^222; 3^300 and 2^500; 13^150; three seeds each): 9/9 with 0 digit runaways |
| 2026-11-13 | `b62876185` | computation-heavy subset 95/120 at 900 tokens a task |
| 2026-11-13 | `b62876185` | lane categories 179/180; check subset 68/72; thinking-off set 18/20, all 20 replies identical to `501674d7f`'s |
| 2026-11-13 | `b62876185` | a 12-prompt thinking-budget probe 12/12; `tool_choice` required, named and any pass; opencode agent turns pass |
| 2026-11-13 | `187e28a03` (the tree `patches/series.tp1` makes on `aaac299d1`, recipe `f0002f9`) | release gates rerun on the image `scripts/build.sh tp1` builds, every reply the same as `b62876185`'s: computation-heavy subset 95/120 at 900 tokens a task (all 120 token counts equal); long-integer reasoning set 9/9 with 0 digit runaways (all 9 token-identical) |
| 2026-11-13 | `187e28a03` | lane categories 179/180 and check subset 68/72, all 252 replies byte-identical to `b62876185`'s; thinking-off set 18/20, all 20 identical; thinking-budget probe 12/12; `tool_choice` required, named and any pass; opencode agent turns pass (the agent-client task 8/8) |
| 2026-11-13 | `187e28a03` | boot smoke, the image built with `scripts/build.sh tp1` and served with section 3's line: ready 181 s; section 5's curl answers 30584; thinking on, off, streamed and a client tool call pass; a computation prompt ran code in the lane (2 calls) and ended with its answer; `tool_choice` required, named and any pass; the CUDA 13.3 compat driver is on |

Decode on `c8e930375`: 48.8 / 47.8 / 53.8 tokens/s at 841 / 6k / 24k-token prompts, 85.1 with 4 at once. With the
compat driver: 53.6 / 49.5 / 55.3, 93.6 with 4 at once. Single cells move several percent between boots.

The full measurement notes, with every arm, are the comment blocks in each profile.

## Known limits

- **Text only on TP2.** Mia's patch 0009 (Python vision) is not ported. The TP2 port refuses `--vision`; her vision
  settings stay in the profile as measured.
- **No structured output on TP2.** The TP2 engine answers `response_format` (JSON object or JSON schema) and the
  `guided_*` fields with HTTP 400 ("not supported by this engine yet"). Forced tool calls (`tool_choice` required,
  named, any) work. TP1 serves both.
- **TP1 tool-full: long exact integers.** The 4-bit TP1 model opens its reasoning by quoting the request. When the
  request itself held `7^222 =`, the runner used to write the exact value into that quotation; the model then took the
  value for the user's text and could run on into long invented digit runs. The pinned engine abstains on an inline
  expression inside an open quotation (`TF_RUNNER_INLINE_QUOTED`, on by default), and the long-integer set passes 9/9
  with 0 runaways. The digit stop after an owned value and the 1,024-digit reasoning run cap are on, but no stop fired
  in the release window, so neither was exercised there. A run the model writes with no engine value before it has no
  engine-side stop.
- **Inline-value mark not final.** Both tool-full profiles set `TF_RUNNER_INLINE_MARK=1`: an engine-owned inline value
  ends in ` @engine`. Without it Qwen extended spliced big integers with invented digits and once overrode the engine's
  right value. The model sometimes glues a word to the mark (`@engineer`) in its reasoning; TP1 leaves that alone
  (`TF_RUNNER_MARK_STOP=0`) because cutting it cost a right answer. The default is still under decision.
- **Agent sessions miss about 1 in 8 on TP2.** In an opencode task that asks for 7^222 mod 1000003 and the sum of the
  squares 1..2026 in a file, without a shell command, three runs of 8 fresh sessions each scored 7/8, on `d306dc2fe`
  and on the current base alike. The misses: twice the model wrote a wrong sum of squares with no tool call; once a
  session ended with an empty reply. The modular power was right in every session that answered.
- **Runner in thinking chats only.** A thinking-off chat gets no runner session and no engine-owned values. The compute
  lane also stays out of `tool_choice` required or named calls and of requests with `max_tokens` under 1024.
- **Thinking budget.** Both tool-full profiles close a think block after 8192 tokens unless the request sets
  `thinking_budget`.
- **Not rerun on every pin.** Each profile's notes say which checks ran on which commit.
- **Heat.** Some 16-stream waves reached 92-95 C and clocks were capped; those runs are marked or excluded in the notes.
- **One pair of nodes, one window per arm.** Treat differences of a few percent, or of one or two items, as
  indicative.

## Qwen4 placeholders: NOT YET WORKING

**These profiles do not work.** Qwen has not announced a 28B / 125B / 256B lineup. As of 9 November 2026, Qwen 4 is
announced as in training and no Qwen4 checkpoint has been published. Nothing in these files has been booted, measured
or qualified; checkpoint, quant and context are `TBD`, and the architecture switches are intents no engine reads.

| Size (unconfirmed) | Placement (planned) | Profile |
| --- | --- | --- |
| Qwen4-28B | one Spark, BF16 first | [qwen4-28b](profiles/provisional/qwen4-28b.env.example) |
| Qwen4-125B | one Spark at NVFP4 if it fits; otherwise TP2 | [qwen4-125b](profiles/provisional/qwen4-125b.env.example) |
| Qwen4-256B | two Sparks, TP2 over RoCE, NVFP4 | [qwen4-256b](profiles/provisional/qwen4-256b.env.example) |

The work order, gates and risks are in [plans/PLAN.md](plans/PLAN.md). Each size's own work starts only when Qwen
publishes that checkpoint.

## Layout and checks

| Path | What it is |
| --- | --- |
| `vendor/TensorFold` | the engine: upstream TensorFold as a submodule, pinned (`config/tensorfold.repo`, `config/tensorfold.rev`), unmodified |
| `patches/` | the engine changes as a numbered `git format-patch` series; `series.tp1` and `series.tp2` say which apply for each profile and the tree each must give |
| `scripts/build.sh` | submodule, patches, native build, image layers |
| `docker/` | the image layers: serving bases, runner layer, CUDA compat layer, and the manifests of Mia's kernel set and the compat driver |
| `config/runner-assets.*` | the pinned runner-assets archive and its file list |
| `profiles/provisional/*.env.example` | the profiles; copy each to a git-ignored `.env` |
| `plans/` | the port plan |
| `scripts/check.sh` | this repo's checks: ruff, the publication scan, the links check, the tests |
| `scripts/check-public.sh` | refuses private addresses, host names, home paths, credentials and local agent files |
| `tests/` | pytest suite for the tooling, the profiles and the patch series |

The checks need Python 3.11+, ruff and pytest: `scripts/check.sh`. Put your own private names in
`.check-public.local` (git-ignored, one extended regex per line). `scripts/check-public.sh --history` also scans every
commit and its author address.

## Credits

- **TensorFold** upstream: https://github.com/ashhart/TensorFold (Apache-2.0). The engine is TensorFold; this repo
  pins it unmodified and carries its changes as patches.
- **Mia / MiaAI-Lab.** The TP2 engine and its serve settings are hers, from her dual-Spark recipe
  https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold (Apache-2.0), forward-ported onto this
  engine base. Her patches 0001-0008 and 0010-0012, including 0007 (the `tool_choice` call gate), are patches
  0002-0012 here, as her own commits (0001 adds her recipe's license notice); her LICENSE, NOTICE and CREDITS are in the patched tree's `LICENSES/` folder and
  in the TP2 image's `/opt/mia-recipe`. The captured Triton kernel set the TP2 image carries is hers, built by her
  recipe's image build and copied from her image. The CUDA 13.3 compat driver files both images carry are NVIDIA's,
  as her NGC-based image ships them. The decode protocol above is hers.

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights keep their own licenses.
