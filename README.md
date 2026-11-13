Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# Qwen3.8-Flash-Next (and later Qwen4) on TensorFold, NVIDIA DGX Spark

Serving profiles (env files) for the native TensorFold engine on one or two NVIDIA DGX Sparks. The working profiles
serve Qwen3.8-Flash-Next, the released preview of the Qwen4 architecture, with the engine's compute lane (the engine
runs exact computation for every thinking chat), the in-model runner on an exact circuit and graph bank, and the lane's
calls on that runner. The Qwen4 profiles are placeholders: no Qwen4 checkpoint has been published.

> **Work in progress.** Every profile is under `profiles/provisional/`. Results come from one Spark or one pair of
> Sparks, one boot per arm. No number here carries over to a Qwen4 size.

## Which profile

| Profile | Checkpoint (pack) | Context | Sparks | Engine family | What is on |
| --- | --- | --- | --- | --- | --- |
| [qwen38-flash-next-tp1-tool-full](profiles/provisional/qwen38-flash-next-tp1-tool-full.env.example) | `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP`, MLX 4-bit g32 with MTP | 65,536 | 1 | `flashnext` | compute lane and its server-side default, in-model runner, lane on the runner, CUDA 13.3 compat driver, an 8192-token default thinking budget, marked inline values |
| [qwen38-flash-next-tp2-mia-tool-full](profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env.example) | `nvidia/Qwen3.8-Flash-Next-NVFP4`, NVFP4 | 1,048,576 | 2 (TP=2, RoCE) | `flashnext`, Mia's engine (`TF_FLASHNEXT_ENGINE=mia`) | the same, plus her serve settings, an 8192-token default thinking budget, marked inline values |
| [qwen38-flash-next-tp1](profiles/provisional/qwen38-flash-next-tp1.env.example) | the TP1 pack | 65,536 | 1 | upstream `flashnext` | nothing: the plain boot plus two prefill chunk settings (the control) |

Pick TP1 if you have one Spark, TP2 if you have two linked Sparks. The control exists to compare against; it is not a
serving recommendation. Each profile pins its checkpoint revision in `CHECKPOINT` (`repo@revision`).

## Prerequisites

- **Hardware:** NVIDIA DGX Spark (GB10). The TP2 profile loads kernels for `sm121`. TP2 needs two Sparks.
- **Link (TP2 only):** a RoCE link between the two Sparks. The profile uses both RoCE rails (`NCCL_IB_HCA`), GID
  index 3 on both nodes and 4 NCCL channels. You need each node's link interface name, both rail device names and
  rank 0's address on the link.
- **Docker** with GPU access through the NVIDIA container runtime, on each Spark.
- **Driver:** the profiles were measured with the host's 580 kernel module. The tool-full images carry the CUDA 13.3
  user-mode driver (libcuda 610.43.02, taken from Mia's NGC-based image) and use it only while the kernel module is
  older than 610 and `cuInit` succeeds through it; otherwise the host driver. `TF_CUDA_COMPAT=0` forces the host
  driver. Each rank logs which one it uses.
- **Checkpoint:** download the exact `CHECKPOINT` revision from Hugging Face into `MODEL_DIR` (on both Sparks for
  TP2). The profiles set `HF_HUB_OFFLINE=1`.
- **Disk:** no disk figure has been measured for these profiles. Budget for the checkpoint, its own prepared folder,
  state folder and the runner images.
- **Toolchain to build images:** the engine repository's pinned Zig and Docker. The runner layers need
  no GPU to build.

## Quickstart

The engine is the public fork https://github.com/jayleaton/TensorFold: branch `flashnext-tp1-lanes` for TP1 and
`flashnext-mia-lanes` for TP2 (the TP1 branch is the TP2 one plus three runner commits). The exact repo and commit
each profile was measured on are its `ENGINE_REPO` and `ENGINE_REV`; build from those.

### 1. Make the env file

```sh
cp profiles/provisional/qwen38-flash-next-tp1-tool-full.env.example \
   profiles/provisional/qwen38-flash-next-tp1-tool-full.env        # TP1
cp profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env.example \
   profiles/provisional/qwen38-flash-next-tp2-mia-tool-full.env    # TP2, on both Sparks
```

The `.env` files are git-ignored. Fill every `<placeholder>`: `IMAGE`, `MODEL_DIR`, `PREPARED_DIR`, `STATE_DIR`, and
for TP2 also `COST_CACHE_DIR`, `TP_MASTER`, `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`. Use absolute paths. Give each
checkpoint its own prepared, state and cost folders; never reuse another pack's prepared files, calibration or
sessions. Leave the `TF_COMPUTE_LANE*` and `TF_RUNNER*` lines as they are.

### 2. Build the runner image

At `ENGINE_REV`, the image is built in layers:

1. **Native serving image** (`tensorfold-native`). For TP2 it is built for the Spark's own CPU (no `-Dtarget` or
   `-Dcpu`) and carries Mia's captured Triton kernel set, copied byte for byte from her recipe's image. *The build
   command for this image is not documented in this repo.*
2. **Runner frontend layer**, from the engine's `packaging/runner/README.md`:

   ```sh
   zig build runner-frontend-check
   packaging/runner/build.sh BASE_IMAGE TAG zig-out/bin/tf-runner-frontend-check
   packaging/runner/check-image.sh TAG
   ```

3. **Runner assets layer** (the circuit `calc.tfpc` and the `graph-v7-powmod` bank, from the `runner-assets-20261107`
   release, checked against `assets.sha256`):

   ```sh
   packaging/runner/build-assets.sh RUNNER_IMAGE TAG [ASSETS_DIR]
   ```

4. **CUDA forward-compatibility layer** (driver files in `/opt/tensorfold/compat`, entrypoint `tf-compat-entry`).
   *The build command for this layer is not documented in this repo.*

Set `IMAGE` to the final tag. On TP2 both Sparks need the same image.

### 3. Serve on one Spark (TP1)

The profile gives the engine environment (the `TF_*` keys), the host folders and `SERVE_ARGS`. The measured boot
passed only the `TF_*` keys to the container and mounted the checkpoint read-only:

```sh
grep '^TF_' profiles/provisional/qwen38-flash-next-tp1-tool-full.env > tf.env
docker run -d --init --gpus all --ipc=host --network host --ulimit memlock=-1 --env-file tf.env \
  -v "$MODEL_DIR":/model:ro "$IMAGE" \
  serve /model --host 127.0.0.1 --port 8000 --context 65536 --thinking-budget 8192
```

The server listens on `HOST:PORT` (default `localhost:8000`).

### 4. Serve on two Sparks (TP2)

Start rank 1 on the second Spark first, then rank 0. Both ranks use the profile's `SERVE_ARGS` plus the TP flags:

```sh
# rank 1, second Spark, first
<engine serve command> $SERVE_ARGS --tp 2 --rank 1 --master $TP_MASTER --master-port $TP_MASTER_PORT
# rank 0, first Spark, then (serves on PORT)
<engine serve command> $SERVE_ARGS --tp 2 --rank 0 --master $TP_MASTER --master-port $TP_MASTER_PORT
```

`SERVE_ARGS` is `--context 1048576 --parallel 16 --max-tokens 32768 --thinking --thinking-budget 8192 --kv-cache fp8`.
Mount `COST_CACHE_DIR` at `~/.cache/tensorfold` inside each rank's container (the image runs as root). A rank with no
kept lane-cost table times it (about 40 s) and writes it; the boot log says `lanes costs (kept)` or `(measured)`.
Delete the folder to time them again. *The full `docker run` line (GPU and RDMA device flags, network mode, mount
points, the serve command) is not documented in this repo yet.*

## Measured results

DGX Spark; the TP2 runs had the thermal guard on (pause at 92 C). The check suite has 797 items; its lane categories
are arith (40), gsm8k (80), reason (40) and track (20). The computation-heavy subset is 120 tasks with a frozen grader,
thinking on.

### TP2: qwen38-flash-next-tp2-mia-tool-full

| Date | Engine | Result |
| --- | --- | --- |
| 2026-11-12 | `d306dc2fe` (= the pinned public twin `9c6c3d04c`) | an agent-client task writing 7^222 mod 1000003 and the sum of squares 1..2026 to a file, in opencode, 5 fresh sessions: 5/5 exact (22-66 s each); with `TF_RUNNER_INLINE_MARK=0` on the same image 4/5 |
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
| 2026-11-13 | `b62876185` (= the pinned public twin `9b1a0c175`) | long-integer reasoning set (7^222; 3^300 and 2^500; 13^150; three seeds each): 9/9 with 0 digit runaways |
| 2026-11-13 | `b62876185` | computation-heavy subset 95/120 at 900 tokens a task |
| 2026-11-13 | `b62876185` | lane categories 179/180; check subset 68/72; thinking-off set 18/20, all 20 replies identical to `501674d7f`'s |
| 2026-11-13 | `b62876185` | a 12-prompt thinking-budget probe 12/12; `tool_choice` required, named and any pass; opencode agent turns pass |

Decode on `c8e930375`: 48.8 / 47.8 / 53.8 tokens/s at 841 / 6k / 24k-token prompts, 85.1 with 4 at once. With the
compat driver: 53.6 / 49.5 / 55.3, 93.6 with 4 at once. Single cells move several percent between boots.

The full measurement notes, with every arm, are the comment blocks in each profile.

## Known limits

- **Text only on TP2.** Mia's patch 0009 (Python vision) is not ported. The TP2 port refuses `--vision`; her vision
  settings stay in the profile as measured.
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
| `config/tensorfold.repo`, `config/tensorfold.rev` | the engine repo and the commit these recipes pin |
| `profiles/provisional/*.env.example` | the profiles; copy each to a git-ignored `.env` |
| `plans/` | the port plan |
| `scripts/check.sh` | every check: ruff, the publication scan, the links check, the tests |
| `scripts/check-public.sh` | refuses private addresses, host names, home paths, credentials and local agent files |
| `scripts/check_links.py` | every relative Markdown link and heading resolves |
| `tests/` | pytest suite for the tooling and the profiles |

The checks need Python 3.11+, ruff and pytest: `scripts/check.sh`. Put your own private names in
`.check-public.local` (git-ignored, one extended regex per line). `scripts/check-public.sh --history` also scans every
commit and its author address.

## Credits

- **Mia / MiaAI-Lab.** The TP2 engine and its serve settings are hers, from her dual-Spark recipe
  https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold (Apache-2.0), forward-ported onto this
  engine base. Her patches 0001-0008 and 0010-0012, including 0007 (the `tool_choice` call gate), are in the engine
  branch as her own commits; her LICENSE, NOTICE and CREDITS ship in its `LICENSES/` folder. The CUDA 13.3
  compat driver files and the Triton kernel set come from her image, and the decode protocol above is hers.
- **TensorFold** upstream: https://github.com/ashhart/TensorFold (Apache-2.0).

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights keep their own licenses.
