# Qwen4 on TensorFold: port plan for the full architecture

Status: plan only. No engine feature code has been written and no GPU has run. Facts below come from the research
report on Qwen4 (QWEN4-RESEARCH.md, 9 November 2026). Anything it could not verify stays `TBD (research)`.
The component inventory this plan works from is `docs/qwen4-arch/ARCHITECTURE.md` in the engine repo, cited below as
"component #N".

## 0. Starting point

### What the research found

- **The 28B / 125B / 256B lineup is not verified.**
  - Alibaba announced Qwen 4 as in training (22 September 2026) without sizes, dates or artifacts.
  - The official Hugging Face and GitHub organisations had no Qwen4 checkpoint or repository on 9 November.
  - In this plan the three sizes are planning placeholders. No recipe is named after a checkpoint until its artifacts exist.
- **The released precursor is Qwen3.8-Flash-Next** (`Qwen/Qwen3.8-Flash-Next`, revision `812d96e1`). Qwen calls it an early release of the Qwen4 architecture. Its parts:
  - a 125B MoE backbone with about 6B active parameters;
  - 512 routed experts, top 10 plus 1 shared;
  - a 51B n-gram table and a 4B MTP head, about 180B all-in;
  - 36 GDN layers and 12 QSA layers, `hc_count=4`;
  - 262,144 native context, with thinking on by default;
  - a vision tower;
  - the Qwen Community License 1.0.
- **Upstream already serves that precursor** through the `flashnext` family (`qwen4_exp`), not through `qwen4`.
- **Spark fit** is weight arithmetic at 114 GiB usable per Spark, not a measurement:
  - 28B all-in fits one Spark at BF16 (52 GiB of weights).
  - 125B all-in fits one Spark at NVFP4 (65 GiB of weights). But about 180B all-in at NVFP4 (94 GiB) only fits with a short context and the prompt cache off.
  - 256B needs TP2 at NVFP4 (about 67 GiB per rank for an even split of the weights).

### Upstream `main` (`c5a044877`, this repo's `main`)

The `qwen4` and `qwen4_text` family is there as **dense native preparation**:

- It has GDN, QSA, HC, PLE and n-gram embeddings, with Metal and CUDA kernels and the shared lane adapter.
- Prompt-cache snapshots go through `lane_snapshots.zig`.
- Its checks are synthetic device gates only. No released checkpoint has been loaded.

Config admission (`families/qwen4/config.zig`) refuses these by name:

- MoE fields;
- a positive MTP head;
- vision;
- `rope_scaling`;
- unknown fields.

The CUDA kernels also require:

- hidden size divisible by 256;
- GDN head dim 128;
- convolution length 4;
- attention head dim 256;
- index dim 128;
- at most 16 rows per stream window and 128 per batch.

Two more facts:

- `native/cuda_refusal.zig` refuses `qwen4` at `tp=2`, and `families/qwen4/native_host.zig` refuses any `tp != 1`.
- The adapter has no EXL3, and refuses FP8 KV and prompt INT8.
- CUDA 13.0 / sm_121 resource receipts exist (`core/contract/measurements/qwen4_130_sm121.zig`).
- Weights may be BF16, MLX affine 4/8-bit, ModelOpt FP8 or ModelOpt NVFP4.

### Upstream `flashnext` (`qwen4_exp`, Qwen3.8 Flash Next)

This is the same mixer family plus **MoE, MTP chains, CUDA graphs and one or two CUDA ranks**. The `qwen4` family
already shares its graph (`families/flashnext/forward_graph.zig`), which owns the dense *or routed* feed-forward boundary.

### The architecture

It is almost entirely shared-core code on the DSV41 branch (`7b79ef92d`). The lane-only part already exists as a
three-PR upstream cut on `1fedcfcb6`, which is 431 commits behind current main:

- `lane/deadline-s` `5488530e6`
- `lane/compute-sandbox` `baab0d1bd`
- `lane/compute-lane` `23eb9e67d`

Those are local branches and are not on GitHub yet.

## 1. Principle: move it into shared core once, then bind each family

Do not copy DSV41 code into `families/qwen4`. Every architecture component that is generic (component table rows
1–14 and 16) lands once in core or the server behind a small per-family binding. That way Qwen4, DSV41 and any later
family inherit it.

| Component (ARCHITECTURE.md #) | Shared home | Per-family binding |
| --- | --- | --- |
| Sandboxed executor (#1) | `core/compute_sandbox*` (upstream PR B) | none |
| Compute lane hold and resume (#2) | `core/compute_lane`, `engine_hold`, `lanes/hold`, `server/compute_*` (upstream PR C) | `serve_engine`: env prefix, `resume_held`, `think_open` |
| `deadline_s` (#3) | `server/request_deadline` (upstream PR A) | none |
| Server default and `compute_executed` (#4, #5) | `server/compute_turns` | the family's built-in tool name and type, its `default_min_tokens`, its variable prefix |
| In-thinking runner, frontend, circuit, graph bank (#6–#9) | `core/runner_*`, `core/lanes/runner_*` | think markers, prefix, the guide file |
| Trust guide (#11) | `server/runner_guide` with a per-family guide file pinned by sha | the guide text, measured per family |
| S3 (#12) | `core/compute_circuit*` | none beyond #2 and #6 |
| Owned bytes (#13) | `server/runner_owned`, `runner_projection` | strip points through the server's tool-format interface, not DSML |
| Sessions RAM and NVMe (#14) | upstream `src/sessions` and the prompt cache | the family codec and `validate` |
| TP settings digest (#16) | `fabric/tp` (`bc7846ef3`) | none |

**Variable names.** Core already takes `comptime prefix`, so a family-neutral spelling costs nothing. The proposal is
`TF_COMPUTE_LANE*` and `TF_RUNNER*` for new families. The DSV41 binding keeps reading `TF_DSV41_*`, because those
names are deployed in the published DSV41 recipe. The maintainer decides this (question 1).

## 2. Phases

The plan is conditional on the final artifacts.

- **Phase 0** needs no Qwen4 release and starts now.
- **Phases 1–3** start per size only when Qwen publishes that checkpoint.

### Phase 0: the architecture into shared core, proved on a model that exists now (CPU now, then one Spark)

| Step | Work | Depends on | Size |
| --- | --- | --- | --- |
| C1 | Push the `lane/*` stack to this repo as reference branches. Rebase PR A, B and C onto current `main` (431 commits of drift, mostly the server split). Make PR B compile out cleanly on macOS. | none | M (2–3 days) |
| C2 | Generalise the server default (the default injection): per-family tool name and type, prefix and `default_min_tokens`. Keep the DSV41 binding byte-identical (its tests unchanged). | C1 | S (1–2 days) |
| C3 | Generalise owned-byte strips through the tool-format interface; DSML becomes one implementation. | C2 | M |
| C4 | Port the in-thinking runner (scanner, holds, delivery, frontend, circuit loader, graph bank, receipts) onto current main as a shared module, with a family binding API. This is the large one: about 180 files and +45k lines on the branch. | C1 | XL (2–3 weeks; split into the reviewable PRs in §6) |
| C5 | Runner image and assets for the new build: the same `calc.tfpc` and v7 bank manifest; re-host or point at the DSV41 asset release | C4 | S |
| C6 | S3 on the shared runner | C4 | S |
| P0 | Bind C1–C6 to the proving model (below). Take its plain-engine baseline first, then the architecture arms: real-pack boot plus smoke, dashboard kit at or above baseline, opencode kit, runner gates. Measure that model's own trust guide. | C1–C4, one Spark window | M on top of C |
| P1 | The same binding on DSV41 against the generalised core: a DSV41 smoke and its unchanged tests, proving nothing regressed for the shipped recipe | C2, C3, C4 | S |

**Recommended proving model: Qwen3.8-Flash-Next through the existing `flashnext` (`qwen4_exp`) adapter.**

- It *is* the Qwen4 architecture preview: GDN plus QSA, HC 4, the n-gram PLE table, MoE (512 experts, top 10 plus 1 shared) and MTP. A binding proved there is what a Qwen4 checkpoint will need.
- Upstream already serves it natively on one GB10 with CUDA (MLX 4-bit group 32 with MTP), so P0 adds only the architecture binding, not model work.
- Its template turns thinking on by default (efforts low / medium / xhigh), which the server default's thinking gate and the in-thinking runner both depend on.
- The maintainer's `jayleaton/qwen38-flash-next-exl3-spark` recipe gives an outside comparison on the same hardware. It is a reference, not a Qwen4 or TensorFold claim.

Watch for two things on this model:

- **Memory:** the 51B n-gram table shares unified memory with everything else. Use the upstream recipe's published checkpoint and its own startup budget.
- **Licence:** the preview is under the Qwen Community License. Read it before any image or pack bundles its files.

**Qwen3.8-27B (`qwen3_5` / `qwen27`) is the second choice.** It fits one Spark at any format and has mature CUDA support, but it is a different architecture: GDN with ordinary full attention, no QSA, HC, PLE or MoE. Use it only as a later cross-family check that the bindings are generic, not as the proving model.

### Phases 1–3: per size, when its artifacts drop

Each size runs the same sequence. Its steps are gated on the official checkpoint being public; until then the size is
a placeholder (§0).

1. **Facts (M0).** Record the model ID, revisions, `config.json`, the tensor inventory (count backbone, lexical tables, MTP and vision separately), tokenizer, chat template, think markers, tool-call format, published quant layouts and licence.
2. **Admission check against the real config first (M1, CPU only, no weights).**
   - Run the released `config.json` and safetensors index through `qwen4` admission and through `flashnext` admission.
   - Record every refused field by name.
   - This decides the family: dense `qwen4`, routed `qwen4` via the shared boundary, or `flashnext` if the schema is Flash-Next-like (question 3).
   - Never change `model_type` or delete fields to force a config through.
3. **Close only the gaps that config declares** (§3), each with its own host tests:
   - MoE routing (M4);
   - MTP (M5);
   - TP2 (M6);
   - quant layouts;
   - scaled RoPE;
   - a vision refusal.
4. **Plain engine on the GPU (M2):**
   - real-pack boot plus smoke;
   - exactness contracts;
   - teacher-forced fidelity;
   - speed and memory receipts;
   - the dashboard-kit **baseline**.
5. **The architecture (M3):** the Phase 0 bindings plus a trust guide measured on that checkpoint, then every gate in §4.

| Phase | Size | Placement (research arithmetic; replaced by the real load plan) | Expected gaps after admission | Size of work |
| --- | --- | --- | --- | --- |
| 1 | 28B | one Spark, BF16 first as the numerical reference | none if dense preview-style and within the kernel shapes; otherwise MTP / vision refusal / RoPE | M (fits) to L (new geometry) |
| 2 | 125B | one Spark at NVFP4 if the all-in count fits; TP2 (M6) only if it does not | MoE (M4), MTP (M5), PLE table placement | L |
| 3 | 256B | two Sparks over RoCE (TP2, M6) at NVFP4; a single rented GPU is the fallback | everything is unknown; TP2 is certain new qwen4 work | L–XL |

| Step | Work | Size |
| --- | --- | --- |
| M4 | Open the `qwen4` routed feed-forward boundary using `flashnext`'s MoE path, without its fixed geometry | L |
| M5 | MTP head: reuse Flash Next's MTP config and chain drafting through the shared lane core | M–L |
| M6 | TP2 for qwen4: lift the `tp != 1` refusal, family TP split, RoCE boot with the recipe's TP2 settings, exactness at two ranks | L |
| M9 | NVMe session tier for qwen4, with a forced-NVMe gate counting disk resumes | S–M |

## 3. What Qwen4 needs first (research gap matrix)

For every size, the first need is the same: official artifacts. Then real-config admission and checkpoint fidelity
(teacher-forced logits, plain vs drafted, prompt cuts, fresh vs resumed, concurrent vs solo, cancel and reuse), then
GB10 receipts.

| Size | Conditional architecture gaps (only if the checkpoint declares them) | Placement and format |
| --- | --- | --- |
| 28B | If it is dense in the preview style: check the HC, GDN, QSA and PLE geometry and every field. Add learned MTP, vision / M-RoPE and scaled RoPE if shipped; MoE only if declared. | One Spark, BF16 first as the numerical reference; FP8 or NVFP4 after qualification |
| 125B | If it is Flash-Next-like: routed admission, expert loader, router and FFN boundary (from `flashnext`, without its fixed constants), MTP, PLE table placement and format; vision and long-context scaling if offered | One Spark at NVFP4 if all-in 125B; BF16 and FP8 do not fit 114 GiB. About 180B all-in needs short context, cache off, or TP2. |
| 256B | Everything unknown. TP2 rank ownership, sharded weights, collectives, state placement, failure handling and two-rank parity are all missing from the adapter. | TP2 at NVFP4. One Spark at about 3 bpw would need EXL3, which this adapter lacks. |

Engine state behind each gap:

| Need | Upstream `qwen4` today | Nearest reuse |
| --- | --- | --- |
| Real-config admission | dense synthetic fields only; refuses MoE, MTP, vision, `rope_scaling`, unknown fields | `qwen4/config.zig`, `flashnext/config.zig` |
| MoE routing | refused (`graph.zig`, `gpu_ops.zig` `Qwen4RoutedLayer`) | `flashnext/forward_graph.zig` routed boundary, `cuda_moe_native.zig` (whole-expert partition), without the fixed geometry in `cuda_prompt_wt.zig` and `cuda_moe.zig` |
| MTP | refused | `flashnext` `Mtp` config and chains; context-copy proposals already exist |
| TP2 | refused (`tp != 1`) | `flashnext` two-rank CUDA (experimental; prompt cache off with MTP at TP2), shared `fabric/tp` |
| Quant format | BF16, MLX 4/8, ModelOpt FP8 (scalar scale), ModelOpt NVFP4 (group 16); no EXL3 | shared quant keys; EXL3 from `qwen3_5` / `qwen27` is separate integration work |
| Long context | default RoPE only | YaRN or scaled RoPE is new work, with memory admission |
| Vision | refused | none in the family; keep refusing it with a clear message |
| Sessions | in-process prompt cache and snapshots | upstream `src/sessions` for the NVMe tier (M9) |

## 4. Gates (unchanged from DSV41)

1. **CPU gates on every change.** These run on a Spark in a 2 GiB scope, never on the dev box:
   - `zig build test test-golden`;
   - `python3 tools/zig/lean_check.py` with 0 new findings over the pinned baseline;
   - `zig fmt --check`;
   - `python3 tools/pr_check.py --base origin/main`;
   - gitleaks.
2. **Review.** Every change gets a fresh reviewer round until clean before it counts as done.
3. **Real-pack boot plus smoke before any review sign-off or bench.** CPU gates alone missed a pack-planner bug on DSV41.
4. **Exactness** (CONTRIBUTING.md), at the same checkpoint, backend and settings:
   - drafted equals plain;
   - resumed equals fresh, with the NVMe tier forced;
   - concurrent equals solo;
   - prompt-cut seams;
   - cancel and retry.
5. **Dashboard kit at or above baseline.** Every category is at or above the same size's plain-engine baseline, with 0 empty, 0 leaked and 0 garbled replies. The baseline is taken on the plain engine (P0 for the proving model, M2 for each size), before any architecture arm.
6. **opencode kit:**
   - strict compute answers are all correct;
   - engine execution is positive on default-tools compute tasks;
   - no denied tool completes;
   - regression replies pass with no 5xx;
   - the owned-record oracle is regenerated on Qwen4 and passes.
7. **Runner gates:**
   - mutation-based ownership against fresh CPython;
   - sequence packets only through `route.Dispatch`;
   - the proof-host test count rises;
   - owned values are checked against the parsed span;
   - the abstention audit is re-run on Qwen4 reasoning text.
8. **Speed and memory receipts** per CONTRIBUTING.md: cold prefill at 2k/8k/32k/64k, decode, memory, the same machine before and after. No DSV41 number is carried over as a Qwen4 claim.

## 5. Risks

| # | Risk | Effect | Mitigation |
| --- | --- | --- | --- |
| 0 | The 28B / 125B / 256B lineup is unannounced, so the sizes may never ship as named | Recipes built for checkpoints that do not exist | Treat the sizes as placeholders; Phase 0 builds and proves the architecture on Qwen3.8-Flash-Next; per-size phases start only on real artifacts |
| 1 | The released Qwen4 geometry does not match the dense preparation (kernel shapes, MoE, new fields) | M1/M2 become kernel work, not admission | Read the research gap matrix first; check every config field and tensor name on the CPU before any GPU window |
| 2 | Runner port size (C4: about 180 files) against 431+ commits of upstream drift | Weeks of rebase; review load | Split into the PRs in §6; land the lane (C1–C3) first, since it does not need the runner |
| 3 | 125B "plus tables" (about 180B all-in) and 256B fit | 125B may need TP2; 256B may not fit at a quality-preserving quant | Inventory the real format map tensor by tensor; per-rank load plans before code; a single rented GPU as the fallback |
| 8 | Licence: the precursor is under the Qwen Community License 1.0, and Qwen4's licence is unknown | A recipe, image or pack may not be redistributable as planned | Read each checkpoint's licence before publishing anything that bundles its files |
| 4 | qwen4 TP2 is new family work (the refusal exists for a reason) | 256B slips (and 125B, if it needs TP2) | Reuse the Flash Next two-rank path and shared fabric; exactness at two ranks before speed |
| 5 | V4.1-tuned behaviour (trust guide, firing, abstention, `default_min_tokens`) regresses Qwen4 quality | Dashboard kit below baseline | Arms with and without each piece; the baseline taken before any architecture |
| 6 | Generalising the server default changes DSV41 bytes | Regression in the shipped DSV41 recipe | C2/C3 keep DSV41's tests byte-identical; a DSV41 smoke on the generalised build |
| 7 | Spark contention with the TP2 priority work | No GPU windows | All C work and M0/M1 are CPU-only; GPU steps queue through the node turn order |

## 6. Upstream candidates (small PRs on current main, Mac-compilable, reuse core)

| PR | Content | Notes |
| --- | --- | --- |
| U1 | `deadline_s` (PR A) | independent |
| U2 | Sandboxed executor (PR B) | Linux-only syscalls compiled out on macOS |
| U3 | Compute lane hold and resume with a family binding (PR C) | on U2; the DSV41 binding as the first user |
| U4 | Server default generalised (C2) plus `compute_executed` | on U3 |
| U5 | TP bootstrap carries a family's settings digest (`bc7846ef3`) | check it is not already upstream |
| U6 | qwen4 real-config admission for a released checkpoint (M1) | needs a maintainer-agreed family gate (CONTRIBUTING: open an issue first) |
| U7 | qwen4 routed feed-forward via the shared Flash Next boundary (M4) | arithmetic change: needs an issue and gate |
| U8 | qwen4 two-rank CUDA (M6) | after a two-rank receipt |
| U9 | Runner core, in parts: scanner and holds; circuit loader; graph bank; frontend; owned bytes | large; open an issue with the maintainer before the first part |

Upstream stacks stay at most two deep. The maintainer approves every push and PR.

## 7. Open questions for the maintainer

1. Variable names: should new families use `TF_COMPUTE_LANE*` and `TF_RUNNER*` while DSV41 keeps `TF_DSV41_*`? Or should every family use its own prefix (`TF_QWEN4_*`)?
2. Should the runner go upstream at all (U9), or stay in the maintainer's fork while the lane (U1–U4) goes up?
3. If the MoE sizes match Flash Next's architecture, admit them into `flashnext` (fast) or grow `qwen4` (one family for all sizes)?
4. 256B placement: two Sparks at NVFP4 (the research default), or a single rented GPU (phase 2)?
5. Where should the runner assets live for Qwen4: point at the DSV41 recipe's release, or re-host the same bytes?
6. Proving model for Phase 0: Qwen3.8-Flash-Next through `flashnext` (recommended), or Qwen3.8-27B?
7. The sizes: keep 28B / 125B / 256B as the repo's placeholders, or rename the recipes once Qwen announces the real lineup?
