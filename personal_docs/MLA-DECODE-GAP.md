# MLA decode on Trainium — the shape of the problem

Written 2026-09-25 against fork point `f8abae6`. Companion to
`GLM53-FLASH-FRAMEWORK-GAP.md`, which covers the framework surfaces; this covers the
kernel.

**Sources.** `nkilib` source checkout at
`/Users/aman/code/aws_infer/third_party/nki-library` @ `92d11f6`
(`src/nkilib_src/nkilib/`), including its integration tests. vLLM's MLA backends from
the local checkout at `/Users/aman/code/infra/mulgpu/third_party/Notvllm` @
`36fa72d2d0`. The plugin at our fork point. Upstream PR #40 fetched as
`refs/pull/40/head` = `254b0ee`.

**Nothing here has been executed.** No kernel was run, no simulator, no device. Every
claim is read from source or from nkilib's own test parameters. `[unverified]` marks
what source could not settle.

---

> **Update 2026-09-25 (2) — tier 1 ran too.** The **aliasing question is answered
> decisively: one latent tensor as both `k_prior` and `v_prior` gives bit-identical
> output** (0.000e+00, every config), and at `tp_k_prior=True` the two have identical
> shapes — so §2's one-tensor latent cache is sound and needs no second layout. The
> semantic mapping holds (nkilib's decode reference computes MLA absorbed attention,
> corr ≥ 0.9975 against an independent fp32 golden), but **reference-vs-reference is the
> wrong numerics gate**: `attention_tkg_torch_ref` faithfully emulates bf16 hardware
> arithmetic while the MLA ref computes fp32, so they differ by 10-30x bf16 epsilon by
> construction. The right gate is kernel-vs-its-own-reference, which upstream already
> runs at `d_head=512`. **The NKI kernel itself was never executed at `q_head=64`.** A
> new unquantified risk: whether bf16 latent decode is adequate at 1M context.
>
> **Update 2026-09-25 — the spike ran, at a cheaper tier than this document planned.**
> `q_head = 64` at `d_head = 512` **validates**, as predicted, at `s_active` 1 and 2 and
> across `bs` 1-64; DeepSeek's 576 is rejected with exactly the predicted assert.
> `kernel_assert` is a plain Python assert, so the whole §6.3 table is reachable by
> calling `_compute_tile_params` directly — no simulator, no compile, no device. Full
> result, including two predictions of mine that were wrong, in
> `dev/progress/2026-09-25-mla-decode-spike-prediction.md`. §6.2's QK-swap paragraph is
> corrected in place. Numerics (tier 1) remain unrun.

## 0. Verdict

1. **GLM-5.3-Flash's MLA fits inside an existing decode kernel's supported range, and
   DeepSeek's does not.** That is a fact about model selection, not just about
   kernels, and it is §1.
2. **Neither half of the problem is greenfield.** The absorbed-latent MLA algebra is
   implemented in nkilib's prefill kernels; the decode-side machinery — paged block-KV,
   context-parallel sharding, distributed softmax correction — is implemented in its
   TKG attention block, which the plugin already wraps. §2.
3. **`d_head = 512` is a tested configuration**, not merely a documented one:
   `test_attention_tkg.py` has a dedicated section for it covering block KV, FP8 KV and
   a production-shaped case. §2.4.

So STRATEGY.md §3.4's "no MLA decode kernel exists anywhere ... the deepest hole on our
roadmap" is literally true and misleading as a risk statement. The right statement is:
*no Neuron MLA decode kernel exists, but both halves do, and the join is a spike before
it is a port.*

---

## 1. Why NoPE decides this, and why it is a selection fact

MLA absorbed decode is mathematically **MQA with one KV head whose head dimension is
the latent width**. The query has absorbed `W_uk`, so attention runs directly against
the latent; the latent serves as both K and V. So the only question that matters for
reuse is: what latent width does the existing decode kernel accept?

Two independent limits in `nkilib/core/attention/attention_tkg.py`:

```
_MAX_D_HEAD = 512                                     (line 52, asserted at 1222)

kernel_assert(cfg.d_head % TC.p_max == 0 or cfg.d_head <= TC.p_max,
    "d_head must be <= p_max or a multiple of p_max (ragged d_head not yet
     supported)")                                     (lines 1305-1309)
```

Now the two models:

| | latent width | ≤ 512? | multiple of 128? |
|---|---|---|---|
| **GLM-5.3-Flash** (NoPE) | `kv_lora_rank` 512 + `qk_rope_head_dim` **0** = **512** | yes | yes, 4×128 |
| **DeepSeek-V3.2 / GLM-5.3** | `kv_lora_rank` 512 + `qk_rope_head_dim` 64 = **576** | **no** | **no, 4.5×128** |

**DeepSeek's 576 fails both checks; GLM-5.3-Flash's 512 passes both.** It is not a
near miss in one dimension — 576 exceeds the maximum *and* is ragged against the
128-wide partition tile.

A third alignment, from the test file's own comment on the d_head=512 cases
(`test_attention_tkg.py:998`): *"d_head > 128 requires PE transpose path, **fuse_rope
not supported**"*. The single documented incompatibility at `d_head > 128` is fused
RoPE — and a NoPE model has no RoPE to fuse. The restriction cannot bind us.

And a fourth, from the plugin side. `_can_use_attention_block_kernel`
(`functional/attention/attention_decode.py:1211-1258`) rejects `kv_heads > 1` unless
the caller passes a 3-D per-head block table. MLA is MQA, `kv_heads == 1`, so it takes
the simple 2-D-table path.

### The consequence Aman should weigh

The roadmap ordered GLM-5.3 (`glm_moe_dsa`, 78 layers, all MLA+DSA) *before*
GLM-5.3-Flash on the theory that GLM-5.3 rides AWS's DeepSeek kernel roadmap. On the
decode side that ordering is inverted: **GLM-5.3 and every DeepSeek-family model need
a 576-wide decode path that does not exist and is ragged against the hardware tile,
while GLM-5.3-Flash needs 512, which is supported and tested.**

This does not by itself reorder the roadmap — GLM-5.3-Flash is harder in other
respects (mHC, KDA, the indexer, four cache kinds) — but "Flash is the harder model"
is false for MLA decode specifically, and the gap is not small.

---

## 2. Neither half is greenfield

### 2.1 The algebra: already absorbed, already referenced

`nkilib/experimental/mla/deepseek/` is prefill-only, as STRATEGY.md says. What
STRATEGY.md does not record is *which form* of MLA it implements. From
`mla_qkv_cte_torch.py`'s module docstring:

> "PyTorch reference for the **absorbed-latent** MLA QKV CTE kernel. ... the per-head
> absorption matmul `q_nope @ W_uk` (bf16), and the latent KV path
> `c_kv = RMSNorm(kv) * gamma` plus `k_pe` RoPE (**no wkv_b matmul**)."

Its outputs are `q_lift[B, S, n_heads, kv_lora_rank]` and `c_kv[B, S, kv_lora_rank]`.

dev1's stub (`dev/progress/2026-09-25-mla-decode-gap-stub.md`, §37) hypothesised that
decode would need "the absorbed-projection trick — `W_kv_b` folded into the query so
attention runs against the 512-wide latent directly, instead of materialising 64×512 K
and V per token". **That is exactly what this kernel already does**, in the projection
stage, with a numerics-matching torch reference. The conceptually hard part of MLA
decode is not ahead of us.

### 2.2 The three prefill stages

| kernel | lines | computes |
|---|---:|---|
| `mla_qkv_cte` | 1384 | hidden → `q_lift`, `q_pe`, `c_kv`, `k_pe`; MX fp8 projections, bf16 absorption |
| `mla_sparse_attention_cte` (KERNEL A) | 642 | absorbed-latent attention; **dense and topk-sparse modes**; causal mask via `q_pos_offset` |
| `mla_vup_oproj_cte` (KERNEL B) | 641 | per-head MX V-up (L→`d_v`), then MX o_proj |

Shared and reusable: `mla_common_cte.py` (1566 lines of tiling/PSUM machinery,
exporting `_H_PACK`, `_K_CHUNK`, `_MM1_TILE`, `_P_MAX`, `_SM_TILE`) and
`mla_validate_params.py` (471 lines of shape validation). Each stage has a
`*_torch.py` reference that replays the hardware pipeline, so any decode work has a
CPU oracle from day one.

What is prefill-shaped about them, precisely:

- **Query-sharded, KV-gathered.** KERNEL A's docstring: *"S-sharded across cores. Under
  Context Parallelism the framework gathers the latent KV (`c_kv` / `k_pe`) to the full
  `S_kv` before calling this kernel."* Decode has one query and must do the opposite.
- **No block table.** `c_kv_hbm` is a contiguous `[B, S_kv, L]`.
- **`B == 1` and `S` divisible by core count** (docstring), i.e. it assumes a long
  query axis to divide.
- MX activation quantisation sized with `S` in the free dimension.

### 2.3 The decode machinery: built, and already wrapped by the plugin

`nkilib/experimental/transformer/attention_block_tkg.py` (2680 lines) is a fused decode
block: hidden → optional RMSNorm → fused `W_qkv` → QK norms → RoPE → attention against
a paged cache via `active_blocks_table` → in-kernel cache update → o_proj. Beneath it,
`core/attention/attention_tkg.py` (6362 lines) is the flash-decode core.

Everything decode needs that prefill lacks is in there, for GQA:

- **Paged block-KV addressing** — `attention_block_tkg_sharding_design_spec.md` §95-145
  ("Block KV Cache Support").
- **Context-parallel sharding with a distributed softmax combine** — spec §440-520. The
  algorithm is the standard flash-decoding rescale, `correction_r = exp(local_max_r −
  global_max)`, over `all_gather` of the stats, `all_reduce` of the corrected sums and
  `all_to_all` of the outputs, with `return_cp_softmax_stats` on the inner kernel.
- **Cache update, masking, bucketing, speculative `s_active > 1`.**

And the plugin already consumes this: `functional/attention/attention_decode.py:2-5`
imports `attention_block_tkg` and its sharding module, with a DCP variant at
`:260`. That file is 1802 lines of wrapper — the template for what an MLA decode
wrapper looks like here, and evidence that the hard integration work (FX-safe shims,
`wrap_nki` LNC indexing, fallbacks) is a solved pattern rather than a new one.

### 2.4 `d_head = 512` is tested, not just permitted

`test/integration/nkilib/core/attention/test_attention_tkg.py` has a section headed
`#### d_head = 512 ####` (lines 997-1013). With
`AttnTKGConfig(bs, q_head, s_active, curr_sprior, full_sprior, d_head, block_len, ...)`
(field order from `core/attention/attention_tkg_utils.py:47-65`):

| line | config | what it covers |
|---|---|---|
| 999-1000 | `(4, 1, 1, 1024/2048, …, 512, 0)` | flat KV, single active token |
| 1002 | `(4, 1, 1, 2048, 2048, 512, 16)` | **block KV**, block_len 16 |
| 1003 | `(1, 4, 1, 256, 256, 512, 128)` | block KV, q_head 4, block_len 128 |
| 1005 | `(1, 4, 1, 256, 256, 512, 16)` | block resize path, "matches Gemma4 global layer config" |
| 1007-1009 | same with `fp8_kv=True` | **FP8 KV**, flat and block |
| 1013 | `(64, 2, 1, 1024, 1024, 512, 16)`, fp8 | "production-like (TP=16, BS=64, s_prior=1024)" |

Every one of them passes `tp_k_prior=True`.

This settles the question I had flagged as the single most important unknown.

### 2.5 The K-transpose question: resolved from source, no spike needed

I had flagged: if a 512-wide `K_prior` cannot be transposed in-kernel, the latent needs
a second cache layout, which collides head-on with the capacity wall in
`GLM53-FLASH-FRAMEWORK-GAP.md` §5. It can be transposed. Two independent confirmations:

1. `attention_tkg.py:785-789`, the `use_dma_transpose` docstring: *"True when
   d_head<=128 and dtype is 2 bytes. **When False (d_head>128, FP8 without fp8_packed,
   or 1-byte dtype), block KV uses `nc_transpose` and flat KV non-FP8 uses
   `dma_transpose` directly per d-tile.**"* There is an explicit alternative path for
   exactly the `d_head > 128` case.
2. `tp_k_prior = not K_cache_transposed` (`attention_block_tkg.py:769-771`) — the
   layout is a caller flag, and `tp_k_prior=True` means the kernel transposes it
   itself. **Every `d_head=512` test sets `tp_k_prior=True`**, so the 512-wide
   in-kernel transpose is under test.

**One cache layout suffices. No second copy, no collision with the capacity wall.**

---

## 3. What decode must do differently

Separating real work from bookkeeping, as this is where cost estimates usually go
wrong.

### Real work — but adaptation, not invention

1. **The parallelization inversion.** Prefill shards queries and gathers the whole KV;
   decode has one query and must shard the *context*, then reconcile partial softmaxes.
   Solved for GQA in the CP path (§2.3). The work is making the MLA score/value core
   emit and consume the same softmax statistics, not inventing the combine.
2. **The paged gather.** Decode addresses the latent through a block table rather than a
   contiguous `S_kv`. Solved for GQA; tested at `d_head=512` with `block_len` 16 and
   128 (§2.4).
3. **The latent cache write, which carries a dependency.** See §4 — this is the one
   place where something we need is genuinely absent from our tree.

### Bookkeeping, not work

4. **Causal masking disappears.** A single query attends its whole prefix. KERNEL A's
   `q_pos_offset` machinery and the dense-mode causal mask are simply unused.
5. **`W_out = None` is mandatory at `d_head > 128`** (`attention_block_tkg.py:510-518`:
   in-kernel output projection requires `D <= 128`, "pass `W_out=None` and project
   externally"). MLA needs a per-head V-up *before* o_proj anyway, so the restriction
   coincides with MLA's structure instead of fighting it. KERNEL B is the external
   projection, already written.
6. **`s_active <= 7`** (`attention_tkg.py:290`) covers plain decode (1) and MTP with one
   draft (2) with room to spare.

### Performance, not correctness

7. **Stage 1 can start as plain torch.** At decode the two-stage projections and the
   absorption are per-token GEMVs. They are weight-bandwidth-bound, not compute-bound,
   and unfusing them costs bandwidth rather than correctness. PR #54 shipped Qwen3.5
   with attention computed inline in fp32 at a measured 6-8% of prefill, which is the
   precedent that unfused paths land. What *cannot* be torch is the attention over the
   cache, because that is what touches the whole cache — and the cache write, for the
   reason in §4.

---

## 4. The cache-write dependency, stated exactly

The chain, each link from source:

1. MLA decode runs at `d_head = 512`.
2. At `d_head > 128` the TKG block **forbids** in-kernel cache update:
   `attention_block_tkg.py:519-526` — *"the in-kernel block-KV cache update transposes K
   over the D dim (`_update_block_cache_vectorized`), which requires D <= 128. Callers
   must pass `update_cache=False` and scatter the returned new K/V into the cache
   externally."*
3. So the latent write must be external.
4. The only external scatter in our tree is torch `index_put_`
   (`model/gpt_oss/model_bf16.py:530-575`, confirmed at `:563-575`).
5. That is precisely the pathology STRATEGY.md §3.6 records: under `neuronx-cc`,
   `index_put_` lowers to a full-pool scatter, ~124 GB of pool traffic per decode step
   on MiMo-V2.5.

Upstream PR #40 fixed this with a NKI scatter kernel, and its own docstring
(`vllm_neuron/functional/attention/kv_cache_write.py` at `254b0ee`) says it exists for
exactly our case:

> "Models whose head dim fits the fused kernel's 128 cap should keep using
> `NF.attention_decode(update_cache=True)`; this exists for the eager-attention models
> (e.g. MiMo's 192-wide Q/K) that the fused path rejects."

### This is a port, not a merge

Worth being precise, because all three options in §5 depend on it and "merge PR #40"
understates it in one direction and overstates it in another.

- **Not a merge.** PR #40's branch is 300 files and +44,812/−15,763 against its
  merge-base with our fork point — it carries everything upstream did in between
  (disaggregated encoder, EC connector, a 2002-line runner diff). Merging it is not on
  the table as a side quest.
- **But the file is self-contained**: `kv_cache_write.py` is **228 lines**. Its only
  non-stdlib imports are `torch`, `nki`/`nki.isa`/`nki.language`,
  `nki.isa.constants.oob_mode`, `nkilib.core.utils.kernel_helpers`, and
  `vllm_neuron.nki.nki_hop`. That last one needs one line changed: in PR #40's tree
  `can_run_kernel` and `wrap_nki` are re-exported from `vllm_neuron/nki/`, which in our
  fork is empty (SPDX header only) — ours live in `utils/neuron_utils.py` and
  `libtorch_neuronx_lite.nki.nki_hop`.
- **It needs one real adaptation for MLA.** Its API is K/V-paired by construction and
  `_can_use_kernel` requires `k_cache.dim() == 4` **and**
  `k_cache.shape == v_cache.shape` — *"Both caches must be the canonical 4D paged buffer
  of the same shape, so one row index serves both."* MLA has **one** latent cache.
  Passing the same tensor as both arguments would issue two identical scatters and give
  the aliasing pass two outputs aliasing one buffer, which is the exact hazard the
  kernel was written to avoid. A single-tensor variant is the clean answer.
- It also rejects FP8 caches (`dtype in (bf16, fp16, fp32)`, because "FP8 caches use a
  packed layout that needs read-modify-write"). Our latent cache is bf16, so this does
  not bind now, but it forecloses an FP8 latent cache later.

**Sizing: ~150-200 lines, ported with provenance, as a latent-only variant.** The
precedent for carrying a kernel in-tree with a provenance note is
`functional/vendored_kernels/rotational_topk/`.

---

## 5. Sizing, and the recommendation

### (a) Recommended: spike the existing kernel before writing one

The prediction from §1-§2 is that MLA decode is a *configuration* of an existing,
tested kernel — MQA, `kv_heads=1`, `d_head=512`, `W_out=None`, `update_cache=False`,
`fuse_rope=False` — with the projections and V-up/o_proj external. That prediction is
cheap to falsify and expensive to assume. §6 is the plan.

**Cost: ~250-400 lines of harness plus simulator time.** Outcome either way is
decisive: a pass means the remaining work is a wrapper and a scatter; a failure names
the assert, which tells us precisely which of (b)'s pieces must be written.

### (b) If the spike fails: port KERNEL A's algebra into a TKG-shaped kernel

Est. **1,200-2,000 NKI lines**, by analogy: KERNEL A is 642 lines and leans on 1,566
lines of `mla_common_cte`; `attention_tkg` is 6,362. You would be writing the
MLA-specific score/value core against TKG's paged, CP-sharded, softmax-corrected
skeleton, reusing the CP combine and the block-KV addressing rather than reinventing
them. This is the number STRATEGY.md's "deepest hole" framing implies, and it is the
worst case, not the expected case.

### (c) From scratch, ignoring both halves

No argument for it. Recorded only so the option is visibly rejected.

### Shared prerequisites, all three

- The latent-cache scatter, §4: ~150-200 lines.
- The framework's latent KV cache spec and allocation
  (`GLM53-FLASH-FRAMEWORK-GAP.md` §2, ~100 lines) — needed before any of this runs on a
  device, though **not** before the simulator spike, which can allocate its own
  tensors.

### Sequencing

The spike needs neither hardware nor the framework work, so it can run in parallel with
everything in the framework document and should start first: it is the cheapest
question on the roadmap with the largest effect on the estimate.

---

## 6. The spike plan

Executable as written, by whoever has the x86 box. Nothing here needs a Trainium
device.

### 6.0 Environment

```bash
VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1        # both required
# and VLLM_NEURON_DISABLE_NKI_KERNELS must be UNSET
```

From `utils/neuron_utils.py:16-24`: `can_run_kernel` returns False in CPU mode unless
`NKI_SIMULATOR == "1"`, and returns False outright if
`VLLM_NEURON_DISABLE_NKI_KERNELS` is set. Level A below needs only `nki` + the nkilib
source on `PYTHONPATH`; it does **not** need `vllm`, so it is unaffected by the macOS
wheel problem and by Milestone 1's compile gate.

### 6.1 It will genuinely reach the kernel — assert this, don't test for it

Before running anything, note what cannot silently happen. The plugin's guard
`_can_use_attention_block_kernel` (`attention_decode.py:1211-1258`) has **no upper
bound on `d_head`**: it reads `d_head = V_cache.shape[-1]` and rejects only
`d_head % 2 != 0`. The `"<= 128"` at `:1362` is a docstring line, not enforcement, and
is stale relative to nkilib's 512. The other gates — `H % 128 == 0` (hidden 4096 ✓),
`attention_dp == 1`, and the `kv_heads > 1` 3-D-table rule (MQA, so `kv_heads == 1` ✓)
— all pass.

So there is no silent-fallback trap: a 512-wide latent reaches the kernel rather than
being quietly routed to the torch path. **Log the kernel-vs-fallback decision anyway**
— it is one line, and a spike that "passes" on the fallback would be worse than a
failure.

### 6.2 Level A — the q_head ladder (the primary question)

The unknown is no longer the transpose (§2.5). It is **`q_head`**. Every tested
`d_head=512` config uses `q_head ∈ {1, 2, 4}`; GLM-5.3-Flash needs **64**.

Start from a known-good tested config and walk one axis:

```python
# Baseline: test_attention_tkg.py:1002, known-good.
AttnTKGConfig(bs=4, q_head=1, s_active=1, curr_sprior=2048, full_sprior=2048,
              d_head=512, block_len=16, tp_k_prior=True, strided_mm1=False,
              use_pos_id=True, qk_in_sb=True, k_out_in_sb=True, out_in_sb=True)
```

Then hold everything fixed and raise `q_head`: **1 → 2 → 4 (all tested) → 8 → 16 → 32
→ 64**. Bisect to the first failure. Repeat at `s_active=2` for MTP.

**Prediction, and the reasoning so a failure is informative:** `q_head` enters only
through flattened dimensions that are *tiled* at `p_max` — `s_active_qh = s_active *
q_head` (`attention_tkg.py:1284`) and `s_active_bqh = bs * s_active_qh` (`:1139`),
with the softmax reductions computed "in tiles of size 128 over `bs * q_head *
s_active`" (`:271`, `:278`). Nothing caps `q_head` itself. The one hard cap is
`fuse_rope`-gated (`:1331`: `not fuse_rope or bs*q_head*s_active <= p_max`), and NoPE
means `fuse_rope=False`, so it cannot fire. **I expect `q_head=64` to work.** If it
does not, the failure is in SBUF pressure or the qk-swap layout, not in an explicit
limit — which is a different and more interesting answer than an assert.

**Corrected 2026-09-25 by the tier-0 run — this paragraph was wrong.** I had written
that the QK-swap fast path engages at `s_active_qh ≤ 128` and disengages above it, making
the threshold something to watch. It engages *never*, for two reasons neither of which is
`s_active_qh`:

- The **installed** nkilib (neuronx-cc 2.27.5334.0) disables it unconditionally —
  `attention_tkg_utils.py:356-358`: `# Disable due to sometimes causing OOB errors.`
  `# TODO: remove this gate.` `return False`. Everything below is dead code, for every
  model, not just MLA.
- That gate is **gone** in the newer source checkout (`92d11f6`), but
  `if d_head > p_max: return False` (*"d_head tiling not yet supported"*) still excludes
  `d_head = 512` there. Present in both versions.

So the swap path is not a knob for MLA, and `NKILIB_EXPERIMENTAL_ATTN_TKG_NO_SWAP=1` is
pointless here. Combined with `use_dma_transpose=False` (confirmed empirically), **both
KV fast paths are off at `d_head=512`** — functional on the slow route on two independent
axes. That is the shape of the performance risk, and it is unquantified.

### 6.3 The asserts to watch, and what each one means

A named assert is a good outcome. These are the ones reachable from an MLA-shaped call:

| where | assert | fires when | meaning if it fires |
|---|---|---|---|
| `attention_tkg.py:1222` | `0 < d_head <= 512` | never at 512 | would fire at DeepSeek's 576 — §1 |
| `attention_tkg.py:1305` | `d_head % 128 == 0 or <= 128` | never at 512 | also fires at 576 (ragged) |
| `attention_tkg.py:1311` | `n_d_tiles == 1 or not strided_mm1` | if `strided_mm1=True` | pass False, as every 512 test does |
| `attention_tkg.py:1316` | `n_d_tiles == 1 or not fp8_packed` | if `fp8_packed=True` | keep False; note fp8 KV *without* packing is tested at 512 |
| `attention_tkg.py:1331` | `not fuse_rope or bs*q_head*s_active <= 128` | only with RoPE fusion | NoPE ⇒ unreachable; would cap `bs` at 2 for a RoPE model |
| `attention_tkg.py:5526, 5763` | `num_blks_covering_s_active <= 128` | tiny `block_len` + large `s_active` | a block-size choice, not a wall |
| `attention_block_tkg.py:514` | `not do_out_proj or d_head <= 128` | if `W_out` passed | expected; project externally (KERNEL B) |
| `attention_block_tkg.py:523` | `not update_cache or d_head <= 128` | if `update_cache=True` | expected; §4's external scatter |

A failure that is **not** in this table — an SBUF allocator error, a PSUM bank
exhaustion, a simulator numerics mismatch — is the genuinely new information, and is
what would move the estimate toward §5(b).

### 6.4 Numerics, not just execution

Executing is necessary and not sufficient. Validate against the MLA torch references
that already exist:

1. Build a tiny MLA config (small `n_heads`, `L = 512` fixed by the kernel, short
   `S_kv`).
2. Run `mla_qkv_cte_torch_ref` to get `q_lift` and `c_kv` — the same inputs a decode
   kernel consumes.
3. Golden: `mla_sparse_attention_cte_torch_ref(..., dense=True)` restricted to the
   final query position. That is decode's answer computed by prefill's reference.
4. Compare against the TKG kernel fed `q_lift` as Q and the latent as both K and V.
5. Then `mla_vupmx_oproj_cte_torch_ref` for the V-up/o_proj tail.

The prefill references are the decode oracle — the algebra is identical, only the
access pattern differs. Set the sabotage controls the way dev1 did for the Qwen gates:
perturb `c_kv` and confirm the comparison fails, so a pass means something.

### 6.5 Level B — through the plugin

Only after Level A passes: call `NF.attention_decode` with an MLA-shaped cache and
confirm it routes to the kernel (§6.1's log line) rather than the fallback. This tests
the guard and the wrapper, not the kernel, and it needs `vllm` — so it is gated on a
Linux host, unlike Level A.

### 6.6 What each outcome means for the plan

- **Passes at `q_head=64`, numerics match** → MLA decode is a wrapper plus §4's
  scatter. Revise §5(b)'s 1,200-2,000 lines down to §5(a)'s 250-400 and tell Aman the
  deepest hole on the roadmap was a configuration question.
- **Passes at low `q_head`, fails at 64** → the core is right and head-group tiling is
  the work. Mid-hundreds of lines, and the failure mode names the tile to fix.
- **Fails at `q_head=1`** → the d-tiled path does not generalise off its tested
  configurations, and §5(b) is the real estimate. Still a cheap way to learn it.

---

## 7. Upstream: what is portable and what is not

vLLM's MLA structure is worth knowing before borrowing it. `cpu_mla.py`'s own
description of the shared scaffolding: `MLACommonImpl` orchestrates *"weight-absorbed
decode (MQA) and non-absorbed prefill (MHA)"*. That is the **opposite split from
nkilib**, whose prefill is already absorbed (§2.1). Anyone porting vLLM's orchestration
wholesale onto nkilib's kernels would be fighting that mismatch.

Non-CUDA references that are readable as specifications:

- `mla/cpu_mla.py` → `torch.ops._C.mla_decode_kvcache` (`csrc/cpu/mla_decode.cpp`),
  explicitly *"reference-quality ... not to be performant"*, constrained to
  `head_dim=576`, `v_head_dim=512`, `block_size=16`. Note `v_head_dim=512`: the value
  *is* the latent, confirming the absorbed-decode shape from a second codebase.
- `mla/amx_mla.py` (Intel AMX).

CUDA/ROCm-shaped and not portable in any useful sense: `flashmla.py`,
`flashmla_sparse.py`, `cutlass_mla.py`, `flashinfer_mla*.py`, `rocm_aiter_mla*.py`,
`triton_mla.py` (and there is no Triton on Trainium — STRATEGY.md §1).

**Conclusion: the algorithm is portable and multiply-referenced; only a Neuron
implementation is missing.** SGLang was not re-examined; STRATEGY.md §1 already
rejected that path on opportunity cost.

Adjacent, for the sparse variant: `mla_sparse_attention_cte` already implements
topk-gathered attention (`topk_indices`, flat or partition-tiled), so when dev2's
indexer lands there is a prefill-side consumer for it. No decode-side equivalent
exists, and at `index_topk = 2048` the decode gather is the same block-table machinery
with a different index source.

---

## 8. Open, and what would close it

`[unverified]` — everything in §6 is a prediction from source; none of it has run:

- **`q_head = 64` at `d_head = 512`.** The primary unknown. §6.2 closes it.
- **Whether the latent may be passed as both `K_cache` and `V_cache`.** The kernel
  takes them as separate arguments; MLA has one tensor. Aliasing behaviour under the FX
  aliasing pass is the specific risk, and it is the same hazard §4 describes for the
  scatter. A tiny simulator test closes it.
- **MX/FP8 interaction.** FP8 KV at `d_head=512` is tested but `fp8_packed` is asserted
  off at `d_head > 128`. Our latent is bf16, so this is future work, not a blocker.
- **Performance, entirely.** Nothing here is a perf claim. Note in particular that
  `use_dma_transpose` is False at `d_head > 128`, so the 512-wide path takes
  `nc_transpose` rather than the fast batched DMA transpose — a known slower route, not
  a wall, and unquantified.
- **Whether `mla_common_cte`'s `L == 512 == P_MAX * 4` assumption is load-bearing
  anywhere a decode path would reach.** Only relevant if §5(b) happens.

---

## 9. Corrections to our own documents

1. **The 128 head-dim cap is prefill-only.** STRATEGY.md §3.2 and AGENTS.md's standing
   constraint say `head_dim` above 128 "falls off every fused attention kernel".
   `MAX_HEAD_DIM = 128` lives in `functional/attention/attention_cte.py:17` and
   `_MAX_HEAD_DIM` in `attention_segmented_cte.py` — both **CTE**, i.e. prefill.
   `attention_decode.py` has no head-dim gate at all, and nkilib's decode FA supports
   `d_head` up to 512 *with tests*. (AGENTS.md has been corrected; STRATEGY.md §3.2
   lives on another branch.)
2. **STRATEGY.md §3.4's "deepest hole" needs qualifying** to: no Neuron MLA decode
   kernel exists, but the absorbed algebra exists in the prefill kernels, the decode
   machinery exists in the TKG block, and `d_head = 512` is a tested configuration. The
   honest risk statement is a spike, not a kernel project.
3. **The DeepSeek-family decode gap is wider than GLM-5.3-Flash's**, per §1 — which
   inverts the assumption behind ordering GLM-5.3 before Flash.
