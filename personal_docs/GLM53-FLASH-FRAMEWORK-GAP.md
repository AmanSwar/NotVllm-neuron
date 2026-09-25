# What vllm-neuron must grow to host GLM-5.3-Flash

Framework gap analysis. Written 2026-09-25 against fork point `f8abae6`
(`release-0.24.0.1.1.0`, vLLM 0.24.0, Neuron SDK 2.32), with upstream PR #54 as the
calibration point.

**Scope.** Framework surfaces only — everything *outside* `vllm_neuron/model/<arch>/`.
The model directory itself (KDA, MLA, mHC, MoE, the DSA indexer) is a separate cost,
budgeted elsewhere. No model code is proposed here.

**Status of the evidence.** Every claim below is traced to a file and line in this
repo, in the pinned vLLM 0.24.0 source, or in the live
`zai-org/GLM-5.3-Flash/config.json` fetched 2026-09-25. Nothing has been executed:
no plugin import, no compile, no device. Arithmetic is marked as arithmetic.
Assumptions are stated where they could not be removed.

**The pinned vLLM is not the vLLM that implements this model.** `requirements/core.txt`
pins `vllm==0.24.0`. A newer vLLM (checked here at `36fa72d2d0`, 2026-09-19) ships a
complete GLM-5.3-Flash implementation including the cache abstractions this document
says we lack. Reading that newer tree and assuming its API is available would
produce a plan that cannot be built. Throughout, **"the pin"** means vLLM 0.24.0 and
is the contract; the newer tree is cited only as a reference implementation. §8 sizes
what a version bump would buy.

---

## 0. Summary

| # | Surface | Change | Additive? |
|---|---|---|---:|
| 1.1 | **`use_mla` is False → `get_head_size()` returns 0** | ~40 lines (a vLLM patch) | plugin-additive, vLLM-invasive |
| 1.2 | Architecture registration timing (`platform.pre_register_and_update`) | ~15 lines | yes |
| 1.3 | vLLM's hybrid contract on our registered class | ~45 lines | yes |
| 2 | Latent KV cache spec + allocation | ~100 lines | yes |
| 3 | Four cache kinds in one model | ~100 lines | mostly |
| 4 | Page-size reconciliation | ~40 lines *if* recommendation (c) | yes, under (c) |
| 5 | 1M context | ~0 lines of framework; a capacity decision | n/a |
| 6 | MTP | ~400+ lines — **descope** | no |

The recommended path totals **roughly 300–400 lines outside the model directory**,
i.e. about one PR #54 (which was 336). That is the headline: the framework work is
PR #54-sized and mostly additive. The cost of this model lives in the model
directory, in the MLA decode kernel, and in §5's capacity ceiling — not in the
framework.

Two findings change the plan rather than adding to it:

- **§1.1** breaks startup before any cache code is reached, and does so by computing
  from a zero. It is the first thing to fix and the cheapest.
- **§5** says 1M context at useful throughput is a decode-context-parallelism
  problem, not a bucketing problem. Worth knowing before the milestone is planned
  around as written.

---

## 1. Front-end prerequisites

These come first because they fail before any KV cache code runs. None of them
applied to PR #54, which is why they are invisible from that diff.

### 1.1 `get_head_size()` returns 0 — lead with this one

`ModelConfig.use_mla` is `is_deepseek_mla and not VLLM_MLA_DISABLE`
(`config/model.py:1627`). `is_deepseek_mla()` is a **hardcoded `model_type`
allowlist** at `transformers_utils/model_arch_config_convertor.py:252-270`. It
contains `deepseek_v2`, `deepseek_v3`, `deepseek_v32`, `deepseek_v4`, `glm_moe_dsa`,
`glm4_moe_lite`, `kimi_k2`, `kimi_linear`, `longcat_flash` and others. It does
**not** contain `glm5_next_text`.

Note the string. The live config nests the text config, and the two `model_type`
values differ:

```
config.json:  model_type = "glm5_next"          architectures = ["Glm5NextForConditionalGeneration"]
  text_config: model_type = "glm5_next_text"
```

`is_deepseek_mla()` reads `self.hf_text_config.model_type`, so the value that must
be recognised is **`glm5_next_text`**, not `glm5_next`.

With `use_mla` False, `get_head_size()` (convertor.py:47-71) skips its MLA branch and
falls through to:

```python
if getattr(self.hf_text_config, "head_dim", None) is not None:
    return self.hf_text_config.head_dim
```

**The live config ships `text_config.head_dim: 0`.** `0 is not None`, so
`get_head_size()` returns **0** rather than `kv_lora_rank + qk_rope_head_dim = 512`.
`get_num_kv_heads()` (`config/model.py:1259-1270`) likewise returns
`num_key_value_heads // TP` = `64 // TP` instead of the MLA-correct 1.

The concrete failure, assuming §1.3 is done so that the hybrid path runs at all:
vLLM's `_align_hybrid_block_size` (`platforms/interface.py:633-790`) computes
`attn_page_size_1_token` from `FullAttentionSpec(head_size=0)` → 0, then

```python
attn_block_size = kernel_block_alignment_size * cdiv(
    mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token)
```

divides by zero. Predicted symptom: **`ZeroDivisionError` inside
`_align_hybrid_block_size`, reached from our `_align_hybrid_page_sizes`**
(`vllm/platform.py:180-244` on `pr54`). That is a clean crash. The dangerous variant
is the near miss: force `head_dim` to a plausible non-zero value via `hf_overrides`
and every page size downstream is quietly wrong instead, which is how you get a
model that loads, runs, and corrupts one cache group.

**Three ways to fix it, and the choice matters:**

1. A plugin-side patch to `is_deepseek_mla` (or to the arch-config convertor),
   living in `vllm_neuron/vllm/patches/` alongside `pin_memory_patch.py` and
   `port_hold_patch.py`. ~40 lines with a provenance note. This is the established
   precedent for "vLLM is wrong about our model".
2. `hf_overrides` to add a recognised `model_type`. Cheap, but it lies to the rest of
   vLLM about the architecture, and the front-end reads `model_type` for more than
   this.
3. A vLLM bump (§8).

Recommend (1). It is explicit, it is where a reader will look, and it does not
misrepresent the checkpoint.

**Additive in the plugin, invasive in vLLM.** Everything else in §1 and §2-§4
depends on this being right, because every page-size calculation starts from
`get_head_size()`.

### 1.2 The architecture is unknown to the pinned vLLM

`vllm/model_executor/models/registry.py` has `GlmMoeDsaForCausalLM` (line 122 →
`deepseek_v2`; that is GLM-5.3, *not* Flash) and no `glm5_next` entry of any kind.
It does have `Qwen3_5ForConditionalGeneration` (line 566), which is exactly why
PR #54 never met this problem.

The plugin registers its models in
`vllm_neuron/vllm/worker/neuron_worker.py:366-383`, deliberately late — the comment
there says vLLM's registry overwrites ours if registration happens earlier in
`platform.py` (TODO CHRYS-72). But `ModelConfig` validation happens in the
**front-end**, before any worker exists, and an unknown architecture fails there.

The hook that runs early enough already exists:
`NeuronPlatform.pre_register_and_update` (`vllm/platform.py:161`), called from vLLM
`engine/arg_utils.py:1794` and `:2685`, documented in the plugin as *"Register Neuron
model architectures before ModelConfig validation"*. Today it registers only the
synthetic test model behind `VLLM_NEURON_SYNTHETIC_MODEL=1`.

**~15 lines, additive.** Register `Glm5NextForConditionalGeneration` there. The
CHRYS-72 ordering hazard needs a look — the worker-side registration exists because
early registration was overwritten — but for an architecture vLLM has never heard of
there is nothing to be overwritten *by*, which is the asymmetry that makes this
cheap for this model and awkward for Qwen3.5.

### 1.3 vLLM's hybrid contract must be satisfied by *our* class

`ModelConfig.is_hybrid` → `self._model_info.is_hybrid` → `is_hybrid(model_cls)`
(`registry.py:789`), which reads the `IsHybrid` protocol's ClassVar
(`interfaces.py:791-816`):

```python
class IsHybrid(Protocol):
    is_hybrid: ClassVar[Literal[True]] = True
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config) -> ...
```

PR #54's factory supplies `get_mamba_state_shape_from_config` and
`get_mamba_state_dtype_from_config` (`model/qwen3_5/factory.py:98,120`) with a
comment stating the contract precisely — *vLLM asks the registered class, not its
own implementation* (factory.py:89-95) — but it never declares `is_hybrid`, because
vLLM's own `Qwen3_5ForConditionalGeneration` already does.

For GLM-5.3-Flash there is no fallback class, and the failure is silent rather than
loud: `is_hybrid` defaults False → our `_align_hybrid_page_sizes` early-returns on
it (`platform.py:210` on `pr54`) → `cache_config.mamba_block_size` stays `None` →
the runner raises the `ValueError` PR #54 added at
`neuron_model_runner.py:8836-8846`. That error message points at `is_hybrid`, which
is good luck rather than good design, and worth preserving.

Geometry is a delegation, not a derivation: **`MambaStateShapeCalculator.kda_state_shape`
already exists at the pin** (`model_executor/layers/mamba/mamba_utils.py:237`),
along with `kda_state_dtype` (line 119). Use them, for the reason PR #54's factory
gives: vLLM sizes the state *pages* from the same helper, so a second copy of the
arithmetic aliases memory instead of raising.

**~45 lines, additive.** `is_hybrid = True`, the two classmethods delegating to
`kda_state_shape` / `kda_state_dtype` with `linear_attn_config`'s `num_heads: 64`,
`head_dim: 128`, `short_conv_kernel_size: 4`.

### 1.4 transformers ≥ 5.16.1

Needed for the `glm5_next` config class. The plugin pins
`transformers>=5.5.1,<6.0.0`, so this is a version choice, not a conflict. Not a code
change.

### 1.5 Text-only serving of a vision checkpoint — already solved

The checkpoint carries a `vision_config` (and 347 vision tensors, per dev1's
converter work), but Milestone 2's scope is text-only. PR #54 built exactly this
surface: `NeuronPlatform._multimodal_inputs_enabled` (`vllm/platform.py`, added in
PR #54) skips vision bucket resolution when every `limit_mm_per_prompt` is 0, for the
stated reason that a hybrid model's state pages are sized off `max_model_len` and
vision buckets a request can never use are pure waste.

**No change.** Serve with `limit_mm_per_prompt={"image": 0, "video": 0}`.

---

## 2. The latent KV cache

### What exists

`vllm_neuron/model/kv_cache.py` is 33 lines: `LayerSpec(name, num_kv_heads,
head_size, dtype, sliding_window_size, chunk_size)` and `KVSpec(layers)`. PR #54
adds `RecurrentLayerSpec(name, shapes, dtypes)` and a `recurrent_layers` list,
taking the file to 61.

`NeuronModelRunner.get_kv_cache_spec` (`neuron_model_runner.py:8635-8690`) turns each
`LayerSpec` into a `FullAttentionSpec` or, when `sliding_window_size` is set, a
`SlidingWindowSpec`. Those are the only two it can emit.

`initialize_kv_cache` (8456-8605) allocates one flat `int8` buffer per
`KVCacheTensor`, reinterprets it to the cache dtype, and views it as
`(2, num_blocks, num_kv_heads, block_size, head_size)` — the leading 2 being K and V.

### What breaks

**`MLAAttentionSpec` subclasses `FullAttentionSpec`** (pinned
`v1/kv_cache_interface.py:367`). So if the plugin emitted one, it would *pass* the
isinstance check at `neuron_model_runner.py:8528` and then fail on the view, because
the MLA page has no V half. The pinned `MLAAttentionSpec.real_page_size_bytes` is
correctly

```
storage_block_size * num_kv_heads * head_size * dtype_size
```

with no factor of 2. Concretely, at `block_size = B`, one latent page holds
`B × 1 × 512` bf16 elements. The typed view therefore has `num_blocks × B × 512`
elements, while `(2, num_blocks, 1, B, 512)` demands `2 × num_blocks × B × 512` —
**exactly twice as many**, so `.view()` raises
`RuntimeError: shape ... is invalid for input of size ...`.

That is a loud failure, which is worth stating plainly: nobody should go hunting for
a numerics bug here.

### What the change looks like

The seam is friendlier than it first appears, for two reasons.

- **The model owns cache writes; the runner only allocates and binds.**
  `bind_kv_cache(kv_caches: dict[str, list[torch.Tensor]])` takes a *list* per layer.
  The FP8-packed-K precedent — `[num_blocks, num_kv_heads, block_size // 2,
  head_size, 2]`, detected in `bind_kv_cache` from `k_cache.dim() == v_cache.dim() + 1`
  — establishes that reshaping a layer's view is inside the seam. A one-tensor latent
  cache is expressible; PR #54's `page_major` path already returns a single-element
  list (`kv_caches[layer_name] = [kv_pages]`).
- **The plugin fabricates its own spec dict.** On GPU, vLLM collects specs from
  `Attention` module instances, so cache cardinality is tied to real layers. Here,
  `get_kv_spec()` returns whatever the model says, keyed by names the model chooses.
  That is what makes §3 and §4 tractable.

Work: a `LatentLayerSpec` (or a `latent: bool` on `LayerSpec` — see below) ≈ 30
lines; a spec-emission branch in `get_kv_cache_spec` ≈ 25; an allocation branch in
`initialize_kv_cache` that views the buffer as
`(num_blocks, num_kv_heads, block_size, head_size)` without the leading 2 ≈ 45.

**~100 lines, additive.** Prefer a distinct spec class over a flag on `LayerSpec`,
following PR #54's reasoning for `RecurrentLayerSpec`: the fields that matter differ,
and a flag makes every existing consumer of `LayerSpec` a place where someone can
forget to check it.

One design note for whoever writes it: dev1's MLA decode kernel research will fix
whether the latent cache is one tensor or two (a separate `k_pe` cache). With
`qk_rope_head_dim: 0` there is no RoPE anywhere in this model's attention stack, so
one tensor of width `kv_lora_rank = 512` is the expectation — but it is the kernel's
call, not the framework's, and the framework should not hardcode the assumption.

---

## 3. Three cache kinds — actually four

GLM-5.3-Flash needs, simultaneously: recurrent state for 34 KDA layers, a latent KV
cache for 11 MLA layers, the indexer's kpool-compressed scoring cache, and the
indexer's raw tail cache. PR #54 only just taught the runner about two kinds.

### What each of the four actually holds

Settled by reading the reference implementation
(`Notvllm/vllm/models/glm5next/common/attention.py:78-207`), which also
independently corroborates dev1's deliverable-4 conclusion:

| cache | contents | geometry |
|---|---|---|
| KDA recurrent state | conv window + recurrent state, per sequence, fixed size | `kda_state_shape` |
| MLA latent | full-fidelity latent, **one per token** — the only thing attention gathers | heads 1, width 512, bf16 |
| indexer kpool | fp8 compressed entry **plus a 4-byte scale**, one per `index_kpool` tokens, scoring only | heads 1, width **132**, uint8 |
| indexer tail | raw bf16 K in head slot 0, bf16 gate score in slot 1 — explicitly *not* the compressed entry | `block_size = index_kpool = 4`, heads 2, width 128 |

The tail cache is a circular buffer of `index_kpool` slots per request, overwritten
by `pos % kpool`. Its block size *is* its semantics.

### How the pinned planner groups them

`_get_kv_cache_groups_uniform_page_size` (`v1/core/kv_cache_utils.py:1108-1229`)
buckets layers by identical spec, then splits so that groups are equal-sized:
`group_size = min(len(layers))`, raised to `max` only when `max < min * 1.5`.

For 34 KDA + 11 MLA + 11 indexer + 11 tail: `min = 11`, `max = 34`, and
`34 ≥ 11 × 1.5 = 16.5`, so `group_size = 11`. KDA splits into `cdiv(34, 11) = 4`
groups (assigned `layers[i::4]`). **Seven groups.**

`get_kv_cache_config_from_groups` (1318-1400) then allocates `group_size` buffers,
where buffer *i* is **shared by the i-th layer of every group**:

> We will have group_size memory pools, each is shared by one layer from each group.
> As layers of different groups have different block table, they will use different
> parts of the shared Tensor.

So: 11 buffers, each shared by up to 7 layers with up to 4 distinct layouts.

### Why that matters, and what PR #54 already solved

This is exactly the constraint PR #54's `page_major` opt-in addresses
(`neuron_model_runner.py:8513-8530` and `8582-8613`; model opt-in
`kv_cache_page_major = True` at `model/qwen3_5/model.py:724`, with a private
zero-page / write-sink pair at model.py:312 and 437-443). Its comment states the rule:
vLLM requires block `b` to live inside page `b` for every layer sharing the buffer, so
a flat "all of state A, then all of state B" layout silently runs through another
group's pages.

PR #54 also documented the hazard that scales worst here:

> block ids, though allocated per group, are global: a page that held another group's
> float32 state comes back as this group's, so the model must treat any byte it did
> not write this step as hostile rather than merely stale.

With two layouts that is a manageable invariant. With four it is four ways to read
another group's bytes as your own, and it only manifests above `max_num_seqs == 1`
where padded rows appear. **Test at batch > 1 from the first bring-up run**, not
after.

### What helps

`attn_metadata` is built **per KV cache group** and fanned out by layer name
(`neuron_model_runner.py:4150-4256`: one `attn_metadata_i` dict per group, then
`for layer_name in kv_cache_group_spec.layer_names: attn_metadata[layer_name] = attn_metadata_i`).
Each group therefore gets its own block table and slot mapping for free. Combined
with the plugin fabricating its own spec dict (§2), **declaring pseudo-layers for the
indexer and tail caches costs nothing structurally** — they are names in a dict, not
modules.

Work: generalising `page_major` past two layouts ≈ 60 lines; pseudo-layer naming and
its bind/dispatch plumbing ≈ 40.

**~100 lines, mostly additive.** The one invasive edge is `page_major` itself: it
changes an existing code path that Qwen3.5 depends on. Any change there should be
gated on the same model opt-in flag, and Qwen3.8-27B should be re-validated
afterwards — which is cheap only while dev1's branch is still live.

---

## 4. Page-size reconciliation — the hard one

### The constraint

Every KV cache group must have the **same page size**. `get_uniform_page_size`
(`kv_cache_utils.py:992-998`) asserts it. `unify_kv_cache_spec_page_size`
(1049-1100) is the only escape, and it offers exactly two moves: grow a smaller
layer's `block_size` by an **integer** ratio, or pad its page — and padding is
permitted *only* for attention specs with `indexes_kv_by_block_stride=True`.
Otherwise `NotImplementedError`.

### The four page sizes

Per rank, at cache block size `B` tokens. Assumptions: TP = 64 (a trn2.48xlarge at
`logical_nc_config=2`: 16 chips × 4 logical cores), bf16 activations, KV cache dtype
bf16, `mamba_cache_mode = "none"`. **This is arithmetic from the live config and the
pinned specs, not measurement.**

```
MLA latent     = B × 1 × 512 × 2 B                              = 1024·B
indexer kpool  = (B/4) × 1 × 132 × 1 B                          =   33·B
indexer tail   = 4 × 2 × (128 + 0) × 2 B                        =    2048     (fixed)
KDA state      = conv (24576/64) × 3 × 2 B  +  1 × 128 × 128 × 4 B
               = 384 × 3 × 2 = 2304        +  65536              =  67840     (fixed)
```

The indexer's 132 is `index_head_dim + index_head_dim // quant_block_size * 4` =
`128 + 1 × 4`, with `quant_block_size = 128` hardcoded and asserted in the reference
(`attention.py:274,366`). The tail's width uses `head_size_v = 0`, which the pinned
`SlidingWindowSpec.real_page_size_bytes` honours (`kv_cache_interface.py:487-504`:
`block_size * num_kv_heads * (head_size + head_size_v) * dtype_size`). The KDA
figures come from `kda_state_shape(tp=64, num_heads=64, head_dim=128,
conv_kernel_size=4)` and `kda_state_dtype` → conv bf16, recurrent fp32.

Sensitivity, since TP is not fixed: the KDA state page is 135,680 B at TP=32 and
**271,360 B at TP=16** — which is, by coincidence, the exact figure PR #54 quotes for
Qwen3.5's DeltaNet page and calls out as factoring into `1024 × 5 × 53`, i.e. not
divisible by any sane block size. The same is true here, and it is why
`mamba_page_size_padded` exists.

### The indexer page can never be unified, for any block size

Take the smallest `B` that is a multiple of the plugin's 32-token kernel alignment
and makes the latent page cover the KDA state: `1024·B ≥ 67840` → `B ≥ 66.25` →
**B = 96**, latent page 98,304.

```
98304 / 3168 = 31.0303…        (3168 × 31 = 98208, remainder 96)
```

Not integral, and not paddable without `indexes_kv_by_block_stride`. So it raises.

But the ratio is worse than a bad `B`. It is

```
1024·B / (0.25 · H · B)  =  4096 / H       where H = indexer width in bytes
```

**`B` cancels.** No block size can unify the indexer page with the latent page; the
ratio depends only on `H`, and unification requires `H | 4096`. `H = 132` does not
divide 4096. `H = 128` would (ratio 32). So:

> **The 4 bytes of inline fp8 block scale per pool entry are the entire reason the
> indexer cache cannot be unified at the pin.** 128 divides 4096; 132 does not.

That is a structural fact, not a tuning problem, and it is the single most useful
thing in this section.

The tail page *does* divide (98304 / 2048 = 48), but "unifying" it means growing its
`block_size` from 4 to 192, which destroys the `pos % index_kpool` circular-buffer
semantics that make it a tail cache at all. A spec that divides is not the same as a
spec that survives.

### Why PR #54's mechanism does not stretch to cover this

`_align_hybrid_page_sizes` (`vllm/platform.py:180-244` on `pr54`) delegates to
vLLM's `_align_hybrid_block_size` (`platforms/interface.py:633-790`), which
reconciles exactly **two** page kinds: one attention page (MLA or full, selected on
`use_mla`) and one mamba page. It grows the attention block size until the attention
page covers the state page, then pads the state page to match. It has no notion of a
third or fourth kind, and it reads `use_mla`, which §1.1 shows is False here.

The newer vLLM solves this structurally rather than arithmetically, with machinery
that does not exist at the pin: `SparseCacheRole` / `cache_role=INDEXER` and
`is_index_group_leader` to force the indexer into the *same group* as its MLA layer
(sharing one block table, so no second page size exists), a settable
`storage_block_size`, `block_stride_alignment`, and a purpose-built
`KpoolTailSpec` whose `max_num_blocks_per_req()` returns 1 with
`uses_slot_mapping = False` and `prefix_cacheable = False`
(`Notvllm/vllm/v1/kv_cache_interface.py:150,641,988`).

Worth noting what the pin *does* have: `MLAAttentionSpec.compress_ratio`, where
`storage_block_size = block_size // compress_ratio`, consumed by the GPU runner and
attention utils (`gpu_model_runner.py:7132-7134`, `gpu/attn_utils.py:273-300`). That
is mechanically the same idea as `tokens_per_state`, and it computes the kpool page
correctly. What it cannot do is put the indexer in the MLA layer's *group* — grouping
buckets by spec equality (kv_cache_utils.py:1177-1180), and a `compress_ratio=4` spec
is not equal to the latent spec. So the indexer gets its own group, its own page size,
and the `4096 / H` wall. **[unverified]** — `compress_ratio` looks usable for the
page arithmetic but has not been exercised.

### Three options

**(a) Bump vLLM** to a version carrying the GLM-5.3 cache API. Deletes most of §2,
§3 and §4. Sized in §8. Fork-wide decision.

**(b) Keep four groups and pad.** Emit plugin-side specs with
`page_size_padded == the common page`. The indexer's real content is 3,168 bytes in a
98,304-byte page — **31× waste**, and the indexer cache is the one whose footprint
scales with context length. At 1M tokens the indexer's real content is ~0.4 GiB/rank;
padded, it is ~12 GiB/rank, which on its own exceeds the budget in §5. Rejected on
arithmetic, not taste.

**(c) Recommended: fold the indexer and tail caches into the MLA layer's own
allocation.** One spec per MLA layer, sized to hold latent + kpool + tail for that
layer's blocks; the model slices its own page. vLLM's planner then still sees exactly
two cache kinds — attention and mamba — which is precisely the configuration PR #54
left working, and `_align_hybrid_page_sizes` keeps applying unchanged.

Why (c):

- It is the only option that does not require either a vLLM bump or paying 31× on the
  fastest-growing cache.
- The seam permits it. The FP8-packed-K precedent establishes that a layer's *view*
  of its page is the model's business as long as the byte footprint and page
  accounting are unchanged; `bind_kv_cache` takes a list, so the model can be handed
  one page and slice three views out of it.
- It keeps the change additive, which per PR #54's experience is the difference
  between landing and rebasing forever.

What it costs, priced honestly:

- **The model owns page arithmetic that vLLM would otherwise own.** Three offsets
  inside one page, recomputed model-side from its own shapes — the same debt PR #54
  took on for `page_major`, and its comment is explicit that a second copy of the
  arithmetic "is a silent memory-aliasing bug waiting to happen". This is real debt
  and it compounds with §3's hostile-bytes invariant.
- **Prefix caching over the indexer cache becomes the model's problem**, because vLLM
  no longer knows the sub-caches exist. Fine for bring-up, a constraint later.
- **It diverges from upstream's structure**, so if we later bump vLLM we throw this
  away rather than rebase it. That is an argument for (a) *if* a bump is coming
  anyway — see §8.

Work under (c): ~40 lines of framework (one spec variant, one allocation branch),
plus model-side slicing that belongs in the model budget.

---

## 5. 1M context — the capacity wall, then the bucketing

### The wall

The MLA latent cache holds **one full-fidelity latent per token**, and under MLA it
is a single KV head (`get_num_kv_heads()` returns 1), so **every rank holds the whole
thing**. Nothing shards a single head.

```
11 MLA layers × 512 × 2 B = 11,264 B/token ≈ 11 KiB/token
× 1,048,576 tokens        ≈ 11.5 GiB per 1M-token sequence, per rank
+ indexer kpool           ≈  0.4 GiB  (33 B/token × 11 layers)
```

Against the budget at TP=64 on a trn2.48xlarge: 96 GiB per chip ÷ 4 logical cores =
**~24 GiB per rank**, minus ~5.1 GiB of FP8 weights (328 GB / 64) — or ~10 GiB if
loading the BF16 repo, per §7.

**One 1M-token sequence fits, at batch size 1, with nothing left over.** In BF16 it
does not comfortably fit at all.

Two consequences worth stating before the milestone is planned around as written:

- **The DSA indexer does not help here.** `index_topk` changes what attention
  *reads*; the latent cache still stores every token. Sparsity buys compute and
  bandwidth, not capacity.
- **1M context at useful throughput is a decode-context-parallelism problem.** The
  lever exists at the pin: `FullAttentionSpec.max_memory_usage_bytes`
  (`kv_cache_interface.py:236-244`) divides `max_model_len` by
  `dcp_world_size * pcp_world_size`, and `MLAAttentionSpec` inherits it. The plugin
  has partial DCP support already (`utils/bucket_utils.py:106-140`,
  `align_prefill_buckets_for_dcp`). But DCP interacts directly with §4's block-size
  arithmetic (`dcp_stride = decode_context_parallel_size × block_size`, and every
  prefill bucket must be divisible by it), so the two have to be designed together.

**[unverified]** The per-rank HBM figure assumes 4 logical cores share a chip's
96 GiB with no other consumer. Not measured on hardware.

### The bucketing, which is the easy half

For `max_model_len = 1,048,576`:

- `resolve_segmented_prefill_config` (`utils/bucket_utils.py:332-388`) forbids
  single-shot prefill above `MAX_MODEL_LEN_SINGLE_SHOT = 16 Ki`, so segmented prefill
  is mandatory, with a segment size from
  `SUPPORTED_KV_SEGMENT_SIZES = {512, 1024, 2048, 4096, 8192}`. A 1M prompt at 8192
  is 128 segments.
- `validate_decode_context_length_buckets` (487-550) requires each bucket to be a
  multiple of 128 (the NKI attention tile, `P_MAX`) and strictly below
  `max_model_len`, which is the implicit fallback bucket.

Both constraints are satisfiable by configuration. **No framework change identified
here** — which is a real result, and the opposite of what "bucketing for 1M is a
first-class design problem" implies.

The cost is not code, it is NEFFs: a power-of-two decode-context ladder from 128 to
1M is ~14 buckets, multiplied by the `num_seqs` buckets, each a separate compiled
graph, and after warmup `fail_on_recompile` is armed so an uncovered shape is a hard
failure. Warmup time and NEFF cache size are the things to measure early. Given the
capacity wall above, a batch-1-at-1M configuration keeps that product small, which is
a silver lining of a sort.

---

## 6. MTP — descope

`num_nextn_predict_layers: 1`, so there is one MTP layer. It is blocked in two
independent places, and the second is the one that matters.

**Hard stop 1.** `neuron_model_runner.py:746-757` raises
`ValueError(f"Unsupported speculative decoding method: {method}")` for anything but
`eagle3`; `vllm/spec_decode/eagle.py:45` asserts the same. There is no MTP proposer.

**Hard stop 2, and it is the real one.** `_update_states_after_model_execute`
(`neuron_model_runner.py:8182-8193`) is still this:

```python
"""On GPU this handles MTP/EAGLE for hybrid models (linear attention
state shifting). Neuron does not support hybrid models yet, so this
is a no-op."""
pass
```

**PR #54 did not touch it.** That function is precisely the recurrent-state
rollback that MTP needs across 34 KDA layers: when a draft token is rejected, the
KDA state must be rewound. So MTP is not merely unimplemented — it is blocked by a
function whose docstring asserts that hybrid models are unsupported, on a branch
where they now are.

The pin does carry the hooks: `MambaSpec.num_speculative_blocks` and
`mamba_cache_mode` (`kv_cache_interface.py:629-668`), and both
`kda_state_shape`/`gated_delta_net_state_shape` take a `num_spec` argument that
widens the conv state. The config also has `index_share_for_mtp_iteration: true`, so
the indexer has MTP-specific behaviour of its own to reproduce.

**Recommendation: descope MTP for first bring-up.** Cost of doing it: a proposer
class parallel to `EagleProposer` (~400+ lines) plus recurrent-state rollback across
the KDA layers, which is new work in an area PR #54 explicitly left alone.
**Invasive.** Cost of descoping: the ~1.5-2× decode throughput MTP would buy, and
1,760 checkpoint tensors left unloaded (dev1's converter already drops them
deliberately).

---

## 7. What does *not* need to change

A shorter accurate list is worth more than a long speculative one. Each of these was
checked and found sufficient.

- **Model-owned KV writes and prefill/decode dispatch.** The model dispatches on
  `max_query_len <= decode_token_threshold` and owns its cache writes; the runner
  allocates and binds. Nothing about four cache kinds changes that contract.
- **`attn_metadata` plumbing.** Already per-group and keyed by layer name
  (`neuron_model_runner.py:4150-4256`). Pseudo-layers get block tables for free.
- **Text-only serving of the vision checkpoint.** PR #54's
  `_multimodal_inputs_enabled` (§1.5).
- **`model/registry.py`.** A one-line in-tree addition. Out-of-tree registration is
  second-class (`vision_utils.py` and `spec_decode/eagle.py` resolve via
  `dict(get_models())`), which argues for in-tree, but that is a convention, not a
  gap.
- **`hlo2tensorizer_options`.** PR #54 already added the escape hatch
  (`model/neuron_config.py:167-177`) for pure-torch graphs that
  `--modular-flow-mac-threshold=10` breaks. GLM-5.3-Flash will likely need it too;
  no new surface.
- **mHC.** Pure model code with a plain-torch reference in vLLM, matched to fp32
  rounding by dev1's oracle. No framework surface. Two constraints carried from that
  work: exactly 20 Sinkhorn iterations, and validate the mixing term via captured
  tensors (`vllm_neuron/accuracy/tensor_capture.py`), never end-to-end logits.

### A correction to our own scope: FP8 is not on the critical path

AGENTS.md currently says FP8 is on the critical path for Milestone 2. The framework
cannot do it, so it cannot be first.

The plugin has **no blockwise `[128,128]` weight path**. Its FP8 is per-projection
static scales (`model/llama3/model_static_fp8.py:124,201-232`); MXFP4 and MXFP8 are
block-scaled but a different format with different metadata. GLM-5.3-Flash ships
`quantization_config: {quant_method: fp8, fmt: e4m3, weight_block_size: [128,128],
activation_scheme: dynamic}`, which nothing in the plugin consumes.

So first bring-up must dequantize offline to BF16 — which dev1's weight converter
already does, and whose plan is verified against all 76,108 real tensor names. The
cost is HBM: 643 GB / 64 ranks ≈ 10 GiB/rank against ~24 GiB, versus ~5.1 GiB for
FP8. That halves the KV budget and interacts with §5, but it blocks nothing.

**FP8 is a production-economics item and a later framework project, not a
correctness gate.**

---

## 8. What a vLLM bump would buy, and what it would cost

Bounded sizing only. **No bump is proposed here**; this is the fact needed to choose
between §4's options.

### What it deletes

A vLLM carrying the GLM-5.3-Flash cache API (present at `36fa72d2d0`) supplies:

- `cache_role=INDEXER` + `is_index_group_leader` → the indexer shares its MLA
  layer's group and block table, so **§4's `4096 / H` wall stops existing**.
- `KpoolTailSpec` with `max_num_blocks_per_req() == 1`, `uses_slot_mapping = False`,
  `prefix_cacheable = False` → the tail cache is expressible without breaking its
  `pos % kpool` semantics.
- A settable `storage_block_size` and `block_stride_alignment`.
- `MambaSpec.tp_replicated`, `num_prefill_checkpoint_blocks`,
  `prefill_checkpoint_alignment`, `tokens_per_state` → a richer recurrent-state
  contract than PR #54 had to work around.
- A complete reference implementation of the model, including MTP
  (`models/glm5next/common/mtp.py`) and the indexer, to diff against.

Rough effect: most of §2, most of §4, and the structural half of §6 become
configuration rather than code. §1.1 likely also evaporates (`glm5_next_text` would
be in the MLA allowlist), and §1.2's registration remains necessary either way.

### What it costs

- **The pin is not incidental.** The plugin's release branch targets vLLM 0.24.0 and
  Neuron SDK 2.32 together; `vllm_neuron/vllm/` reimplements the platform, worker,
  model runner, scheduler and patches against that surface. The 9,086-line
  `neuron_model_runner.py` in particular tracks vLLM's v1 worker closely.
- **PR #54 is written against 0.24.0**, including `_align_hybrid_page_sizes`'s
  reliance on `Platform._align_hybrid_block_size` and the `MultipleOf` stub backend.
  Both moved in the newer tree.
- **dev1's Qwen3.8-27B verification sits on that branch**, unverified end-to-end
  (no Linux host has run it yet). A bump invalidates the surface that work was
  validated against before it has been validated at all.
- The newer tree's layout differs structurally — models moved from
  `model_executor/models/` to `vllm/models/` — so this is a rebase of the plugin's
  vLLM-facing layer, not a version bump.

**[unverified]** No estimate of that rebase is offered. Sizing it means diffing
0.24.0 against the target version across `vllm/v1/worker`, `vllm/config` and
`vllm/platforms`, which is a scoped piece of work and not this document's.

### How that bears on §4

If a bump is coming within the horizon of this milestone, option (a) dominates and
recommendation (c) is throwaway work. If it is not, (c) is right and the divergence
from upstream is a known, priced cost. **That is the decision this analysis cannot
make for you**, and it is the one worth making first, because it determines whether
§2-§4 are written at all.

---

## 9. Open questions and unverified claims

Decisions needed:

1. **vLLM bump, yes or no** (§8). Determines whether §2-§4 get written.
2. **TP and `max_model_len`.** §4's page sizes and §5's capacity both move with TP.
   Everything above uses TP=64 and states the sensitivity.
3. **Whether 1M is in scope for first bring-up** given §5. A 128K configuration makes
   the capacity wall disappear and costs nothing in framework terms.
4. **§4's option (a) / (b) / (c).** Recommendation is (c), conditional on (1).

Marked `[unverified]`:

- The ~24 GiB/rank HBM figure (§5) assumes 4 logical cores share a chip's 96 GiB with
  no other consumer; not measured.
- Whether `MLAAttentionSpec.compress_ratio` at the pin really can express kpool
  compression (§4). Mechanically it looks identical to `tokens_per_state`; not
  exercised.
- Whether `index_n_heads: 32` — the live value, and *not* DeepSeek's 64 — affects
  anything in the cache layout. It does not appear in any page-size formula read
  here.
- The indexer's `quant_block_size`. Hardcoded 128 and asserted in the reference
  implementation (`attention.py:274,366`), but read from vLLM's code, not from our
  checkpoint.
- Whether `index_share_for_mtp_iteration: true` has consequences beyond MTP, which
  §6 descopes.
- The ZeroDivisionError predicted in §1.1 is traced through the source, not observed.
  It needs a Linux host to confirm, like everything else in the plugin
  (`import vllm_neuron` cannot work on macOS).

Carried from other branches, not re-derived here:

- The dense-sparse-attention exactness ceiling is **2051**, not the 2048 in
  STRATEGY.md and AGENTS.md (dev2).
- MLA decode has no kernel anywhere in `nkilib` — every file under
  `experimental/mla/deepseek/` is `*_cte`. That gates §2's final layout and is dev1's
  current work.
