# GLM-5.3-Flash oracle — audit by constant

dev1, 2026-09-25. Branch `oracle-provenance-audit` off `dev`.

Audit only — **no fixes made here.** The oracle lives on `glm53-indexer`, which
dev2 owns; this branch carries the finding, not the change.

## Why this is a table of constants, not of components

The KDA output-gate bug is the worked example, and it disqualifies component-level
auditing:

> `RMSNormGated` had an external oracle available the whole time. transformers
> implements it (`Glm5NextTextRMSNormGated`, with `self.activation = "sigmoid"`
> spelled out on its own line) and so does vLLM. The component was never
> unreferenced. It was wrong because **no test was ever pointed at that one
> value**.
>
> Worse, the suite *looked* covered: `test_gate_is_per_channel_not_scalar` has a
> 159,436x discriminating margin and passes. But it discriminates the **decay**
> gate — `ForgetGate`'s per-channel `exp(g)` — while the thing that was wrong was
> the **output** gate, `o_norm`'s activation. Different gate, similar name.
>
> **A component-level audit would have scored `RMSNormGated` green.**

So the question asked below is per constant and per activation choice: *is there a
test whose two sides could disagree about this specific value?* Credit to dev2 for
the reframing.

## Method

Every row diffed against `transformers` 5.17.0
`models/glm5_next/modeling_glm5_next.py` (2,426 lines), which ships a one-to-one
counterpart for every component of the oracle, and against the live
`zai-org/GLM-5.3-Flash` `config.json`. "Test could disagree?" is judged against
the suite as it stands on `glm53-indexer`.

One structural note: **no test imports transformers.** The oracle is deliberately
transformers-free, so external references are *vendored* (`tests/gdn_refs.py` from
nkilib, `tests/indexer_refs.py` from transformers + vLLM). That is a sound design,
but it means external coverage exists only where someone explicitly vendored a
reference — which happened for exactly two areas, KDA and the indexer.

---

## The table

`agrees?` ✓ = matches transformers. ✗ = does not.
`test?` = would the suite **as it stands** fail if this value were wrong.

### Norms and activations

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 1 | `rms_norm_eps` | `cfg.rms_norm_eps` = 1e-5 | config; tf same | ✓ | **no** |
| 2 | RMSNorm scale form | plain `weight` | tf plain `weight` | ✓ | **no** — and this is a Qwen trap: Qwen3.5 uses `(1 + weight)` |
| 3 | RMSNorm cast order | `(w · x_fp32).to(dt)` | tf `w * x.to(input_dtype)` | **✗** | **no** |
| 4 | `UnweightedRMSNorm` eps | `cfg.rms_norm_eps` | tf passes `config.rms_norm_eps` too | ✓ | **no** |
| 5 | `RMSNormGated` activation | `sigmoid` | tf `self.activation = "sigmoid"` | ✓ **now** | **yes** (dev2, `b50f62d`) — was `silu` |
| 6 | `o_norm` eps | `cfg.rms_norm_eps` | tf `layer_norm_epsilon` | ✓ | **no** |
| 7 | `l2norm` form | `x * rsqrt(Σ + eps)` | tf `x / sqrt(Σ + eps)` — *"intentionally use sqrt and / to match original triton"* | **✗** | **no** |
| 8 | `l2norm` eps | 1e-6 | tf 1e-6 | ✓ | **no** |
| 9 | conv activation | hardcoded `F.silu` | tf `config.hidden_act` (= `"silu"`) | ✓ value, hardcoded | **no** |

### KDA

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 10 | `beta` | `sigmoid(b_proj(x))` | tf same | ✓ | partial (`gdn_refs`) |
| 11 | `gate_lower_bound` | `cfg` = −5.0 | config; tf same | ✓ | partial |
| 12 | ForgetGate activation | `lower · sigmoid(decay · g)` | tf same | ✓ | partial |
| 13 | softplus branch threshold | 20.0 | tf 20.0 | ✓ | **no** |
| 14 | gate is **per channel** `[B,S,H,K]` | yes | tf same | ✓ | **yes** |
| 15 | recurrence / chunked scan | — | nkilib GDN torch ref | ✓ | **yes** (dev2) |

### mHC

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 16 | `hc_mult` 4 / `hc_sinkhorn_iters` 20 / `hc_eps` 1e-6 | from cfg | config; vLLM | ✓ | **yes** |
| 17 | post multiplier | hardcoded `2 *` | vLLM `mhc_post_mult_value = 2.0` | ✓ | **yes** |
| 18 | mix width | `(2 + H) · H` | vLLM `mix_hc = (2 + n) · n` | ✓ | **yes** |
| 19 | HyperHead collapse | `streams.mean(2)` | tf `Glm5NextTextHyperHead` — unweighted mean | ✓ | **no** |

### Sparse-MLA

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 20 | attention scale | `qk_head_dim ** -0.5` | tf identical (`qk_rope = 0`, so 256) | ✓ | **no** |
| 21 | softmax dtype | ambient (bf16) | tf forces `dtype=torch.float32` | **✗** | **no** |
| 22 | latent slice / NoPE | `[..., :kv_lora_rank]`, no RoPE | config `qk_rope_head_dim = 0` | ✓ | **no** |

### MoE and MLP

| # | Constant / choice | Oracle | Authority | agrees? | test? |
|---|---|---|---|---|---|
| 23 | `swiglu_limit` in routed experts | 10.0, clamped | tf `_apply_gate` identical | ✓ | **no** |
| 24 | `swiglu_limit` in `MLP` | **absent** | tf `Glm5NextTextMLP` clamps | **✗ BUG** | **no — and a naive external test would also pass** |
| 25 | `routed_scaling_factor` | `cfg` = 2.5 | config; tf same | ✓ | **no** |
| 26 | top-k norm denominator | `+ 1e-20` | tf `+ 1e-20` | ✓ | **no** |
| 27 | scoring function | `sigmoid` | config `scoring_func: sigmoid` | ✓ | **no** |
| 28 | group masking (`n_group`) | omitted | tf masks by group | ✓ *only because* `n_group = 1` | **no** — unguarded |
| 29 | `first_k_dense_replace` | `cfg` = 3 | config | ✓ | **no** |

**Not audited:** `LinearAttention` wiring, `DecoderLayer`, `FlashTextModel`. I
diffed rows 1–29 and did not reach the layer/model wiring. Recorded as unknown,
not as matching.

---

## The four findings, in order of severity

### 1. `MLP` omits the swiglu clamp — a real bug on 45 of 45 layers

```python
# oracle
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
# transformers — the comment is theirs
gate = gate.clamp(min=None, max=self.swiglu_limit)          # 10.0
up   = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
```

`MLP` is used at layers 0–2 (`first_k_dense_replace = 3`) **and as
`MoE.shared_experts` on all 42 MoE layers**, so the shared expert runs unclamped
everywhere. The routed experts *are* clamped and match exactly — the same
one-place-not-the-other shape as the gate bug.

**And an external test would also have missed it.** The clamp is a no-op until
activations exceed ±10:

| gate/up std | fraction clamped | mean rel. diff |
|---|---|---|
| 1.0 | 0.000 | 0.00e+00 |
| 3.0 | 0.002 | 8.8e-05 |
| 5.0 | 0.089 | 9.7e-03 |
| 8.0 | 0.378 | 7.5e-02 |

`tiny_cfg` initialises at `normal_(0, 0.02)`, keeping activations far inside the
limit, so comparing `MLP` against `Glm5NextTextMLP` on those weights **passes with
the bug present**. This row needs an external oracle *and* inputs driven past the
limit. An external reference is necessary but not sufficient.

### 2. `l2norm` uses the Qwen form (row 7)

transformers computes `x / sqrt(Σ + eps)` and comments *"main difference to qwen's
gdn variation: intentionally use sqrt and / to match original triton"*. The oracle
uses `x * rsqrt(Σ + eps)` — precisely the Qwen variant transformers is warning
about. ULP-level, but transformers considered it worth pinning, and this is the
**third** Qwen→GLM carry-over after the output gate and the 2051 ceiling.

### 3. Softmax and cast-order divergences (rows 3, 21)

Sparse-MLA does not force fp32 softmax where transformers does; `RMSNorm` casts in
the other order. Both are precision divergences invisible to self-consistency.

### 4. `n_group` is an unguarded assumption (row 28)

Correct today and the docstring says why, but nothing asserts `n_group == 1`. One
assertion converts correct-by-luck into correct-by-construction.

---

## What the pattern says

Sorted by provenance, the diagnostic holds exactly:

- **Written against an external reference** (indexer, mHC): rows 16–18 and the
  indexer — cross-checked, none wrong.
- **Inherited verbatim from the NxDI port**: 1 confirmed bug (row 5), 1 found here
  (row 24), 2 divergences (rows 3, 7, 21), 1 unguarded assumption (row 28), and
  20 of 29 rows with no test that could disagree.

`reference.py` says *"The math is unchanged from that CPU-verified version"*.
"CPU-verified" in the NxDI fork meant its own tests passed — not that it had been
diffed against transformers. **Provenance is not verification.**

## Suggested order of work

1. Row 24, the `MLP` clamp — a real bug with a known blast radius.
2. Rows 7, 3, 21 — decide whether to match transformers exactly or record the
   divergence deliberately. Either is defensible; silently differing is not.
3. Vendor a transformers reference for the norms, MLP, router and MLA the way
   `indexer_refs.py` already does, and **drive inputs hard enough to exercise
   clamps and saturation** or the tests inherit the blindness they remove.
4. Row 28's assertion.
5. Audit `LinearAttention` wiring, `DecoderLayer`, `FlashTextModel`.
