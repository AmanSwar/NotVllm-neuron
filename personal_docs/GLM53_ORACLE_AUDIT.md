# GLM-5.3-Flash oracle — provenance audit

dev1, 2026-09-25. Branch `oracle-provenance-audit` off `dev`.

Audit only — **no fixes made here.** The oracle lives on `glm53-indexer`, which
dev2 owns; this branch carries the finding, not the change.

Prompted by the KDA output-gate bug (silu where both references say sigmoid,
fixed in `b50f62d`), and by dev2's observation that the blindness tracked
**provenance** rather than effort: the newest component (the indexer) was
cross-referenced against two independent implementations and was never wrong; the
oldest material, inherited verbatim from the NxDI port, was.

The question asked of every component: **is there an external oracle for this
value, or only internal consistency?**

---

## The headline: there are no known-unknowns

**`transformers` 5.17.0 ships `models/glm5_next/modeling_glm5_next.py`** (2,426
lines) with a **one-to-one counterpart for every component of the oracle**:

| oracle | transformers |
|---|---|
| `RMSNorm` / `UnweightedRMSNorm` / `RMSNormGated` | `Glm5NextTextRMSNorm` / `…UnweightedRMSNorm` / `…RMSNormGated` |
| `HyperConnection` / `hc_expand` | `Glm5NextTextHyperConnection` |
| `streams.mean(2)` | `Glm5NextTextHyperHead` |
| `ForgetGate` | `Glm5NextTextForgetGate` |
| `recurrent_kda` / `chunk_kda` | `recurrent_kimi_delta_attention` / `chunk_kimi_delta_attention` |
| `LinearAttention` | `Glm5NextTextLinearAttention` |
| `Indexer` | `Glm5NextTextIndexer` |
| `SparseMLAttention` | `Glm5NextTextAttention` |
| `MLP` | `Glm5NextTextMLP` |
| `TopkRouter` / `MoE` | `Glm5NextTextTopkRouter` / `…MoE` / `…Experts` |
| `DecoderLayer` / `FlashTextModel` | `Glm5NextTextDecoderLayer` / `…TextModel` |

So every gap below is a **cheap-test gap**, not a known-unknown. Nothing here
needs a reference that does not exist; it needs someone to use the one that does.

---

## The table

"Checked" = I diffed it against `modeling_glm5_next.py` during this audit.
"Distinguishing test?" = would the suite **as it stands** fail if this component
were wrong.

| # | Component | Provenance | External reference | Cross-referenced before this audit | Distinguishing test? | Verdict |
|---|---|---|---|---|---|---|
| 1 | `MLP` (dense + `shared_experts`) | NxDI, verbatim | `Glm5NextTextMLP` | no | **no — and a naive one would not either** | **BUG — no swiglu clamp** |
| 2 | `SparseMLAttention` | NxDI, verbatim | `Glm5NextTextAttention` | no | no | **divergence — softmax dtype** |
| 3 | `RMSNorm` | NxDI, verbatim | `Glm5NextTextRMSNorm` | no | no | **divergence — cast order** |
| 4 | `TopkRouter` | NxDI, verbatim | `Glm5NextTextTopkRouter` | no | no | correct, but on an **unguarded config assumption** |
| 5 | `ForgetGate` | NxDI, verbatim | `Glm5NextTextForgetGate` | no | no | matches |
| 6 | `MoE` expert path | NxDI, verbatim | `Glm5NextTextExperts` | no | no | matches |
| 7 | `UnweightedRMSNorm` | NxDI, verbatim | `…UnweightedRMSNorm` | no | no | matches |
| 8 | HyperHead (`streams.mean(2)`) | NxDI, verbatim | `Glm5NextTextHyperHead` | no | no | matches |
| 9 | `l2norm` | NxDI, verbatim | `l2norm` | no | no | matches (textually identical) |
| 10 | `RMSNormGated` | NxDI, verbatim | `…RMSNormGated` | **now yes** (dev2) | yes | was **WRONG**, fixed `b50f62d` |
| 11 | `HyperConnection` / `hc_expand` | NxDI, verbatim | vLLM `kernels/mhc/torch.py` | yes (vs vLLM) | yes | matches, 0.0–9.5e-7 |
| 12 | `Indexer` | dev2, new | `Glm5NextTextIndexer` + vLLM | yes (both) | yes | matches |
| 13 | `recurrent_kda` / `chunk_kda` | NxDI, verbatim | `*_kimi_delta_attention` | partly (dev2 vs nkilib GDN ref) | yes | matches |
| 14 | `LinearAttention` wiring | NxDI, verbatim | `…TextLinearAttention` | no | **not checked in this audit** | unknown |
| 15 | `DecoderLayer` / `FlashTextModel` | NxDI, verbatim | `…DecoderLayer` / `…TextModel` | no | **not checked in this audit** | unknown |

Rows 14–15 are honest gaps in *this audit*, not established matches. I compared
rows 1–9 line by line and did not get to the layer/model wiring.

---

## Finding 1 — `MLP` omits the swiglu clamp (a real bug)

```python
# oracle
def forward(self, x):
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# transformers — the comment is theirs
gate = gate.clamp(min=None, max=self.swiglu_limit)          # 10.0
up   = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
return self.down_proj(self.act_fn(gate) * up)
```

**Blast radius is larger than "the three dense layers".** `MLP` is used at:

- layers 0, 1, 2 (`first_k_dense_replace = 3`), and
- **`MoE.shared_experts` on all 42 MoE layers** — `self.shared_experts = MLP(...)`.

So the shared expert runs unclamped on every MoE layer. The routed experts *are*
clamped (`MoE.forward` does it inline and matches `Glm5NextTextExperts._apply_gate`
exactly), which is the same one-place-not-the-other pattern as the silu/sigmoid
bug.

### Why an external test would *also* have missed it unless deliberately scaled

The clamp is a no-op until activations exceed ±10. Measured:

| gate/up std | fraction clamped | mean rel. diff | max rel. diff |
|---|---|---|---|
| 1.0 | 0.000 | 0.00e+00 | 0.00e+00 |
| 3.0 | 0.002 | 8.8e-05 | 4.0e-01 |
| 5.0 | 0.089 | 9.7e-03 | 7.7e-01 |
| 8.0 | 0.378 | 7.5e-02 | 8.8e-01 |

`tiny_cfg` initialises weights at `normal_(0, 0.02)`, which keeps activations far
inside the limit. **A test that compared `MLP` against `Glm5NextTextMLP` on those
weights would pass with the bug present.** This one needs an external oracle *and*
inputs driven past the limit — the sharper form of the lesson.

## Finding 2 — `SparseMLAttention` does not force fp32 softmax

transformers (`eager_attention_forward`, line 1059):

```python
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

The oracle does `att.masked_fill(~mask, -inf).softmax(-1)` in the ambient dtype.
`scaling` matches exactly (`qk_head_dim ** -0.5`, and `qk_rope_head_dim = 0` so
`qk_head_dim = qk_nope_head_dim = 256`), and the LoRA/latent structure matches.
This is a precision divergence, not a structural one — but it is exactly the kind
a self-consistency test cannot see.

## Finding 3 — `RMSNorm` cast ordering

| | expression |
|---|---|
| transformers | `self.weight * hidden_states.to(input_dtype)` — cast **then** multiply |
| oracle | `(self.weight * (xf * rsqrt(...))).to(x.dtype)` — multiply in fp32 **then** cast |

Also differs in returned dtype (transformers returns the weight's dtype, the
oracle the input's). Both use plain `weight`, **not** `(1 + weight)` — worth
noting because Qwen3.5 uses `(1 + weight)` and that is precisely the kind of
sibling-model fact that has already caused two bugs here.

## Finding 4 — `TopkRouter`'s group logic is omitted on an unguarded assumption

transformers masks experts by group before the top-k; the oracle skips that
entirely. With `n_group = 1` and `topk_group = 1` — both verified in the live
config — the group logic is provably the identity, so the oracle is **correct**,
and its docstring says so.

But nothing asserts it. If `n_group` ever changed, the oracle would silently
route differently. A one-line assertion would convert a correct-by-luck into a
correct-by-construction.

---

## What the pattern says

Ordering the table by provenance reproduces dev2's observation exactly:

- **Newest, written against external references** (indexer, mHC): cross-checked,
  never wrong.
- **Inherited verbatim from the NxDI port**: 1 confirmed bug (`RMSNormGated`),
  1 more found here (`MLP`), 2 divergences, 1 unguarded assumption, and 2
  components nobody has looked at.

The `reference.py` docstring says *"The math is unchanged from that CPU-verified
version"* — which is what made this feel safe. "CPU-verified" in the NxDI fork
meant its own tests passed, not that it had been diffed against transformers.
**Provenance is not verification.**

## Suggested order of work (for whoever owns the oracle)

1. `MLP` clamp — a real bug on 45 of 45 layers, with a known blast radius.
2. Add the missing external-reference tests for rows 1–9. They are cheap: import
   `transformers.models.glm5_next.modeling_glm5_next`, build both, compare.
   **Drive the inputs hard enough to exercise clamps and saturation**, or the
   tests inherit the blindness they are meant to remove.
3. Rows 14–15 (`LinearAttention` wiring, `DecoderLayer`/`FlashTextModel`) —
   unaudited; someone should diff them.
4. The `n_group` assertion.
