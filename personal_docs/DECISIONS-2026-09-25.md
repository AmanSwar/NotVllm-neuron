# Four decisions on GLM-5.3-Flash — a memo

For Aman, 2026-09-25. Four open decisions, what each one actually turns on, and a
recommendation for each with the reasoning visible so you can disagree with the
reasoning rather than the conclusion.

Supporting detail is **not** repeated here. It lives in
`personal_docs/GLM53-FLASH-FRAMEWORK-GAP.md` (framework surfaces),
`personal_docs/MLA-DECODE-GAP.md` (the kernel), and `dev/progress/` (raw evidence).
Section references point there.

---

## What changed since these questions were posed

Five things, each of which moves at least one decision.

1. **FP8 is not consumable by the plugin at all.** No blockwise `[128,128]` weight path
   exists; its FP8 is per-projection static scales. So **first bring-up is BF16**, from
   dev1's offline converter — 643 GB rather than 328 GB. This changes the memory
   arithmetic underneath decisions 2 and 3, and it means "FP8 on the critical path" was
   wrong (AGENTS.md corrected).
2. **MLA decode is far less risky than the roadmap assumed.** `d_head = 512` is a
   *tested* nkilib configuration; `q_head = 64` validates; the absorbed-latent algebra is
   already implemented in the prefill kernels. And one latent tensor can serve as both K
   and V, bit-identically, so the cache needs one layout, not two.
3. **GLM-5.3-Flash's NoPE is what makes that true, and DeepSeek-family models do not
   inherit it.** 512 passes both `_MAX_D_HEAD = 512` and the "must be a multiple of 128"
   check; DeepSeek's 576 fails both. This bears on what gets ported *after* this — see
   the note at the end.
4. **bf16 numerics at long context are a non-issue.** Deviation from fp32 is 0.4-2% RMS
   and **independent of context length**; what governs it is attention peakedness, and
   `index_topk = 2048` caps that by construction. 1M context cannot amplify it.
5. **Milestone 1 closed on a compiler internal error**, `NCC_ISMP902`, reproduced on the
   real 64-layer graph. Read decision 1 carefully: this is *not* a vLLM problem.

---

## Decision 1 — Bump vLLM past 0.24.0?

**The question.** Stay on the pinned `vllm==0.24.0`, or rebase the plugin's vLLM-facing
layer onto a version that already ships GLM-5.3-Flash's cache API.

**What a bump buys.** Most of framework §2 and §4 and the structural half of §6 stop
being code: `cache_role=INDEXER` + `is_index_group_leader` (the indexer shares its MLA
layer's group, so §4's un-unifiable page size stops existing), `KpoolTailSpec`, a
settable `storage_block_size`, a richer `MambaSpec`, and a complete reference
implementation to diff against. Probably §1.1 too, since `glm5_next_text` would be in
the MLA allowlist.

**What it costs.** A rebase of `vllm_neuron/vllm/`, not a version bump: the 9,086-line
runner tracks vLLM's v1 worker closely, models moved from `model_executor/models/` to
`vllm/models/`, PR #54 is written against 0.24.0's `_align_hybrid_block_size` and its
`MultipleOf` stub, and dev1's Qwen3.8-27B work sits on that surface. Unestimated, and
estimating it means a real diff across `vllm/v1/worker`, `vllm/config`, `vllm/platforms`.

**Where I disagree with the earlier advice — in both directions.** The standing
recommendation was "defer until Milestone 1 closes on 0.24.0". Milestone 1 has now
closed as far as it can without hardware, and it closed on `NCC_ISMP902`. The inference
"so 0.24.0 doesn't clearly work, and the reason to stay is weaker" does not hold:

> `NCC_ISMP902` is a **compiler** bug, not a vLLM one. dev1's diagnosis is a pybind11
> overload-resolution failure **with an empty candidate list**, at `loc(unknown)` even
> under `XLA_IR_DEBUG`, in `neuronx-cc`'s own Python bindings, on the newest cp312 wheel
> available. The compiler's own message asks for an AWS support ticket. **A vLLM bump
> would not touch it.**

So the bump's case is neither strengthened nor weakened by Milestone 1. What Milestone 1
*does* invalidate is the **gate**: "wait until Milestone 1 closes" is not a usable
trigger, because Milestone 1 cannot close without either hardware or a resolution to a
compiler bug we cannot diagnose.

**Recommendation: do not bump now. Replace the gate with one cheap experiment.**

Run the Qwen3.8-27B compile on a **DLAMI or the Neuron dev container** instead of the
pip-assembled toolchain on wsl-box. dev1 already identified this and it is the decisive
test: `NCC_ISMP902` is either an artefact of assembling `neuronx-cc` + `torch-xla` +
`libtorch-neuronx-lite` by pip, in which case it evaporates and the pin is fine, or it is
real, in which case it is a support ticket and the toolchain's resolution may force a
move anyway — and at that point do both together rather than separately.

Two further reasons to wait: the bump only *bites* when framework §2-§4 get written,
which is Milestone 2 work, so the real deadline is "before anyone writes the latent cache
spec", not today. And a bump now would land on top of dev1's unverified Qwen work,
invalidating the only surface anything has been validated against.

**What deferring forecloses:** nothing permanent, but it commits us to writing §4's
option (c) — see decision 4 — which a later bump would throw away rather than rebase.
Price that as roughly 40 framework lines plus the model-side page arithmetic, not as a
milestone.

---

## Decision 2 — TP and `max_model_len`

**The question.** Both are inputs to every page-size and capacity number in the
framework document; nothing downstream can be sized without them.

**What we know.** Per-rank HBM is about **24 GiB** (96 GiB per chip ÷ 4 logical cores at
`logical_nc_config=2`) — **[unverified]**, assumes nothing else shares the chip. Weights
at BF16 (decision: see "what changed" #1) are 643 GB, so **10 GiB/rank at TP=64**. The
KDA recurrent state page is 67,840 B at TP=64, 135,680 at TP=32, **271,360 at TP=16**.
The MLA latent cache is replicated per rank at 11 KiB/token.

**Recommendation: TP=64, and `max_model_len = 131072` for first bring-up.**

TP=64 for three reasons, in order of weight:

- It **minimises the KDA state page** — which is the honest claim; it does **not** avoid
  the awkward factorisation, and that distinction matters. The page is exactly

  ```
  page(TP) = (64/TP) x 128 x 128 x 4   +   (24576/TP) x 3 x 2   =   265 x 2^14 / TP
                 recurrent, fp32              conv window, bf16        265 = 5 x 53
  ```

  so **the `5 x 53` factor is TP-invariant**: 67,840 at TP=64, 135,680 at TP=32, 271,360
  at TP=16 — all of them `265 x` a power of two. No TP choice makes the state page
  divide a sane attention page, so PR #54's grow-and-pad reconciliation is required
  either way. TP only scales the magnitude, and smaller is better: at TP=64 the smallest
  viable attention block size is 96 tokens.

  Worth noting separately that PR #54 reports the identical value, 271,360, for
  **Qwen3.5 at TP=4** — a different model at a different TP degree, and a coincidence of
  value rather than a measurement of ours. It is still weak evidence that this class of
  recurrent-state page factors badly in general, which is the only use made of it here.
- **TP=64 is also the ceiling**, not merely a choice. `kda_state_shape` divides the 64
  KDA heads across ranks, so TP must divide 64; above it there is no head per rank. The
  recommendation therefore sits at a hard boundary rather than in the middle of a range.
- It is the only setting where BF16 weights leave real headroom: 10 GiB of 24 GiB.
- The hardware is already a trn2.48xlarge; there is no saving from using less of it.
- **A second, independent constraint points the same way** (dev2, from the KDA decode
  kernel rather than from cache arithmetic). `gdn_tkg`'s SBUF envelope caps decode batch
  size as a function of heads-per-rank, and heads split across TP ranks *before* LNC
  sharding, so `heads_per_rank = 64 // TP`:

  | TP | 1 | 2 | 4 | 8 | 64 |
  |---|---|---|---|---|---|
  | max decode batch | 2 | 5 | 10 | 20 | 166 |

  Binding below TP=8; **at TP=64 it is not binding at all**. This matters because it is
  arrived at from a different direction than the page-size argument, so high TP is not
  resting on one line of reasoning. dev2 notes separately that `gdn_tkg`'s upstream test
  table covers a single case (`bh=24`), so none of this envelope is verified upstream at
  any TP.

**One thing to check before committing**, which I have not: 288 routed experts do not
divide 64 evenly (4.5 each). Whether that matters depends on the expert-parallel layout
rather than TP alone, and PR #55's author already flags 397B as needing experts sharded
across DP replicas. Flagging, not asserting.

At 128K the budget is comfortable: 10 GiB weights + 1.4 GiB latent KV + ~0.05 GiB
indexer, leaving roughly 12 GiB for activations, the KDA states and concurrency.

---

## Decision 3 — Is 1M context in scope for first bring-up?

**The question.** Milestone 2 is written as "text-only, FP8, long context". Does long
context mean 1M on the first attempt?

**What we now know, and it cuts both ways.**

- **1M is not dangerous.** The two things that looked frightening are resolved: bucketing
  needs **no framework change** (segmented prefill at ≤8192 plus a decode ladder of
  multiples of 128 is pure configuration), and bf16 numerics are **length-independent**
  with the governing quantity capped by `index_topk`.
- **1M is very expensive.** The latent cache is replicated per rank at 11 KiB/token, so
  one 1M-token sequence costs **11.5 GiB/rank**. Against ~24 GiB minus 10 GiB of BF16
  weights, that is one sequence, batch 1, nothing spare. At batch 1 the throughput has no
  product meaning. The NEFF ladder also grows to ~14 decode-context buckets, each a
  separate compiled graph with `fail_on_recompile` armed after warmup.

**Recommendation: no. Start at 128K. Make 1M a later milestone, explicitly gated on
decode context parallelism.**

Deferring costs **nothing structurally** — 128K and 1M need identical framework work, and
the difference is config plus NEFF count. DCP is the lever that makes 1M economic by
sharding the latent cache (`FullAttentionSpec.max_memory_usage_bytes` already divides by
`dcp_world_size`, and `MLAAttentionSpec` inherits it), and the plugin has partial DCP
support already.

**The one thing to preserve while deferring:** do not design the cache in a way that
forecloses DCP. Concretely, `dcp_stride = dcp_size x block_size` must divide every prefill
bucket, so DCP and decision 4's block-size choice interact and should be decided together
rather than sequentially.

---

## Decision 4 — Page-size reconciliation: (a), (b) or (c)

**The question.** GLM-5.3-Flash needs four cache kinds; vLLM 0.24.0 requires every KV
cache group to share one page size. The indexer's page cannot be unified **for any block
size** — the ratio is `4096 / H` with the block size cancelling, and the indexer's `H` is
132 bytes because of 4 bytes of inline FP8 scale per pool entry. 128 divides 4096; 132
does not. Structural, not a tuning problem. Framework §4 has the derivation.

- **(a) Bump vLLM** so the indexer shares the MLA layer's group and the problem
  disappears. **Removed by decision 1.**
- **(b) Keep four groups and pad** the indexer page to the common page. 31x waste on the
  one cache that grows with context — ~12 GiB/rank at 1M against a ~14 GiB budget.
  Rejected on arithmetic.
- **(c) Fold the indexer and tail caches into the MLA layer's own allocation.** vLLM then
  still sees two cache kinds — exactly the configuration PR #54 left working — and the
  model slices its own page.

**Recommendation: (c), and the case is stronger than when I first made it.** The tier-1
result showed one latent tensor serving as both K and V is accepted and **bit-identical**,
which is direct evidence that the model owning its own view of a shared page is workable
rather than merely permitted. It also keeps the change additive, which per PR #54's
experience is the difference between landing and rebasing forever.

**The cost, priced honestly.** The model takes on page arithmetic that vLLM would
otherwise own — three offsets inside one page, recomputed model-side. That is the same
debt PR #54 accepted for `page_major`, and its own comment warns that a second copy of
that arithmetic "is a silent memory-aliasing bug waiting to happen". It also puts prefix
caching over the indexer cache out of reach, and it is thrown away rather than rebased if
we later bump vLLM.

**One implementation note that falls out of tier 1:** reading may alias, writing should
not. One buffer may serve as both K and V on the read path; on the write path, passing it
as both `k_cache` and `v_cache` to PR #40's scatter would issue two identical scatters and
hand the aliasing pass two outputs on one buffer — the hazard that kernel exists to
prevent. Hence the latent-only scatter variant (~150-200 lines).

---

## What unblocks what

Not a plan — the plan depends on your answers. Just the dependency order:

1. **The DLAMI/container compile test** (decision 1's new gate) is the cheapest
   high-value experiment left on the whole project, and it is currently blocked only on
   not having that environment. It decides whether Milestone 1 is finished or merely
   parked.
2. **Decisions 2 and 3 are answerable today** and unblock all of framework §2-§4.
3. **Decision 4 only needs deciding before the latent cache spec is written.**
4. **MLA decode's one open item** — executing the kernel at `q_head = 64` under the
   simulator — is a narrow increment over upstream's own coverage and needs a box window.
   Deliberately left open.

## The implicit fifth decision: what gets ported after this

Not one of your four, but it follows from #3 in "what changed" and I would rather it be
explicit. The roadmap orders GLM-5.3 (`glm_moe_dsa`, all-MLA) before GLM-5.3-Flash partly
because GLM-5.3 rides AWS's DeepSeek kernel investment. **On the decode side that
reasoning is inverted:** GLM-5.3 and every DeepSeek-family model need a 576-wide decode
path that does not exist and is ragged against the hardware tile, while Flash needs 512,
which is supported and tested. Flash remains harder on mHC, KDA, the indexer and four
cache kinds — so this does not by itself reorder anything. But the specific justification
for the ordering was wrong.

## What this memo does not claim

- **Whether 0.4-2% RMS bf16 deviation is acceptable end-to-end.** That is a model-quality
  question for `vllm_neuron/accuracy/logit_validation.py` against the HF oracle, not a
  kernel comparison, and it needs the real checkpoint plus a Linux host.
- **Any performance number.** Both KV fast paths are off at `d_head = 512`
  (`use_dma_transpose` False, QK-swap structurally excluded). That is a shape, not a
  number, and we have no number.
- **That the ~24 GiB/rank figure is measured.** It is arithmetic from AWS's published
  per-chip HBM divided by logical cores, and it assumes no other consumer.
- **That anything has run on Neuron hardware.** Nothing has.
