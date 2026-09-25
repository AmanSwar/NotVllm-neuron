# SPDX-License-Identifier: Apache-2.0
"""Is running the 11 sparse-MLA layers dense exact, and up to what length?

Run: python3 -m pytest personal_reference/glm5_next/tests/test_dense_mla_exactness.py -v

**Answer: yes, and the ceiling is seq_len <= index_topk == 2048.** Above that the
oracle is not approximate, it is *wrong* — it attends to tokens the model's
indexer would have excluded.

This was checked against vLLM's implementation rather than by re-deriving the
arithmetic, because the arithmetic was never the risk. The risk was the meaning of
``index_kpool_compress``: if attention ran over *compressed pool representatives*
then selecting every pool would still be lossy and the dense-first plan would be
built on sand.

It does not. vLLM keeps three separate caches
(`vllm/models/glm5next/common/attention.py`):

* ``Glm5NextIndexerCache`` — "Indexer K cache that stores kpool-compressed
  entries", ``tokens_per_state = index_kpool``. This is the compressed one and it
  exists **only to score**.
* ``Glm5NextTailCache`` — "Holds the trailing incomplete pool's **raw K** + gate
  score: one block of ``index_kpool`` slots per request, overwritten in place by
  ``pos % kpool``", and explicitly "**not** the fp8-compressed entry, which lives
  in ``Glm5NextIndexerCache``". So ``index_kpool_always_select_tail`` appends the
  *real tail tokens* at token granularity, not another pool.
* the ordinary MLA latent KV cache — full fidelity, and what attention gathers.

The indexer's output is ``topk_indices_buffer``, which holds **token** indices.
Compression never reaches attention.

Better still, vLLM implements the same shortcut we want, in both directions
(`vllm/models/glm5next/nvidia/sparse_indexer.py`,
`vllm/models/glm5next/common/sparse_indexer.py`):

* prefill: "Short sequences select every pool, so skip sparse scoring and fill the
  top-k buffer with all causal token indices", gated on
  ``max_prefill_seq_len <= topk_tokens``;
* decode: ``_fill_short_decode_causal_indices`` — "Fill **exact** causal rows when
  sparse decode would select every token", gated on ``max_seq_len > topk_tokens``
  returning False.

and ``self.topk_tokens = config.index_topk``.

So this is not our shortcut being tolerated; it is the reference implementation's
own fast path, taken for the same reason.

### What it depends on

The tests below pin each of these so a config change trips them:

1. ``index_topk == 2048`` — vLLM compares ``seq_len <= topk_tokens`` directly, and
   ``topk_tokens`` *is* ``index_topk``.
2. ``index_topk % index_kpool == 0``. The pool-count derivation
   (``ceil(S / kpool) <= index_topk // kpool``) and vLLM's direct
   ``S <= index_topk`` give the same 2048 **only because 2048 % 4 == 0**. With a
   non-dividing ``index_kpool`` they diverge and vLLM's is authoritative.
3. ``index_kpool_compress`` stays confined to the indexer cache.
4. ``index_kpool_always_select_tail`` keeps appending raw tail tokens.
5. ``indexer_types`` is ``"full"`` on all 45 layers — a layer with a different
   indexer type is not covered by this argument.

### Consequence for the roadmap

Milestone 2's scope is **1M context**. This oracle is only valid to 2048 tokens,
so it cannot validate long-context behaviour at all. Anything past 2048 needs a
real indexer in the oracle. That is a scoping fact, not a defect.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import pytest

# Values verified against the live zai-org/GLM-5.3-Flash config.json on 2026-09-25.
INDEX_TOPK = 2048
INDEX_KPOOL = 4
NUM_LAYERS = 45
DENSE_EXACT_MAX_SEQ_LEN = INDEX_TOPK


def _live_config():
    """The real config, if a copy is available; otherwise the verified constants."""
    path = os.environ.get("GLM53F_CONFIG")
    if path:
        return json.loads(pathlib.Path(path).read_text())["text_config"]
    return {
        "index_topk": INDEX_TOPK,
        "index_kpool": INDEX_KPOOL,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "indexer_types": ["full"] * NUM_LAYERS,
        "num_hidden_layers": NUM_LAYERS,
    }


def test_threshold_is_index_topk():
    """vLLM's gate is ``seq_len <= topk_tokens`` and ``topk_tokens = index_topk``."""
    cfg = _live_config()
    assert cfg["index_topk"] == INDEX_TOPK
    assert DENSE_EXACT_MAX_SEQ_LEN == cfg["index_topk"]


def test_pool_derivation_agrees_with_vllms_direct_check():
    """Both routes give 2048 — but only because index_topk divides by index_kpool.

    Pool route: n_pools = ceil(S / kpool) must be <= select_k = index_topk // kpool.
    vLLM route: S <= index_topk.
    """
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    assert topk % kpool == 0, (
        f"index_topk {topk} no longer divides by index_kpool {kpool}; the pool-count "
        f"derivation and vLLM's S <= index_topk check now disagree, and vLLM's is "
        f"authoritative. Re-derive the ceiling."
    )
    select_k = topk // kpool
    # largest S with ceil(S / kpool) <= select_k
    largest = select_k * kpool
    assert largest == topk == DENSE_EXACT_MAX_SEQ_LEN

    # and it really is the largest: one more token needs one more pool
    assert math.ceil((largest + 1) / kpool) > select_k


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 5, 64, 2047, 2048])
def test_every_pool_selected_at_or_below_the_ceiling(seq_len):
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    n_pools = math.ceil(seq_len / kpool)
    assert n_pools <= topk // kpool, f"S={seq_len} needs {n_pools} pools"


@pytest.mark.parametrize("seq_len", [2049, 4096, 1 << 20])
def test_above_the_ceiling_the_oracle_is_wrong_not_approximate(seq_len):
    """Past 2048 the indexer genuinely excludes tokens the dense oracle attends to."""
    cfg = _live_config()
    topk, kpool = cfg["index_topk"], cfg["index_kpool"]
    n_pools = math.ceil(seq_len / kpool)
    assert n_pools > topk // kpool
    excluded_pools = n_pools - topk // kpool
    assert excluded_pools > 0, (
        f"S={seq_len}: expected the indexer to exclude pools; if it does not, "
        f"the ceiling is higher than documented"
    )


def test_compress_and_tail_flags_are_as_assumed():
    """The argument depends on what these two flags mean, not just their value.

    ``index_kpool_compress`` must remain confined to the indexer's scoring cache
    (vLLM: ``Glm5NextIndexerCache``), and ``index_kpool_always_select_tail`` must
    append raw tail tokens (vLLM: ``Glm5NextTailCache`` stores "raw bf16 K ... not
    the fp8-compressed entry"). If either changes, exactness has to be re-argued.
    """
    cfg = _live_config()
    assert cfg["index_kpool_compress"] is True
    assert cfg["index_kpool_always_select_tail"] is True
    assert cfg["index_kpool"] == INDEX_KPOOL


def test_all_layers_use_the_full_indexer():
    """A layer with a different indexer type is outside this argument."""
    cfg = _live_config()
    types = cfg["indexer_types"]
    assert len(types) == cfg["num_hidden_layers"] == NUM_LAYERS
    assert set(types) == {"full"}, f"unexpected indexer types: {sorted(set(types))}"


def test_oracle_cannot_cover_the_million_token_target():
    """Recorded deliberately: Milestone 2 wants 1M context, the oracle reaches 2048."""
    assert DENSE_EXACT_MAX_SEQ_LEN < (1 << 20)
