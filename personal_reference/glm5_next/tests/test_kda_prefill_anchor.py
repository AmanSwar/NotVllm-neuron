# SPDX-License-Identifier: Apache-2.0
"""The anchored sub-block factorisation KDA prefill needs, validated in torch.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_kda_prefill_anchor.py -v -s

THE PROBLEM. Adapting nkilib's ``gdn_cte`` to KDA is a rewrite, not a port, because of
one thing. GDN builds ``decay[i,j] = exp(cg[i] - cg[j])`` as a ``[CHUNK, CHUNK]``
matrix and forms the intra-chunk operator as ``A = -beta * (k @ k^T) * decay`` -- two
cheap ops. KDA's decay is per-channel, so what is actually needed is

    A[i,j] = -beta[i] * sum_d  k[i,d] * k[j,d] * exp(cg[i,d] - cg[j,d])

with the decay **inside the d-sum**, where it does not factor out. Done literally it
needs a ``[CHUNK, CHUNK, K]`` intermediate: 64x64x128 fp32 = 2 MB per chunk per head.

THE FIX, which FLA/vLLM already use. Anchor the gate at each row-block's first token:

    exp(cg[i,d] - cg[j,d]) = exp(cg[i,d] - cg_n[d]) * exp(cg_n[d] - cg[j,d])

Pre-scale ``k`` on each side by its own factor and ``A = -(Kb' @ K''^T)`` is a **plain
matmul** again. Verified below to reproduce ``chunk_kda``'s operator to fp32 noise.

THE SUB-BLOCK SIZE IS DERIVED HERE, NOT INHERITED. nkilib's ``_SUBBLK = 16`` is sized
for its nilpotent recursive-doubling solve; FLA's ``BC = min(16, BT)`` is unexplained
in source. Both happen to be 16, which is a coincidence of two different arguments and
not a reason. The constraint that actually applies to *this* factorisation:

* Row factors ``exp(cg[i] - cg_n)`` for ``i >= n`` are **<= 1** (cg decreases). Safe;
  underflow is benign, since a decay too small to represent is one that does not matter.
* Column factors ``exp(cg_n - cg[j])`` on the **diagonal block** have ``j >= n``, so
  they are **>= 1** and grow with the distance to the anchor. This is the binding side.
* Worst case over ``BC`` tokens is ``exp(BC * |gate_lower_bound|)``, so safety requires

      BC * |gate_lower_bound|  <  ln(float32_max) = 88.7

  With GLM's ``gate_lower_bound = -5.0`` that is **BC <= 17**, and the largest power of
  two is **16**. Measured: BC 4/8/16 exact, BC 32 overflows to NaN even on real gates.

So 16 is correct for GLM for a reason that has nothing to do with either source's
reason -- and it is **config-derived**: a different ``gate_lower_bound`` moves it
(-6.0 would give 14, and 16 would overflow). fp32 and bf16 give the same bound, since
they share an exponent range.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

CHUNK = 64
K = 128
F32_MAX = torch.finfo(torch.float32).max
LN_F32_MAX = math.log(F32_MAX)


def max_safe_subblock(gate_lower_bound, dtype=torch.float32):
    """Largest sub-block whose anchored column factor cannot overflow ``dtype``."""
    return int(math.log(torch.finfo(dtype).max) / abs(gate_lower_bound))


def _inputs(seed=0, decay="real", B=1, Hh=2, T=CHUNK):
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    k = R.l2norm(r(B, Hh, T, K))
    beta = torch.sigmoid(r(B, Hh, T))
    if decay == "real":
        g = -5.0 * torch.sigmoid(r(B, Hh, T, K))
    else:                                   # adversarial: every gate pinned at the bound
        g = torch.full((B, Hh, T, K), -5.0)
    return k, beta, g


def exact_operator(k, beta, g):
    """``chunk_kda``'s intra-chunk operator, with the decay inside the d-sum."""
    cg = g.cumsum(-2)
    k_beta = k * beta[..., None]
    stri = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), 1)
    tri = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), 0)
    decay = (cg.unsqueeze(-2) - cg.unsqueeze(-3)).masked_fill(stri[..., None], float("-inf")).exp()
    return -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay).sum(-1).masked_fill(tri, 0)


def anchored_operator(k, beta, g, BC):
    """The same operator as PLAIN MATMULS of anchored, pre-scaled blocks.

    ``CHUNK % BC == 0`` is required and asserted: a non-dividing BC silently leaves
    trailing tokens uncovered, which cost me a spurious "BC=17 is numerically unstable"
    result before I noticed the loop simply skipped 13 tokens.
    """
    assert CHUNK % BC == 0, f"BC={BC} must divide CHUNK={CHUNK}; a remainder is dropped"
    B, Hh = k.shape[:2]
    cg = g.cumsum(-2)
    k_beta = k * beta[..., None]
    A = torch.zeros(B, Hh, CHUNK, CHUNK)
    for bi in range(CHUNK // BC):
        n = bi * BC
        anchor = cg[..., n : n + 1, :]
        rows = slice(n, n + BC)
        kb_s = k_beta[..., rows, :] * (cg[..., rows, :] - anchor).exp()     # <= 1
        for bj in range(bi + 1):
            cols = slice(bj * BC, bj * BC + BC)
            k_s = k[..., cols, :] * (anchor - cg[..., cols, :]).exp()       # >= 1 on diagonal
            A[..., rows, cols] = -(kb_s @ k_s.transpose(-1, -2))            # PLAIN MATMUL
    return A.masked_fill(torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), 0), 0)


def _worst_growth(g, BC):
    cg = g.cumsum(-2)
    return max((cg[..., bi * BC : bi * BC + 1, :] - cg[..., bi * BC : (bi + 1) * BC, :]).max().item()
               for bi in range(CHUNK // BC))


# ------------------------------------------------------- the factorisation is exact
@pytest.mark.parametrize("decay", ["real", "adversarial"])
@pytest.mark.parametrize("BC", [4, 8, 16])
def test_anchored_factorisation_reproduces_the_exact_operator(decay, BC):
    """Plain matmuls of pre-scaled blocks == the decay-inside-the-sum form."""
    k, beta, g = _inputs(decay=decay)
    ref = exact_operator(k, beta, g)
    got = anchored_operator(k, beta, g, BC)
    assert ref.abs().max() > 1e-3, "operator is ~zero; comparison would be vacuous"
    rel = (got - ref).abs().max().item() / ref.abs().max().item()
    assert torch.isfinite(got).all()
    assert rel < 1e-5, f"BC={BC} {decay}: relative error {rel:.2e}"


def test_the_anchor_position_bounds_the_range_not_the_answer():
    """A result I did not expect, and it sharpens why BC is the bound.

    Moving the anchor back by one block is **algebraically the identity** -- the two
    factors ``exp(cg_i - cg_wrong)`` and ``exp(cg_wrong - cg_j)`` still multiply to
    ``exp(cg_i - cg_j)`` for any anchor. So a "wrong" anchor cannot change the answer.

    What it changes is the RANGE each factor has to span: an anchor one block early
    makes the column factor reach across ``2 * BC`` tokens, which overflows exactly as
    ``BC = 32`` does. So the anchor is not a correctness choice at all -- it is purely
    a numerical-range choice, and that is precisely why the bound is on the SPAN from
    anchor to token rather than on the block size as such.
    """
    k, beta, g = _inputs(seed=3)
    cg = g.cumsum(-2)
    B, Hh = k.shape[:2]
    BC = 16
    A = torch.zeros(B, Hh, CHUNK, CHUNK)
    k_beta = k * beta[..., None]
    for bi in range(CHUNK // BC):
        n = bi * BC
        wrong = cg[..., max(n - BC, 0) : max(n - BC, 0) + 1, :]      # anchor one block early
        rows = slice(n, n + BC)
        kb_s = k_beta[..., rows, :] * (cg[..., rows, :] - wrong).exp()
        for bj in range(bi + 1):
            cols = slice(bj * BC, bj * BC + BC)
            k_s = k[..., cols, :] * (wrong - cg[..., cols, :]).exp()
            A[..., rows, cols] = -(kb_s @ k_s.transpose(-1, -2))
    finite = bool(torch.isfinite(A).all())
    print(f"\n  anchor one block early at BC=16: finite={finite} "
          f"(same algebra, twice the span -- overflows like BC=32)")
    assert not finite, (
        "an early anchor no longer overflows; the span-based bound needs re-deriving"
    )


# --------------------------------------------------------------- the sub-block bound
def test_the_bound_is_derived_from_gate_lower_bound():
    """BC * |gate_lower_bound| < ln(float_max). Config-derived, not inherited."""
    cfg = R.FlashCfg()
    assert cfg.linear_lower_bound == -5.0
    assert max_safe_subblock(cfg.linear_lower_bound) == 17
    assert max_safe_subblock(cfg.linear_lower_bound, torch.bfloat16) == 17, (
        "fp32 and bf16 share an exponent range, so the bound is the same"
    )
    # the bound moves with the config, which is the point
    assert max_safe_subblock(-6.0) == 14 and max_safe_subblock(-4.0) == 22
    chosen = 16
    assert chosen <= max_safe_subblock(cfg.linear_lower_bound)
    assert chosen * 2 > max_safe_subblock(cfg.linear_lower_bound), "16 is the largest safe power of two"


@pytest.mark.parametrize("decay", ["real", "adversarial"])
def test_an_oversized_subblock_overflows(decay):
    """BC=32 is not merely inadvisable, it produces NaN -- on REAL gates, not just
    adversarial ones. This is what makes 16 a bound rather than a preference."""
    k, beta, g = _inputs(decay=decay)
    got = anchored_operator(k, beta, g, 32)
    growth = math.exp(min(_worst_growth(g, 32), 700))
    print(f"\n  {decay:>11} BC=32: worst growth exp() = {growth:.2e} vs fp32 max {F32_MAX:.2e}")
    assert growth > F32_MAX
    assert not torch.isfinite(got).all(), "BC=32 did not overflow; re-derive the bound"


@pytest.mark.parametrize("decay", ["real", "adversarial"])
def test_bc_16_has_real_but_finite_headroom(decay):
    """Recorded because it is thinner than it looks: ~6 orders of magnitude on
    adversarial gates. Not comfortable, and it is why the bound is worth asserting."""
    k, beta, g = _inputs(decay=decay)
    growth = math.exp(min(_worst_growth(g, 16), 700))
    headroom = F32_MAX / growth
    print(f"  {decay:>11} BC=16: growth {growth:.2e}, headroom {headroom:.2e}x")
    assert headroom > 1.0
    if decay == "adversarial":
        assert headroom < 1e12, "headroom is larger than measured; re-check the gate bound"


def test_non_dividing_subblock_is_rejected():
    """The guard for the harness bug that nearly became a false finding."""
    k, beta, g = _inputs()
    with pytest.raises(AssertionError, match="must divide"):
        anchored_operator(k, beta, g, 17)
