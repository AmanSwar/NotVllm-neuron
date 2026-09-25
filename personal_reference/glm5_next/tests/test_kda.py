# SPDX-License-Identifier: Apache-2.0
"""KDA: chunked prefill must equal the single-step recurrence.

Run: python3 -m pytest personal_reference/glm5_next/tests/test_kda.py -v

Three things this file is careful about, because a weak version of each passes
while proving nothing:

1. **Chunk count.** ``chunk_kda`` pads to a multiple of ``chunk`` and carries state
   between chunks in a Python loop. At T < chunk that loop runs once and the carry
   is never exercised. Every tolerance test below spans 1 -> 4+ chunks and asserts
   the error does not grow with chunk count.
2. **Decay saturation.** GLM's real gate is ``-5.0 * sigmoid(...)``, so ``exp(g)``
   sits near 0.08 and the state is forgotten within a few tokens — which makes the
   carry almost irrelevant and any carry bug invisible. A long-memory regime
   (``exp(g) -> 1``) is tested explicitly.
3. **Per-channel vs scalar gate.** KDA's decay is per K-channel; GDN's is per-head
   scalar. ``test_gate_is_per_channel_not_scalar`` substitutes the scalar form and
   asserts the comparison *notices*, so the suite cannot silently accept a GDN
   kernel dropped in where KDA was meant.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from glm5_next import reference as R  # noqa: E402

# Real GLM-5.3-Flash KDA geometry: 64 heads x head_dim 128, K == V.
H, K = 64, 128
CHUNK = 64


def _inputs(T, B=2, heads=H, dim=K, decay="real", seed=0):
    """Random KDA inputs shaped as the layer produces them: [B, T, H, dim].

    decay="real"  -> exp(g) ~ 0.08, GLM's actual gate (state forgotten fast)
    decay="long"  -> exp(g) in [0.95, 1.0], state persists across chunks
    """
    g_ = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g_)
    q, k, v = r(B, T, heads, dim), r(B, T, heads, dim), r(B, T, heads, dim)
    if decay == "real":
        # g = lower * sigmoid(exp(A_log) * raw), lower = -5.0  (ForgetGate)
        g = -5.0 * torch.sigmoid(r(B, T, heads, dim))
    else:
        g = torch.empty(B, T, heads, dim).uniform_(0.95, 1.0, generator=g_).log()
    beta = torch.sigmoid(r(B, T, heads))
    state = r(B, heads, dim, dim) * 0.1
    return q, k, v, g, beta, state


def _run_recurrent(q, k, v, g, beta, state):
    """Drive recurrent_kda one token at a time -> ([B,T,H,V], final_state)."""
    outs = []
    s = state.clone()
    for t in range(q.shape[1]):
        o, s = R.recurrent_kda(
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
            g[:, t : t + 1], beta[:, t : t + 1], s,
        )
        outs.append(o)
    return torch.cat(outs, 1), s


def _max_diff(T, decay, seed=0, heads=H, dim=K):
    q, k, v, g, beta, s0 = _inputs(T, heads=heads, dim=dim, decay=decay, seed=seed)
    out_c, s_c = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_r, s_r = _run_recurrent(q, k, v, g, beta, s0)
    return (
        (out_c - out_r).abs().max().item(),
        (s_c - s_r).abs().max().item(),
        out_r.abs().max().item(),
    )


# --------------------------------------------------------------------- tolerance
@pytest.mark.parametrize("T", [64, 128, 192, 256])
@pytest.mark.parametrize("decay", ["real", "long"])
def test_chunk_matches_recurrent(T, decay):
    """The two paths agree to ~1e-7 regardless of chunk count or decay regime."""
    d_out, d_state, scale = _max_diff(T, decay)
    assert d_out < 1e-6, f"T={T} decay={decay}: output diff {d_out:.3e}"
    assert d_state < 1e-5, f"T={T} decay={decay}: state diff {d_state:.3e}"
    # guard against a vacuous pass on all-zero outputs
    assert scale > 1e-3, f"outputs are ~0 ({scale:.3e}); the test proves nothing"


@pytest.mark.parametrize("decay", ["real", "long"])
def test_error_does_not_grow_with_chunk_count(decay):
    """A broken inter-chunk carry shows up as error scaling with chunk count."""
    diffs = [_max_diff(T, decay)[0] for T in (64, 128, 192, 256)]
    assert max(diffs) < 4 * max(min(diffs), 1e-12), (
        f"decay={decay}: error grows with chunk count {diffs} — suspect the carry"
    )


def test_ragged_sequence_length():
    """T not a multiple of chunk: padding must not leak into output or state."""
    for T in (1, 7, 65, 130):
        d_out, d_state, _ = _max_diff(T, "long", seed=T)
        assert d_out < 1e-6, f"T={T}: output diff {d_out:.3e}"
        assert d_state < 1e-5, f"T={T}: state diff {d_state:.3e}"


# ------------------------------------------------------------------- non-vacuity
def test_carry_reaches_across_chunks():
    """Changing the initial state must move the LAST token, 3 chunks downstream."""
    q, k, v, g, beta, s0 = _inputs(192, decay="long", seed=1)
    out_a, _ = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_b, _ = R.chunk_kda(q, k, v, g, beta, (s0 * 2).clone(), chunk=CHUNK)
    moved = (out_a[:, -1] - out_b[:, -1]).abs().max().item()
    agreement = _max_diff(192, "long", seed=1)[0]
    assert moved > 100 * agreement, (
        f"initial state barely reaches the final token ({moved:.3e} vs agreement "
        f"{agreement:.3e}); the carry is not being exercised"
    )


def test_comparison_is_sensitive():
    """A small perturbation must be visible far above the agreement floor."""
    q, k, v, g, beta, s0 = _inputs(192, decay="long", seed=2)
    out, _ = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_p, _ = R.chunk_kda(q, k, v, g, beta, (s0 + 1e-2).clone(), chunk=CHUNK)
    moved = (out - out_p).abs().max().item()
    assert moved > 100 * _max_diff(192, "long", seed=2)[0]


# --------------------------------------------------- KDA is not GDN (per-channel)
def _scalar_gate_chunk(q, k, v, g, beta, state, chunk=CHUNK):
    """GDN's gate: collapse the per-channel decay to one scalar per head.

    Mean over K is the most favourable possible scalar reduction — it preserves
    the average decay rate, so any difference this test sees is genuinely the
    per-channel structure and not a change in overall magnitude.
    """
    g_scalar = g.mean(-1, keepdim=True).expand_as(g)
    return R.chunk_kda(q, k, v, g_scalar, beta, state, chunk=chunk)


def test_gate_is_per_channel_not_scalar():
    """Substituting a per-head scalar decay must change the result materially.

    If this ever fails, the suite can no longer tell KDA from GDN, and a
    gated-DeltaNet kernel could be dropped in where KDA is required.
    """
    q, k, v, g, beta, s0 = _inputs(192, decay="long", seed=3)
    out_kda, s_kda = R.chunk_kda(q, k, v, g, beta, s0.clone(), chunk=CHUNK)
    out_gdn, s_gdn = _scalar_gate_chunk(q, k, v, g, beta, s0.clone())
    rel = (out_kda - out_gdn).abs().max().item() / out_kda.abs().max().item()
    agreement = _max_diff(192, "long", seed=3)[0]
    assert (out_kda - out_gdn).abs().max().item() > 1000 * agreement, (
        "per-head scalar decay is indistinguishable from per-channel decay here; "
        "this test cannot detect a GDN kernel substituted for KDA"
    )
    assert rel > 1e-3, f"relative difference only {rel:.3e}"


def test_forget_gate_is_per_channel_shaped():
    """ForgetGate must emit [B,S,H,K], not [B,S,H] — the structural difference."""
    cfg = R.tiny_cfg()
    fg = R.ForgetGate(cfg)
    g = fg(torch.randn(2, 5, cfg.hidden_size))
    assert g.shape == (2, 5, cfg.linear_num_heads, cfg.linear_head_dim), g.shape
    # the real gate is bounded in (lower, 0)
    assert (g <= 0).all() and (g >= cfg.linear_lower_bound).all()
    # and it must not be constant across the channel axis (that would be GDN)
    assert g.std(-1).min() > 1e-6, "gate is constant per head — that is GDN, not KDA"


# ------------------------------------------------------- full layer, incl. conv
def test_layer_prefill_matches_step_decode():
    """LinearAttention prefill == its own token-by-token decode path.

    Covers the depthwise conv state handoff as well as the recurrence, which the
    function-level tests above do not reach.
    """
    torch.manual_seed(0)
    cfg = R.tiny_cfg()
    la = R.LinearAttention(cfg).eval()
    x = torch.randn(1, 12, cfg.hidden_size)
    with torch.no_grad():
        full, _ = la(x)
        stepped = R.step_decode(la, x)
    d = (full - stepped).abs().max().item()
    assert full.abs().max().item() > 1e-3
    assert d < 1e-5, f"prefill vs decode max|diff| = {d:.3e}"


def test_conv_state_round_trips():
    """forward must RETURN an advanced conv_state; torch.roll does not mutate.

    Regression guard: the upstream reference returned only rec_state, so a caller
    stepping one token at a time silently reused an all-zero conv window.
    """
    torch.manual_seed(0)
    cfg = R.tiny_cfg()
    la = R.LinearAttention(cfg).eval()
    kernel = la.conv1d.weight.shape[-1]
    cs = torch.zeros(1, 3 * la.HK, kernel)
    rs = torch.zeros(1, la.H, la.K, la.K)
    with torch.no_grad():
        _, (cs2, rs2) = la(torch.randn(1, 1, cfg.hidden_size), conv_state=cs, rec_state=rs)
    assert cs2 is not None and cs2.shape == cs.shape
    assert not torch.equal(cs2, cs), "conv_state did not advance"
    assert not torch.equal(rs2, rs), "rec_state did not advance"
    # prefill must also produce a usable handoff window
    with torch.no_grad():
        _, (cs3, _) = la(torch.randn(1, 9, cfg.hidden_size))
    assert cs3 is not None and cs3.shape[-1] == kernel
