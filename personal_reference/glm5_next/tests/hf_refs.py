# SPDX-License-Identifier: Apache-2.0
"""transformers 5.17 references for the oracle's non-KDA components, vendored.

Provenance: ``transformers/models/glm5_next/modeling_glm5_next.py`` v5.17.0. Each
function below is the corresponding ``forward`` **verbatim**, with only these changes:
parameters are passed as arguments instead of read from ``self``/``config``, and
``ACT2FN[...]`` is spelled out (``config.hidden_act`` is ``"silu"``).

Why vendored rather than imported: the oracle is deliberately transformers-free, and
the installed copy lives in an unrelated venv (the homebrew interpreter has
transformers 5.5, which has no ``glm5_next``). Same discipline as ``indexer_refs.py``
and ``gdn_refs.py``.

**Why this file exists.** External coverage of the oracle existed exactly where
someone had taken the trouble to vendor a reference -- KDA, mHC and the indexer -- and
nowhere else. Two bugs lived in the gap (the silu/sigmoid output gate, and the missing
swiglu clamp). This closes it for the norms, the MLP, the router and sparse-MLA
attention.

**And a vendored reference is not enough on its own.** The swiglu clamp is a no-op
below +-10, so a comparison against ``Glm5NextTextMLP`` on ``tiny_cfg``'s
``normal_(0, 0.02)`` weights passes *with the bug present*. Every test built on this
file must drive its inputs into the regime under test -- past clamps, past saturation,
across dtype boundaries. See ``test_mlp_moe.py``'s measured table.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hf_rms_norm(x, weight, eps):
    """``Glm5NextTextRMSNorm.forward``. Note the cast lands BEFORE the weight multiply."""
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def hf_l2norm(x, dim=-1, eps=1e-6):
    """``l2norm``, with its comment preserved because it is the point:

        # NOTE: FLA compares against `F.normalize` but does + eps instead of
        # max(..., eps) leading to a slight differences
        # main difference to qwen's gdn variation: intentionally use sqrt and /
        # to match original triton
    """
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def hf_mlp(x, gate_w, up_w, down_w, swiglu_limit):
    """``Glm5NextTextMLP.forward``. The clamp is what transformers calls
    "# Key difference using clamping"."""
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    gate = gate.clamp(min=None, max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.linear(F.silu(gate) * up, down_w)


def hf_topk_router(x, weight, e_score_correction_bias, top_k, n_group, topk_group,
                   norm_topk_prob, routed_scaling_factor):
    """``Glm5NextTextTopkRouter.forward``, including the group masking the oracle omits.

    At ``n_group == topk_group == 1`` the masking is provably the identity: one group
    holds every expert, and it is always selected. The oracle relies on that.
    """
    num_experts = weight.shape[0]
    hidden_states = x.view(-1, x.shape[-1])
    router_logits = F.linear(hidden_states.type(torch.float32), weight.type(torch.float32))
    scores = router_logits.sigmoid()
    scores_for_choice = scores + e_score_correction_bias
    group_scores = (
        scores_for_choice.view(-1, n_group, num_experts // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, num_experts // n_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    return router_logits, topk_weights, topk_indices


def hf_attention_softmax(attn_weights, query_dtype):
    """The dtype-relevant half of ``eager_attention_forward``:

        attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                             dtype=torch.float32).to(query.dtype)
    """
    return F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_dtype)
