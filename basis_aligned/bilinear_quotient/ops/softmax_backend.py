"""Loader and attention instrument for the SOFTMAX sibling of bilin18: Elriggs/gpt2-bilinear-18l-9h-1152embd (same 18L / 9H / 1152 bilinear
MLPs, rotary, QK-norm, lambda-mixed values; softmax attention via F.scaled_dot_product_attention instead of the unnormalised squared
pattern). Same tokenizer and vocabulary, so the bilin18 row caches apply unchanged.

instrument(model) replaces F.scaled_dot_product_attention inside jacclust.tt_model with a function that computes the causal softmax
explicitly from the pre-softmax logits and lets a per-layer `edit` rewrite the logits (kernel programs, content terms) or the head outputs
(mean ablation). state["layer"] is set by a pre-hook on each block's attention.
"""
from __future__ import annotations
import glob, json, sys, os
import torch
import torch.nn.functional as F

REPO = "Elriggs/gpt2-bilinear-18l-9h-1152embd"
NATIVE_REF = None   # registered by the first rung (v750)


def snapshot_dir():
    hf = os.environ.get("HF_HOME", "/workspace/.hf_home")
    cands = glob.glob(os.path.join(hf, "hub", "models--Elriggs--gpt2-bilinear-18l-9h-1152embd", "snapshots", "*", "pytorch_model.bin"))
    if not cands:
        raise FileNotFoundError("softmax checkpoint not downloaded")
    return os.path.dirname(cands[0])


def load(device="cuda"):
    sys.path.insert(0, "/workspace/tensor_language")
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/bilinear_quotient/ops")
    import jacclust.tt_model as TT
    import fastload
    snap = snapshot_dir(); cfg = json.load(open(os.path.join(snap, "config.json"))); cfg.pop("step", None)
    with fastload._no_random_init():
        m = TT.GPT(TT.GPTConfig(**cfg)).float().eval()
    sd = torch.load(os.path.join(snap, "pytorch_model.bin"), map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True, assign=True)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    assert not m.transformer.h[0].attn.squared_attn
    return m, TT


def instrument(model, TT):
    """Returns state; state['edit'] is None or a callable edit(layer, logits[B,H,T,T] (causal-masked with -inf), z[B,H,T,D] after softmax) -> (logits, z_override or None).
    Simpler contract used by the rungs: state['logit_edit'](layer, logits) -> logits and state['z_edit'](layer, z) -> z, either may be None."""
    state = {"layer": None, "logit_edit": None, "z_edit": None}
    hooks = [blk.attn.register_forward_pre_hook(lambda m, a, l=l: state.__setitem__("layer", l)) for l, blk in enumerate(model.transformer.h)]

    def sdpa(q, k, v, is_causal=True, **kw):
        # q, k, v: [B, H, T, D] (already transposed by the model); scale 1/sqrt(D) as in F.scaled_dot_product_attention
        B, H, T, D = q.shape
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / (D ** 0.5)
        if state["logit_edit"] is not None:
            logits = state["logit_edit"](state["layer"], logits)
        causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        logits = logits.masked_fill(~causal, float("-inf"))
        pat = torch.softmax(logits.float(), dim=-1).to(q.dtype)
        z = torch.einsum("bhqk,bhkd->bhqd", pat, v)
        if state["z_edit"] is not None:
            z = state["z_edit"](state["layer"], z)
        return z

    TT.F.scaled_dot_product_attention = sdpa
    state["_restore"] = lambda: (setattr(TT.F, "scaled_dot_product_attention", F.scaled_dot_product_attention), [h.remove() for h in hooks])
    return state
