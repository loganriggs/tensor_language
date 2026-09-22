"""Loader and attention instrument for GPT-2 small (openai-community/gpt2; 12L x 12H x 768, softmax, learned absolute positions, same BPE
tokenizer as bilin18 so the row caches apply; rows are fed as 513 tokens with labels = inputs, i.e. 512 predictions per row like bilin18).

instrument(model) replaces transformers' eager_attention_forward with an explicit version that exposes the pre-softmax logits to a
per-layer `logit_edit(layer, logits)` and the head outputs to `z_edit(layer, z)` (z: [B, H, T, D] before the output projection).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import transformers.models.gpt2.modeling_gpt2 as G
from transformers import GPT2LMHeadModel

REPO = "gpt2"


def load(repo=REPO, device="cuda"):
    if repo in ("cuda", "cpu"):
        repo, device = REPO, repo
    m = GPT2LMHeadModel.from_pretrained(repo, attn_implementation="eager").float().eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def ce(model, rows, device="cuda", batch=32):
    """Mean CE over the 512 predictions of each 513-token row (labels shifted inside HF)."""
    total = 0.0; n = 0; fw = 0
    with torch.no_grad():
        for s in range(0, rows.shape[0], batch):
            ids = rows[s:s + batch].contiguous().to(device)
            out = model(input_ids=ids, labels=ids); total += float(out.loss) * ids.shape[0] * (ids.shape[1] - 1); n += ids.shape[0] * (ids.shape[1] - 1); fw += 1
    return total / n, fw


def family(model):
    return "gpt2"


def geometry(model):
    c = model.config; H = c.n_head; D = c.n_embd; hd = D // H
    return c.n_layer, H, D, hd, H


def head_qk(model, l, h):
    """q / k weight rows [hd, D] and biases for head h of layer l from the Conv1D c_attn ([D, 3D]: q | k | v), and the logit scale."""
    at = model.transformer.h[l].attn; c = model.config; H = c.n_head; D = c.n_embd; hd = D // H
    W = at.c_attn.weight.detach().float(); b = at.c_attn.bias.detach().float()
    return W[:, h * hd:(h + 1) * hd].T.contiguous(), b[h * hd:(h + 1) * hd], W[:, D + h * hd:D + (h + 1) * hd].T.contiguous(), b[D + h * hd:D + (h + 1) * hd], hd ** -0.5


def instrument(model):
    state = {"logit_edit": None, "z_edit": None, "n": {}, "cos": None, "sin": None}
    original = G.eager_attention_forward
    hooks = [blk.attn.register_forward_pre_hook(lambda m, a, kw, l=l: state["n"].__setitem__(l, a[0] if a else kw["hidden_states"]), with_kwargs=True) for l, blk in enumerate(model.transformer.h)]

    def eager(module, query, key, value, attention_mask, head_mask=None, **kwargs):
        # query/key/value: [B, H, T, D]
        logits = torch.matmul(query, key.transpose(-1, -2))
        if module.scale_attn_weights:
            logits = logits / (value.size(-1) ** 0.5)
        if module.scale_attn_by_inverse_layer_idx:
            logits = logits / float(module.layer_idx + 1)
        if state["logit_edit"] is not None:
            logits = state["logit_edit"](module.layer_idx, logits)
        T = query.size(-2); causal = torch.tril(torch.ones(T, T, device=query.device, dtype=torch.bool))
        logits = logits.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, :T]
        w = torch.softmax(logits.float(), dim=-1).type(value.dtype)
        z = torch.matmul(w, value)                                   # [B, H, T, D]
        if state["z_edit"] is not None:
            z = state["z_edit"](module.layer_idx, z)
        return z.transpose(1, 2), w

    G.eager_attention_forward = eager
    state["_restore"] = lambda: (setattr(G, "eager_attention_forward", original), [h.remove() for h in hooks])
    return state
