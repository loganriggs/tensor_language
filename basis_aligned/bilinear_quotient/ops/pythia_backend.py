"""Loader and attention instrument for GPT-NeoX models (Pythia family): softmax attention, rotary on the first `rotary_ndims` of each head
(25%), parallel attention + MLP, GPT-NeoX tokenizer (row caches .rowcache/pythia_*.pt built from the same text as the GPT-2 caches).

instrument(model) replaces transformers' eager_attention_forward for GPT-NeoX with an explicit version exposing the pre-softmax logits
(scaled) to `logit_edit(layer, logits)` and the head outputs to `z_edit(layer, z)` (z: [B, H, T, D]); it also records the rotary cos/sin
of the current forward (state["cos"], state["sin"]) and each layer's attention input (state["n"][layer]) for low-rank re-computation.
"""
from __future__ import annotations
import torch
import transformers.models.gpt_neox.modeling_gpt_neox as N
from transformers import GPTNeoXForCausalLM


def load(repo, device="cuda", revision=None):
    m = GPTNeoXForCausalLM.from_pretrained(repo, attn_implementation="eager", revision=revision).float().eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def geometry(model):
    c = model.config; L = c.num_hidden_layers; H = c.num_attention_heads; D = c.hidden_size; hd = D // H
    return L, H, D, hd, model.gpt_neox.layers[0].attention.rotary_ndims


def ce(model, rows, device="cuda", batch=32):
    total = 0.0; n = 0; fw = 0
    with torch.no_grad():
        for s in range(0, rows.shape[0], batch):
            ids = rows[s:s + batch].contiguous().to(device)
            out = model(input_ids=ids, labels=ids); total += float(out.loss) * ids.shape[0] * (ids.shape[1] - 1); n += ids.shape[0] * (ids.shape[1] - 1); fw += 1
    return total / n, fw


def head_qk(model, l, h):
    """Native q / k weight rows ([hd, D]) and biases for head h of layer l (query_key_value is interleaved per head: q, k, v blocks of hd)."""
    at = model.gpt_neox.layers[l].attention; hd = at.head_size; W = at.query_key_value.weight.detach().float(); b = at.query_key_value.bias.detach().float()
    base = h * 3 * hd
    return W[base:base + hd], b[base:base + hd], W[base + hd:base + 2 * hd], b[base + hd:base + 2 * hd]


def rotary_qk(q, k, cos, sin, rot):
    """q, k: [B, T, hd]; cos, sin: [B, T, rot]. Rotate the first `rot` dims exactly as GPT-NeoX does."""
    def rot_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]; return torch.cat((-x2, x1), -1)
    qr, qp = q[..., :rot], q[..., rot:]; kr, kp = k[..., :rot], k[..., rot:]
    return torch.cat([qr * cos + rot_half(qr) * sin, qp], -1), torch.cat([kr * cos + rot_half(kr) * sin, kp], -1)


def instrument(model):
    state = {"logit_edit": None, "z_edit": None, "n": {}, "cos": None, "sin": None}
    original = N.eager_attention_forward
    def rot_hook(m, a, o):
        state["cos"], state["sin"] = o[0], o[1]          # return None: a forward hook's return value would replace the output
    hooks = [model.gpt_neox.rotary_emb.register_forward_hook(rot_hook)]
    for l, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.attention.register_forward_pre_hook(lambda m, a, kw, l=l: state["n"].__setitem__(l, a[0] if a else kw["hidden_states"]), with_kwargs=True))

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, head_mask=None, **kwargs):
        logits = torch.matmul(query, key.transpose(2, 3)) * scaling
        if state["logit_edit"] is not None:
            logits = state["logit_edit"](module.layer_idx, logits)
        T = query.size(-2); causal = torch.tril(torch.ones(T, T, device=query.device, dtype=torch.bool))
        logits = logits.masked_fill(~causal, float("-inf"))
        w = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        z = torch.matmul(w, value)
        if state["z_edit"] is not None:
            z = state["z_edit"](module.layer_idx, z)
        return z.transpose(1, 2).contiguous(), w

    N.eager_attention_forward = eager
    state["_restore"] = lambda: (setattr(N, "eager_attention_forward", original), [h.remove() for h in hooks])
    return state
