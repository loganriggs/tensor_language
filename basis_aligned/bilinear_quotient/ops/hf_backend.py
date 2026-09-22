"""Generic loader / attention instrument for HF decoder families used in the replication lane: OPT (learned absolute positions, q pre-scaled,
biases) and Llama-style (SmolLM: full-dim rotary, GQA, no biases). Same contract as gpt2_backend / pythia_backend: instrument(model) patches
the family's eager_attention_forward so that `logit_edit(layer, logits)` sees the scaled pre-softmax logits [B, H, T, T] and `z_edit(layer, z)`
the per-head outputs [B, H, T, D]; state["n"][layer] holds each layer's attention input; state["cos"/"sin"] the rotary tables (Llama).
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM


def load(repo, device="cuda"):
    m = AutoModelForCausalLM.from_pretrained(repo, attn_implementation="eager").float().eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def family(model):
    return model.config.model_type            # "opt" or "llama"


def layers(model):
    return list(model.model.decoder.layers) if family(model) == "opt" else list(model.model.layers)


def attn_of(layer, model):
    return layer.self_attn


def geometry(model):
    c = model.config; H = c.num_attention_heads; D = c.hidden_size; hd = D // H; kv = getattr(c, "num_key_value_heads", None) or H
    return len(layers(model)), H, D, hd, kv


def ce(model, rows, device="cuda", batch=32):
    total = 0.0; n = 0; fw = 0
    with torch.no_grad():
        for s in range(0, rows.shape[0], batch):
            ids = rows[s:s + batch].contiguous().to(device)
            out = model(input_ids=ids, labels=ids); total += float(out.loss) * ids.shape[0] * (ids.shape[1] - 1); n += ids.shape[0] * (ids.shape[1] - 1); fw += 1
    return total / n, fw


def head_qk(model, l, h):
    """q rows [hd, D] (+ bias or None), k rows for the kv head serving query head h (+ bias or None), and the logit scale to apply to q.k."""
    at = attn_of(layers(model)[l], model); c = model.config; H = c.num_attention_heads; D = c.hidden_size; hd = D // H; kv = getattr(c, "num_key_value_heads", None) or H
    g = H // kv; hk = h // g
    Wq = at.q_proj.weight.detach().float()[h * hd:(h + 1) * hd]; bq = at.q_proj.bias.detach().float()[h * hd:(h + 1) * hd] if at.q_proj.bias is not None else None
    Wk = at.k_proj.weight.detach().float()[hk * hd:(hk + 1) * hd]; bk = at.k_proj.bias.detach().float()[hk * hd:(hk + 1) * hd] if at.k_proj.bias is not None else None
    scale = hd ** -0.5        # OPT pre-scales q by this and calls attention with scaling 1.0; Llama passes it as scaling — the product is the same
    if getattr(c, "head_dim", None) is not None:
        hd2 = c.head_dim                        # Qwen3 and friends set head_dim independently of hidden_size / n_head
        if hd2 != hd:
            hd = hd2
            Wq = at.q_proj.weight.detach().float()[h * hd:(h + 1) * hd]; Wk = at.k_proj.weight.detach().float()[hk * hd:(hk + 1) * hd]
            bq = at.q_proj.bias.detach().float()[h * hd:(h + 1) * hd] if at.q_proj.bias is not None else None
            bk = at.k_proj.bias.detach().float()[hk * hd:(hk + 1) * hd] if at.k_proj.bias is not None else None
            scale = hd ** -0.5
    return Wq, bq, Wk, bk, scale


def qk_norm(model, l):
    """Per-head RMSNorm weights applied to q and k before rotary (Qwen3 etc.), or (None, None, eps) if the model has none.
    Qwen3 normalises over the head dimension, exactly as bilin18 does; OLMo-2 normalises the whole projection and is NOT handled here."""
    at = attn_of(layers(model)[l], model)
    qn = getattr(at, "q_norm", None); kn = getattr(at, "k_norm", None)
    if qn is None or kn is None:
        return None, None, None
    hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads
    if qn.weight.shape[0] != hd:
        raise ValueError(f"q_norm is over {tuple(qn.weight.shape)}, not the head dim {hd} — whole-projection QK-norm is not supported")
    return qn.weight.detach().float(), kn.weight.detach().float(), float(getattr(qn, "variance_epsilon", 1e-6))


def apply_qk_norm(x, w, eps):
    """RMSNorm over the last (head) dimension, then the learned per-dimension weight."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotary_qk(q, k, cos, sin):
    """Llama full-dim rotary: q, k [B, T, hd]; cos, sin [B, T, hd]."""
    def rot_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]; return torch.cat((-x2, x1), -1)
    return q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin


def instrument(model):
    fam = family(model)
    if fam == "opt":
        import transformers.models.opt.modeling_opt as M
    else:
        import sys
        M = sys.modules[type(model).__module__]          # the model's OWN module: Qwen3 &co. import eager_attention_forward by name,
        if not hasattr(M, "eager_attention_forward"):    # so patching llama's copy would not be seen
            import transformers.models.llama.modeling_llama as M
    state = {"logit_edit": None, "z_edit": None, "n": {}, "cos": None, "sin": None}
    original = M.eager_attention_forward
    hooks = []
    for l, layer in enumerate(layers(model)):
        hooks.append(attn_of(layer, model).register_forward_pre_hook(lambda m, a, kw, l=l: state["n"].__setitem__(l, a[0] if a else kw["hidden_states"]), with_kwargs=True))
    if fam != "opt":
        def rot_hook(m, a, o):
            state["cos"], state["sin"] = o[0], o[1]
        hooks.append(model.model.rotary_emb.register_forward_hook(rot_hook))
    idx_of = {id(attn_of(layer, model)): l for l, layer in enumerate(layers(model))}

    def eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if fam != "opt":
            rep = getattr(M, "repeat_kv", None)
            if rep is None:
                import transformers.models.llama.modeling_llama as _L; rep = _L.repeat_kv
            key = rep(key, module.num_key_value_groups); value = rep(value, module.num_key_value_groups)
        logits = torch.matmul(query, key.transpose(-1, -2)) * scaling
        layer_idx = idx_of[id(module)]
        if state["logit_edit"] is not None:
            logits = state["logit_edit"](layer_idx, logits)
        T = query.size(-2); causal = torch.tril(torch.ones(T, T, device=query.device, dtype=torch.bool))
        logits = logits.masked_fill(~causal, float("-inf"))
        w = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        z = torch.matmul(w, value)
        if state["z_edit"] is not None:
            z = state["z_edit"](layer_idx, z)
        return z.transpose(1, 2).contiguous(), w

    M.eager_attention_forward = eager
    state["_restore"] = lambda: (setattr(M, "eager_attention_forward", original), [h.remove() for h in hooks])
    return state
