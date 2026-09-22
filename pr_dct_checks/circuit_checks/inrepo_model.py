"""Load the Elriggs bilinear checkpoints with this repository's own model code (jacclust.tt_model), not AJ's clone.

jacclust/tt_model.py and AJ's tensor_model.py are the same modded-nanogpt lineage: Block(x, v1, x0) with lambdas,
attn(rms_norm(x), v1) -> (y, v1), Bilinear MLP with Left/Right/Down/Down_bias, config.gated / n_embd. The handoff adapter
(circuit_checks.tensorgpt.TensorGPTSpans) therefore works unchanged on these modules; only `state_before_block` (which AJ's
TensorGPT defines and tt_model.GPT does not) is provided here.
"""
from __future__ import annotations

import glob, json, os, sys

import torch
import torch.nn.functional as F

ROOT = "/workspace/tensor_language"
REPOSITORIES = {
    "bilinear": "Elriggs/gpt2-bilinear-18l-9h-1152embd",
    "bilinear-attn": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
    "swiglu": "Elriggs/gpt2-swiglu-18l-9h-1152embd-v2",
    "swiglu-attn": "Elriggs/gpt2-swiglu-sqrd-attn-18l-9h-1152embd",
}


def snapshot_dir(repo_id):
    hf = os.environ.get("HF_HOME", "/workspace/.hf_home")
    cands = glob.glob(os.path.join(hf, "hub", "models--" + repo_id.replace("/", "--"), "snapshots", "*", "pytorch_model.bin"))
    if not cands:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id, "config.json"); cands = [hf_hub_download(repo_id, "pytorch_model.bin")]
    return os.path.dirname(cands[0])


def load(arch="bilinear", device="cuda"):
    """Model in float32 eval mode with grads off, plus its config and the checkpoint step."""
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "basis_aligned/bilinear_quotient/ops"))
    import jacclust.tt_model as TT
    import fastload
    snap = snapshot_dir(REPOSITORIES[arch]); cfg = json.load(open(os.path.join(snap, "config.json"))); step = cfg.pop("step", None)
    with fastload._no_random_init():
        m = TT.GPT(TT.GPTConfig(**cfg)).float().eval()
    sd = torch.load(os.path.join(snap, "pytorch_model.bin"), map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    m.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True, assign=True)
    m = m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, m.config, {"step": step, "snapshot": snap}


def state_before_block(model, token_ids, source_layer):
    """(values, initial_values, first_values) entering block `source_layer`, as AJ's TensorGPT.state_before_block."""
    x = F.rms_norm(model.transformer.wte(token_ids), (model.config.n_embd,)); x0 = x; v1 = None
    for block in model.transformer.h[:source_layer]:
        x, v1 = block(x, v1, x0)
    return {"values": x, "initial_values": x0, "first_values": v1}


def collect_states(model, tokenizer, texts, source_layer, sequence_length, device):
    """AJ's collect_middle_inputs: right-pad/truncate every text to `sequence_length` tokens (pad = eos), run to the source block."""
    vals, inits, firsts = [], [], []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=sequence_length).input_ids.to(device)
        with torch.no_grad():
            st = state_before_block(model, ids, source_layer)
        vals.append(st["values"].float()); inits.append(st["initial_values"].float()); firsts.append(st["first_values"].float())
    return torch.cat(vals), torch.cat(inits), torch.cat(firsts)
