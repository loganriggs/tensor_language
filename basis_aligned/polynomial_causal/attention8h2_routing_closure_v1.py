"""Standalone exact two-QK routing for one squared-attention head and source."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def rotary_tables(tokens, dim, device):
    inv = 1 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(tokens, dtype=torch.float32), inv)
    return angles.cos().bfloat16().to(device), angles.sin().bfloat16().to(device)


def rotate(z, cos, sin):
    left, right = z.chunk(2, -1)
    return torch.cat((left * cos + right * sin, -left * sin + right * cos), -1).type_as(z)


def routing(program, current, city_index):
    """Return B,T routing scalars from normalized block8 current state."""
    tokens, dim = current.shape[1], 128
    cos, sin = rotary_tables(tokens, dim, current.device)
    factors = []
    for qname, kname in (("q1_weight", "k1_weight"), ("q2_weight", "k2_weight")):
        q = rotate(F.rms_norm(F.linear(current, program[qname]), (dim,)), cos, sin)
        k = rotate(F.rms_norm(F.linear(current, program[kname]), (dim,)), cos, sin)
        factors.append((q * k[:, city_index:city_index + 1]).sum(-1) / dim)
    result = factors[0] * factors[1]
    result[:, :city_index] = 0
    return result


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parents[1]))
    from jacclust.tt_model import apply_rotary_emb
    out = root / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V2_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(14091842)
    errors = []
    for tokens, city in ((18, 3), (24, 5)):
        current = torch.randn(2, tokens, 1152)
        program = {name: torch.randn(128, 1152) for name in ("q1_weight", "k1_weight", "q2_weight", "k2_weight")}
        candidate = routing(program, current, city)
        cos, sin = rotary_tables(tokens, 128, current.device)
        factors = []
        for qname, kname in (("q1_weight", "k1_weight"), ("q2_weight", "k2_weight")):
            q = F.rms_norm(F.linear(current, program[qname]), (128,))[:, :, None, :]
            k = F.rms_norm(F.linear(current, program[kname]), (128,))[:, :, None, :]
            qr = apply_rotary_emb(q, cos[None, :, None], sin[None, :, None])[:, :, 0]
            kr = apply_rotary_emb(k, cos[None, :, None], sin[None, :, None])[:, :, 0]
            factors.append(torch.einsum("bqd,bkd->bqk", qr, kr) / 128)
        reference = (factors[0] * factors[1]).masked_fill(~torch.ones(tokens, tokens, dtype=torch.bool).tril(), 0)[:, :, city]
        errors.append(float((candidate - reference).abs().max()))
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": max(errors) <= 1e-7,
        "native_fp32_tolerance": 1e-7,
        "max_abs_errors": errors, "state_port_width": 1152,
        "exported_qk_scalars": 4 * 128 * 1152,
        "scope": "Model-free single-source squared-attention routing replay with native BF16-rounded rotary semantics; no native weights or behavior.",
    }
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
