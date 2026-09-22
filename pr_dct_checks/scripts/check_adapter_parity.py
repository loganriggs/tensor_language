"""Check the TensorGPT adapter reproduces AJ's forward pass, then smoke-test E1.

Usage:
  python scripts/check_adapter_parity.py --repo /path/to/redesigned-octo-couscous            # tiny random model, CPU
  python scripts/check_adapter_parity.py --repo ... --hf Elriggs/gpt2-bilinear-18l-9h-1152embd --device cuda
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to AJ's repo (for tensor_model.py)")
    ap.add_argument("--hf", default=None, help="HF repo id; omit for a tiny random model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", type=int, default=1)
    ap.add_argument("--target", type=int, default=3)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import tensor_model as tm
    from circuit_checks.tensorgpt import TensorGPTSpans
    from circuit_checks.checks import (pathway_completeness, top_units, hidden_response,
                                       all_units, output_score)
    from circuit_checks.core import participation_ratio

    torch.manual_seed(0)
    if a.hf:
        model, cfg, _ = tm.load_tensor_gpt(a.hf, a.device)
    else:
        cfg = tm.TensorGPTConfig(vocab_size=100, n_layer=4, n_head=2, n_embd=32, bilinear=True)
        model = tm.TensorGPT(cfg)
        for p in model.parameters():            # undo zero-inits so everything is nontrivial
            if p.dim() == 2:
                torch.nn.init.normal_(p, std=0.2)
        model = model.to(a.device).eval()
    model.requires_grad_(False)

    ids = torch.randint(0, cfg.vocab_size, (1, 12), device=a.device)
    st = model.state_before_block(ids, a.source)
    ctx = (st["values"].float(), st["initial_values"].float(), st["first_values"].float())
    spans = TensorGPTSpans(model, a.source, a.target)
    span = spans.make_span(ctx)

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        ref = tm.TensorMiddleSpan(model, a.source, a.target)(*ctx)[:, -3:].mean(1)[0]
        ours = span(torch.zeros(cfg.n_embd, device=a.device))[0]
    err = (ref - ours).abs().max().item()
    print(f"forward parity max abs err: {err:.2e}")
    assert err < 1e-4 * max(1.0, ref.abs().max().item()), "adapter does not match TensorMiddleSpan"

    d = cfg.n_embd
    l, r, u = (torch.nn.functional.normalize(torch.randn(d, device=a.device), dim=0) for _ in range(3))
    with sdpa_kernel(SDPBackend.MATH):
        resp = hidden_response(span, l, r, spans.aj_pr_layers)
        pr = participation_ratio(torch.cat([resp[k] for k in spans.aj_pr_layers])).item()
        widths = {L: spans.mlp_width(L) for L in spans.all_layers}
        res = pathway_completeness(spans.make_span, [ctx], u, l, r, {
            "top1_aj_range": lambda sp: top_units(hidden_response(sp, l, r, spans.aj_pr_layers), 1),
            "all_mlps": lambda sp: all_units(widths, spans.all_layers),
            "source_block_mlp": lambda sp: all_units(widths, [a.source]),
        })
        from circuit_checks.core import FreezeSpec
        both = FreezeSpec(mlp_masks=all_units(widths, spans.all_layers).mlp_masks,
                          attn_layers=set(spans.all_layers))
        leftover = output_score(span, u, l, r, both).item()
    print(f"random factor: PR={pr:.1f}  " + "  ".join(
        f"{k}={v['mean']:.3f}" for k, v in res.items() if not k.startswith("_")))
    print(f"score with all MLPs+attention frozen (should be ~0; residual passthrough is linear): {leftover:.2e}")


if __name__ == "__main__":
    main()
