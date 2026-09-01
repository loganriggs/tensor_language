"""RUNG 381 -- REAL MLP0 INVARIANT OUTPUT-TUCKER SCREEN.

Compute the exact output-mode Gram of the symmetric bilinear contraction under
Euclidean, exhaustive-embedding, and split context input metrics.  Permuting
Down columns relative to product factors is the matched alignment null.

Frozen predictions
------------------
pred_a: context-B top256/top512 invariant energy>=.75/.90.
pred_b: context A/B top512 overlap>=.80; embedding/context-B>=.70.
pred_c: real-null energy gaps at p256/p512>=.05; identities/prices exact.

Null: context-B p512<=.70, split overlap<=.50, or both gaps<=.01.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mlp0_real_invariant_output_tucker_screen_results.json"
CACHE = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FIT_A = (0, 24)
FIT_B = (24, 48)
D = 1152
H = 4608
RANKS = (128, 256, 512, 768)
SEED = 2026090102
OUTPUT_P512_PRICE = D * 512 + 512 * H + D
NATIVE_DOWN_PRICE = D * H + D


def _sqrt(covariance: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(.5 * (covariance + covariance.T))
    floor = float(values[-1]) * 1e-6
    return (vectors * values.clamp_min(floor).sqrt()) @ vectors.T


@torch.no_grad()
def _output_gram(left, right, down, covariance):
    sqrt = _sqrt(covariance.to(left.device).float())
    l = left @ sqrt
    r = right @ sqrt
    ll = l @ l.T
    rr = r @ r.T
    lr = l @ r.T
    product_gram = .5 * (ll * rr + lr * lr.T)
    result = down @ product_gram @ down.T
    return .5 * (result + result.T)


def _spectrum(gram):
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    total = values.sum().clamp_min(1e-30)
    retained = {rank: float(values[:rank].sum() / total) for rank in RANKS}
    return retained, vectors


def _overlap(left, right, rank):
    return float((left[:, :rank].T @ right[:, :rank]).square().sum() / rank)


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert CACHE.exists() and FIT_A[1] == FIT_B[0]
        assert OUTPUT_P512_PRICE == 2_950_272
        assert NATIVE_DOWN_PRICE == 5_309_568
        print("REAL MLP0 OUTPUT TUCKER | dry run: cache, metrics, price, bars valid")
        return

    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    import mlp_all_layer_context_metric_shared_input_screen as M
    from mlp_shared_input_svd_all_layers_screen import _manual_logits
    from tier2_model import load_elriggs

    cached = torch.load(CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    rows_a = cached[FIT_A[0]:FIT_A[1], :257].long().contiguous()
    rows_b = cached[FIT_B[0]:FIT_B[1], :257].long().contiguous()
    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and model.transformer.wte.weight.shape[0] == 50_304
    M.LAYERS = (0,)
    context_a = M._covariances(model, rows_a, _manual_logits)[0]
    context_b = M._covariances(model, rows_b, _manual_logits)[0]
    embedding = F.rms_norm(model.transformer.wte.weight.detach().float(), (D,))
    embedding_covariance = embedding.T @ embedding / embedding.shape[0]
    identity = torch.eye(D, device=embedding.device)

    mlp = model.transformer.h[0].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    metrics = {"euclidean": identity, "full_embedding": embedding_covariance,
               "context_a": context_a, "context_b": context_b}
    spectra, bases = {}, {}
    for name, covariance in metrics.items():
        gram = _output_gram(left, right, down, covariance)
        spectra[name], bases[name] = _spectrum(gram)
        print(name, spectra[name], flush=True)

    generator = torch.Generator(device=down.device).manual_seed(SEED)
    permutation = torch.randperm(H, generator=generator, device=down.device)
    null_gram = _output_gram(left, right, down[:, permutation], context_b)
    null_spectrum, _null_basis = _spectrum(null_gram)
    split_overlap = _overlap(bases["context_a"], bases["context_b"], 512)
    embedding_overlap = _overlap(bases["full_embedding"], bases["context_b"], 512)
    gaps = {rank: spectra["context_b"][rank] - null_spectrum[rank] for rank in RANKS}

    identity_ok = (embedding.shape == (50_304, D) and context_a.shape == context_b.shape == (D, D)
                   and left.shape == right.shape == (H, D) and down.shape == (D, H)
                   and OUTPUT_P512_PRICE == 2_950_272
                   and NATIVE_DOWN_PRICE - OUTPUT_P512_PRICE == 2_359_296)
    pred_a = spectra["context_b"][256] >= .75 and spectra["context_b"][512] >= .90
    pred_b = split_overlap >= .80 and embedding_overlap >= .70
    pred_c = gaps[256] >= .05 and gaps[512] >= .05 and identity_ok
    null = (spectra["context_b"][512] <= .70 or split_overlap <= .50
            or (gaps[256] <= .01 and gaps[512] <= .01))
    result = {
        "status": "mlp0_real_invariant_output_tucker_screen_complete",
        "rung": 381,
        "claim_level": "real_mlp0_invariant_output_mode_metric_spectrum_screen",
        "tensor_definition": "Down columns times symmetric Left/Right input forms",
        "metrics": spectra, "context_b_permuted_down_null": null_spectrum,
        "real_minus_null_energy_gaps": gaps,
        "context_split_top512_overlap": split_overlap,
        "embedding_context_top512_overlap": embedding_overlap,
        "embedding_rows": int(embedding.shape[0]),
        "fit_cache": CACHE.name, "fit_a": list(FIT_A), "fit_b": list(FIT_B),
        "dimensions": {"d": D, "product_width": H},
        "output_p512_factor_price": OUTPUT_P512_PRICE,
        "native_down_bias_price": NATIVE_DOWN_PRICE,
        "saving_scalars_if_output_p512_installed": NATIVE_DOWN_PRICE - OUTPUT_P512_PRICE,
        'pred_a_real_context_output_mode_is_low_rank': bool(pred_a),
        'pred_b_output_subspace_transfers_across_inputs': bool(pred_b),
        'pred_c_low_rank_is_joint_alignment_not_down_only': bool(pred_c),
        "null_real_output_tucker_structure_absent": bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
