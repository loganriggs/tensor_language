"""Diagnostic direction/scale counter-review, with no deployed gain fitting."""
from pathlib import Path
import json
import torch

P = Path(__file__).resolve().parent


def main():
    artifact = torch.load(P/'HEAD17_COMPONENT_MIXED_REUSE_V1_ARTIFACT.pt', weights_only=True)
    cells = []
    for group in range(5):
        z = artifact['readouts'][24*group:24*(group+1), :, 0].double()
        full, old, projected = (z[:, j]-z[:, 0] for j in (1, 4, 5))
        gain = (old*full).sum()/old.square().sum()
        opposite = (old*full) < 0
        mismatch = projected.sign() != old.sign()
        cells.append(dict(
            group=group,
            cosine=float((old*full).sum()/(old.norm()*full.norm())),
            aligned_fraction=float((old*full).sum()/full.square().sum()),
            same_nonzero_sign=int(((old*full)>0).sum()),
            full_zero_count=int((full==0).sum()),
            old_zero_count=int((old==0).sum()),
            opposite_nonzero_sign=int(((old*full)<0).sum()),
            reference_norm_fraction_on_opposite_signs=float(full[opposite].norm()/full.norm()),
            max_absolute_full_effect_on_opposite_signs=float(full[opposite].abs().max()) if opposite.any() else 0.0,
            diagnostic_best_gain=float(gain),
            diagnostic_gain_residual=float((gain*old-full).norm()/full.norm()),
            mean_absolute_old_effect=float(old.abs().mean()),
            max_absolute_projection_error=float((projected-old).abs().max()),
            projection_sign_mismatches=int((projected.sign()!=old.sign()).sum()),
            max_absolute_old_effect_on_projection_sign_mismatches=float(old[mismatch].abs().max()) if mismatch.any() else 0.0,
        ))
    result = dict(cells=cells, scope='Existing diagnostic contexts. Gain fits the scored effect vector and is not an independent prediction or adopted correction. Aligned conditional-effect fraction is not unique causal coverage; separate component/remainder intervention composition remains untested.')
    (P/'HEAD17_COMPONENT_MIXED_REUSE_V1_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
