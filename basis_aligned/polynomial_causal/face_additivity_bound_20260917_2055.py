"""Read-only opened-artifact bound; no native model, fitting, or package changes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import torch

P = Path(__file__).resolve().parent
STEM = 'ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1'

def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    artifact = P / (STEM + '_ARTIFACT.pt')
    manifest = P / (STEM + '_ROWS.json')
    data = torch.load(artifact, map_location='cpu', weights_only=True)
    rows = json.loads(manifest.read_text())['rows']
    assert data['values'].shape == (8, 144, 5)
    assert list(data['selected_masks']) == [1, 4, 5]
    # Four vector-valued outputs ordered 00,10,01,11; current-value port fixed.
    y = data['values'][[0, 1, 4, 5], 48:, 0].double()
    signs = torch.tensor([1., -1., -1., 1.], dtype=torch.float64)
    interaction = signs @ y
    residual = signs[:, None] * interaction[None, :] / 4
    additive = y - residual
    # Orthogonal projection onto constants and the two single-port effects.
    design = torch.tensor([[1.,0.,0.],[1.,1.,0.],[1.,0.,1.],[1.,1.,1.]], dtype=torch.float64)
    closure = float((signs @ additive).abs().max())
    orthogonality = float((design.T @ residual).abs().max())
    groups = {'all': list(range(len(rows)))}
    groups.update({c: [i for i,r in enumerate(rows) if r['cue']==c] for c in ('British','American')})
    groups.update({f'context_{c}': [i for i,r in enumerate(rows) if r['context_id']==c]
                   for c in sorted({r['context_id'] for r in rows})})
    records = {}
    for name, ids in groups.items():
        a, b = y[1, ids]-y[0, ids], y[2, ids]-y[0, ids]
        denominator = min(float(a.norm()), float(b.norm()))
        assert denominator > 1e-12
        bound = float(interaction[ids].norm()) / 4
        records[name] = {
            'rows': len(ids), 'distinct_contexts': len({rows[i]['context_id'] for i in ids}),
            'distinct_source_cells': len({(rows[i]['source_cache'],rows[i]['source_row']) for i in ids}),
            'smallest_single_l2_logits': denominator,
            'sharp_worst_corner_l2_lower_bound_logits': bound,
            'bound_over_smallest_single': bound/denominator,
            'min_four_corner_frobenius_error_logits': 2*bound,
            'attained_max_corner_l2_logits': float(residual[:,ids].norm(dim=1).max()),
            'necessary_condition_for_all_four_errors_le_035_single': bound/denominator <= .35,
        }
    # Exhaust all anchor flips. Absolute interaction is invariant, singles are not.
    anchors = {}
    for anchor in range(4):
        perm = [anchor ^ k for k in range(4)]
        z = y[perm]
        mixed = signs @ z
        anchors[str(anchor)] = {
            'interaction_l2_logits': float(mixed.norm()),
            'interaction_over_smallest_single': float(mixed.norm()/torch.minimum((z[1]-z[0]).norm(),(z[2]-z[0]).norm())),
        }
    # Nonidentifiability witness: identical full input/output functions, distinct
    # placement of interaction. Fifth same-site additive-write arm distinguishes.
    points = [(a,b) for a in (0.,.25,1.) for b in (0.,.5,1.)]
    kappa = 2.
    fa = lambda z: z[0]+z[1]+kappa*z[2]
    fb = lambda z: z[0]+z[1]+kappa*z[0]*z[1]
    toy_error = max(abs(fa((a,b,a*b))-fb((a,b,0.))) for a,b in points)
    toy = {'function_agreement_maxabs': toy_error,
           'same_four_corners': [0.,1.,1.,4.],
           'producer_interaction_model_additive_write_score': fa((1.,1.,0.)),
           'suffix_interaction_model_additive_write_score': fb((1.,1.,0.)),
           'scope': 'Polynomial counterexample, not fitted to native weights or native evidence.'}
    result = {
        'utc': datetime.now(timezone.utc).isoformat(), 'panel_status':'opened',
        'evidence':'edit-artifact reanalysis and synthetic algebra; no new edits',
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'control_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows': len(rows), 'distinct_contexts': len({r['context_id'] for r in rows}),
        'distinct_token_sequences': len({tuple(r['ids']) for r in rows}),
        'additive_closure_maxabs': closure, 'residual_orthogonality_maxabs': orthogonality,
        'groups': records, 'anchor_sensitivity': anchors, 'identifiability_toy': toy,
        'validation_pass': closure < 1e-12 and orthogonality < 1e-12 and toy_error == 0
            and all(abs(r['attained_max_corner_l2_logits']-r['sharp_worst_corner_l2_lower_bound_logits']) < 1e-12 for r in records.values()),
        'seconds': time.perf_counter()-start,
        'scope':'Sharp additive approximation obstruction on four existing output corners. Does not identify interaction source, native causal fidelity, or matched-null specificity. Original gates unchanged.'}
    assert result['validation_pass']
    output = P / 'FACE_ADDITIVITY_BOUND_20260917_2055_RESULT.json'
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
