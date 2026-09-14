"""Post hoc scope audit, preregistered10:54UTC; no fitting or revised old gates."""
import json
import time
import torch
from pathlib import Path

P=Path(__file__).resolve().parent


def main():
    out=P/'FRESH_CUE_SCOPE_V1_RESULT.json';assert not out.exists()
    start=time.monotonic()
    cache=torch.load(P/'MINIMAX_FRESH_CACHE_V1_ARTIFACT.pt',weights_only=True)
    receipt=json.loads((P/'MINIMAX_FRESH_BASELINE_V1_RESULT.json').read_text())
    own=torch.tensor(receipt['reference_own_effects'],dtype=torch.float64)[24:]
    native=cache['measures'][24:,0,0].double()
    rows=[]
    for family in range(4):
        sl=slice(24*family,24*(family+1))
        n=native[sl];o=own[sl]
        contrast=n[::2]-n[1::2];piece=o[::2]-o[1::2]
        rows.append(dict(family=family+1,native_positive_pairs=int((contrast>0).sum()),
                         native_zero_pairs=int((contrast==0).sum()),
                         native_contrasts=contrast.tolist(),mixed_contrasts=piece.tolist(),
                         native_contrast_norm=float(contrast.norm()),mixed_contrast_norm=float(piece.norm()),
                         mixed_over_native_norm=float(piece.norm()/contrast.norm().clamp_min(1e-30)),
                         aligned_mixed_pairs=int((piece*contrast>0).sum())))
    result=dict(families=rows,pred_a=all(r['native_positive_pairs']>=9 for r in rows),
                pred_b=any(r['mixed_over_native_norm']>=.01 for r in rows),
                seconds=time.monotonic()-start,
                scope='Post hoc native capability/contribution scope audit. Mixed term is conditional, not total circuit attribution; no fit or altered earlier compression gates.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
