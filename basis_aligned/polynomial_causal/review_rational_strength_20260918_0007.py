"""Bounded opened-fixture CPU control; no suffix, fitting, or primary writes.

Prior art: RESPONSE_PRODUCT_BASIS_V2_MATH.md. Test its rational response
restriction on the NEW donor-free head8.2 source and measure approximation
error at signed strengths. Gates fixed before execution: exact identity and
two-node reconstruction relative error <1e-9, direct zero/outside exactly zero.
Strength error curves are diagnostics, with no posthoc scientific pass gate.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
import torch

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P / 'extracted_circuits/typed_face_single_head_norm_v1'))
import city_inherited_removal_v1 as removal
from regional_endpoint_batching_v1 import group_rows

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    start = time.perf_counter()
    paths = [P/'SINGLE_HEAD_FRESH_V1_ROWS.json', P/'SINGLE_HEAD_FRESH_V1_ARTIFACT.pt',
             P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',
             P/'city_inherited_removal_v1.py', Path(__file__)]
    hashes = {str(p.relative_to(P)): digest(p) for p in paths}
    rows = json.loads(paths[0].read_text())['rows']
    groups, _ = group_rows(rows)
    fixtures = torch.load(paths[1], map_location='cpu', weights_only=True)['fixtures']
    p = torch.load(paths[2], map_location='cpu', weights_only=True)
    L, R, D = (p['mlp8'][k].double() for k in ('left', 'right', 'down'))
    eps = torch.finfo(torch.float32).eps
    strengths = [-.5, .125, .25, .5, .75, 1., 1.5]
    maxima = {'identity_relative': 0., 'interpolation_relative': 0., 'zero': 0., 'outside': 0.}
    stats = {}
    cells = []
    def rel(x,y):
        return float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-30))
    for row, f in zip(groups, fixtures):
        x = {k:f['candidate_inputs'][k] for k in ('residual7','token_ids','city','destination')}
        w, g2 = removal.source(p, **x, strength=1.)
        w, g2, g = w.double(), g2.double(), f['inputs']['post_attention8'].double()
        s0 = g.square().mean(-1,keepdim=True)+eps
        s2 = g2.square().mean(-1,keepdim=True)+eps
        beta = 2*(g*w).mean(-1,keepdim=True)
        gamma = w.square().mean(-1,keepdim=True)
        lg,rg,lw,rw,lg2,rg2 = g@L.T,g@R.T,w@L.T,w@R.T,g2@L.T,g2@R.T
        B0 = (lg*rg)@D.T
        B1 = (lw*rg+lg*rw)@D.T
        B2 = (lw*rw)@D.T
        C1 = (lw*rg2+lg2*rw)@D.T
        pv, qv = B1-beta*B0/s0, B2-gamma*B0/s0
        def denom(t):
            return (g+t*w).square().mean(-1,keepdim=True)+eps
        def direct(t):
            return t*w+(((g+t*w)@L.T)*((g+t*w)@R.T)/denom(t)-lg*rg/s0)@D.T
        # Known skip and denominator reduce interpolation to degree ONE.
        a,b = .25,.75
        za,zb = (direct(a)-a*w)*denom(a)/a,(direct(b)-b*w)*denom(b)/b
        maxima['zero'] = max(maxima['zero'],float(direct(0).abs().max()))
        entry = {'variant':row['variant'], 'token_ids':x['token_ids'].tolist(), 'strengths':{}}
        for t in strengths:
            actual = direct(t)
            exact = t*w+(t*pv+t*t*qv)/denom(t)
            interp = t*w+t*((b-t)*za+(t-a)*zb)/(b-a)/denom(t)
            approx = t*w+(t*C1+t*t*B2)/s2
            # Exact error decomposition; two separately named omissions.
            full_frozen = t*w+(t*B1+t*t*B2)/s0
            normalization = full_frozen-actual
            background = approx-full_frozen
            maxima['identity_relative'] = max(maxima['identity_relative'],rel(exact,actual))
            maxima['interpolation_relative'] = max(maxima['interpolation_relative'],rel(interp,actual))
            maxima['outside'] = max(maxima['outside'],float(actual[:,~x['destination']].abs().max()))
            assert torch.allclose(approx-actual,normalization+background,atol=1e-12,rtol=1e-10)
            v = stats.setdefault((row['variant'],t),[0.]*7)
            values = [(approx-actual).square().sum(),actual.square().sum(),approx.square().sum(),
                      (approx*actual).sum(),normalization.square().sum(),background.square().sum(),
                      (normalization*background).sum()]
            for j,val in enumerate(values): v[j] += float(val)
            entry['strengths'][str(t)] = {'relative_error':rel(approx,actual),
                'min_denominator_ratio':float((denom(t)/s0).min())}
        cells.append(entry)
    # Count actual text contexts excluding cue side, independently of endpoints.
    context_keys = sorted({(str(r.get('variant')),str(r.get('context_id'))) for r in groups})
    summaries = {}
    for (family,t),v in stats.items():
        summaries.setdefault(family,{})[str(t)] = {
            'relative_error':(v[0]/v[1])**.5,'norm_ratio':(v[2]/v[1])**.5,
            'aligned_fraction':v[3]/v[1], 'normalization_error_ratio':(v[4]/v[1])**.5,
            'background_error_ratio':(v[5]/v[1])**.5,
            'error_term_inner_product_over_reference_squared':v[6]/v[1]}
    assert len(fixtures)==len(groups)==40
    assert maxima['identity_relative']<1e-9 and maxima['interpolation_relative']<1e-9
    assert maxima['zero']==0 and maxima['outside']==0
    assert hashes=={str(p.relative_to(P)):digest(p) for p in paths}
    result = {'utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,
        'gates_pass':True,'maxima':maxima,'fixtures':len(fixtures),'endpoint_rows':len(rows),
        'group_row_keys':list(groups[0]),'observed_variant_context_id_keys':context_keys,
        'families':summaries,'per_sequence':cells,'input_hashes':hashes,
        'scope':'Opened native-state FP64 local response only. No native suffix execution; strength interpolation is algebraic replay, not OOD or composition evidence.'}
    out=P/'REVIEW_RATIONAL_STRENGTH_20260918_0007_RESULT.json'
    with out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_sequence','input_hashes','observed_variant_context_id_keys')},indent=2))

if __name__=='__main__': main()
