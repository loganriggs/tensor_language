"""Frozen-order dictionary pruning on fit gradients; held feasibility is diagnostic."""
import json,hashlib
from pathlib import Path
import numpy as np
from shared_selective_source_lp import choose
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/refined_selective_sources_v1_result.json'
a=json.loads(source.read_text());parent=np.array([0.,0.,1.,1.,1.,0.]);reference=parent[a['parent_mapping']]
# Preserve the full-reference target while restricting selectable columns by
# appending a zero-bounded target-coordinate is unnecessary: rescale gradients
# through a reference vector r on selected columns solving n.r=b independently
# for each row. choose uses r only to define that row's target, not as an arm.
def ceiling(g,keep):
    target=float(g[0]@reference);n=g[0,keep]
    if float(n@n)<1e-24:return 0.
    ref=n*target/float(n@n)
    _,check=choose(g[:,keep],ref)
    return check['retention']
records=[]
for role in ['subject','attractor']:
    fit=np.concatenate([np.array(g['gradient']) for g in a['gradients'] if g['role']==role and g['panel']=='opposite'])
    held=np.concatenate([np.array(g['gradient']) for g in a['gradients'] if g['role']==role and g['panel']=='congruent'])
    keep=list(range(23));history=[]
    for remove in range(23):
        candidate=[i for i in keep if i!=remove]
        vals=[ceiling(g,candidate) for g in fit]
        accepted=min(vals)>=.8-1e-9
        history.append(dict(removed=a['source_names'][remove],accepted=bool(accepted),fit_min_retention=min(vals)))
        if accepted:keep=candidate
    fitvals=[ceiling(g,keep) for g in fit];heldvals=[ceiling(g,keep) for g in held]
    records.append(dict(role=role,indices=keep,names=[a['source_names'][i] for i in keep],count=len(keep),fit_min_retention=min(fitvals),held_min_retention=min(heldvals),held_pass_count=sum(v>=.8-1e-9 for v in heldvals),held_count=len(heldvals),history=history))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),removal_order=a['source_names'],records=records,scope='Single fixed-order greedy source dictionary screen. Per-input amplitudes remain oracle-derived. No globally sparsest guarantee, no native validation for the pruned support, no shared-selector success.')
(P/'REFINED_COMMON_SUPPORT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({r['role']:{k:v for k,v in r.items() if k!='history'} for r in records},indent=2))
