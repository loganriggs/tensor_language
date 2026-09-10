"""Paired uncertainty for stored factor dependencies; no new causal fitting."""
import hashlib,json
from pathlib import Path
import numpy as np
from paired_panel_bootstrap_v1 import PairedPanelBootstrap


def main():
    p=Path(__file__).resolve().parent;source=p/'CORRELATIVE_THREE_FACTOR_V1_RESULT.json'
    result=json.loads(source.read_text());reports={}
    for j,(name,row) in enumerate(result['reports'].items()):
        bs=PairedPanelBootstrap(16,9111350+j);branches={}
        for branch,r in row['branches'].items():
            dep=r['conditional_dependencies'];q1=np.asarray(dep['qk1']['per_row']);q2=np.asarray(dep['qk2']['per_row'])
            dependencies={key:{'mean':v['mean'],'ci95':bs.mean(v['per_row'])} for key,v in dep.items()}
            interaction={key:{'relative_l2':v['relative_l2'],
                              'ci95':bs.relative_l2(v['squared_per_row'],r['full_effect_squared_per_row'])}
                         for key,v in r['interactions'].items()}
            branches[branch]={'dependencies':dependencies,'score_half_difference':{'mean':float((q1-q2).mean()),'ci95':bs.mean(q1-q2)},
                              'interactions':interaction,'higher_order_ci95':bs.relative_l2(r['higher_order_squared_per_row'],r['full_effect_squared_per_row'])}
        reports[name]=branches
    out={'schema':'correlative.three_factor.audit.v1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reports':reports,
         'scope':'4000 paired authored-group resamples per panel, shared across factors/branches. No canonical QK labels, no fresh-data claim or threshold change; endpoint interaction norms are not additive causal shares.'}
    with (p/'CORRELATIVE_THREE_FACTOR_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({n:{k:v for k,v in b['R' if n=='C' else 'P'].items() if k!='interactions'} for n,b in reports.items()},indent=2))


if __name__=='__main__':main()
