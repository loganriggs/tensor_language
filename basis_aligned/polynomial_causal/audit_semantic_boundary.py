"""Output-budget audit of the ordered boundary decomposition, without new fitting."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    r=json.loads((A/'semantic_pair_boundary_v1_result.json').read_text());out=[];identity=[]
    for panel,roles in r['data'].items():
        for role,families in roles.items():
            for family,arms in families.items():
                cs=[c for c in r['cells'] if c['panel']==panel and c['role']==role and c['family']==family]
                y=np.array(arms['B']['actual']);den=max(np.linalg.norm(y),1e-30)
                primitive=sum(np.array(arms[k]['actual']) for k in ['middle_writes_4_7','mlp_8','mlp_10'])
                boundary=sum(np.array(c['boundary_induced']) for c in cs);suffix=sum(np.array(c['suffix_only']) for c in cs);total=sum(np.array(c['total']) for c in cs)
                identity.append(float(np.max(np.abs(total-boundary-suffix))))
                metrics={name:float(np.linalg.norm(pred-y)/den) for name,pred in [('additive',primitive),('all_pair',primitive+total),('boundary_only',primitive+boundary),('suffix_only',primitive+suffix)]}
                out.append(dict(panel=panel,role=role,family=family,relative_errors=metrics,B_effect_norm=float(den),boundary_sum_over_B=float(np.linalg.norm(boundary)/den),suffix_sum_over_B=float(np.linalg.norm(suffix)/den),boundary_suffix_cosine=float(boundary@suffix/max(np.linalg.norm(boundary)*np.linalg.norm(suffix),1e-30))))
    assert max(identity)<1e-12
    result=dict(max_vector_identity_error=max(identity),cells=out,max_errors={k:max(c['relative_errors'][k] for c in out) for k in out[0]['relative_errors']},scope='Posthoc output-budget audit; no replacement of failed registered interaction or native capability gates.')
    (P/'SEMANTIC_BOUNDARY_CPU_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result['max_errors'])
if __name__=='__main__':main()
