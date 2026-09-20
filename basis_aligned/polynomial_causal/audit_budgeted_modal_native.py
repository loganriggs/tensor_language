"""Compare native budgeted edit strength, collateral and predictive errors."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    r=json.loads((A/'native_budgeted_modal_v1_result.json').read_text());summary={};failures=[]
    for dataset in ['opened','ood_opened']:
        summary[dataset]={}
        for arm in ['budget0.05','budget0.1']:
            rows=[c for c in r['comparisons'] if c['dataset']==dataset and c['arm']==arm]
            summary[dataset][arm]=dict(cells=len(rows),joint_passes=sum(c['retention']>=.8 and c['modal_ratio']<=.1 for c in rows),strength_failures=sum(c['retention']<.8 for c in rows),collateral_failures=sum(c['modal_ratio']>.1 for c in rows),median_retention=float(np.median([c['retention'] for c in rows])),max_modal_ratio=max(c['modal_ratio'] for c in rows),max_number_prediction_error=max(c['number_error'] for c in rows),max_modal_prediction_error=max(c['modal_error'] for c in rows))
            failures.extend(c for c in rows if c['retention']<.8 or c['modal_ratio']>.1)
    out=dict(predictions=r['predictions'],summary=summary,failures=failures,scope='Both datasets already opened; coefficients chosen from analytic gradients only. Exact-null baseline previously passed joint retention/collateral1/32 and1/16. Partial gains do not satisfy all-cell circuit gate.')
    (P/'BUDGETED_MODAL_NATIVE_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
