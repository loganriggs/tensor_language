"""Independent cell-level accounting for prospective source response test."""
import json
from pathlib import Path
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    r=json.loads((A/'source_ood_v1_result.json').read_text());records=r['records'];cells={}
    for c in records:cells[(c['panel'],c['role'],c['family'])]=c['native_capability']
    arms={}
    for arm in r['plan']['amplitudes']:
        rows=[c for c in records if c['arm']==arm]
        arms[arm]=dict(number_error=max(c['number_error'] for c in rows),modal_error=max(max(c['modal_error']) for c in rows),number_failures=sum(c['number_error']>.1 for c in rows),modal_failures=sum(max(c['modal_error'])>.05 for c in rows))
    worst=sorted(records,key=lambda r:r['number_error'],reverse=True)[:10]
    selection=r['selection'];out=dict(predictions=r['predictions'],distinct_cells=len(cells),capability_failures=[dict(panel=k[0],role=k[1],family=k[2],accuracy=v) for k,v in cells.items() if v<.9],arms=arms,selectivity_passes=sum(c['aligned_retention']>=.8 and c['modal_ratio']<=.1 for c in selection),max_null_modal_ratio=max(c['modal_ratio'] for c in selection),min_null_retention=min(c['aligned_retention'] for c in selection),worst_prediction_cells=worst,scope='Rows were prospective at preregistration and are now opened. New native derivatives/context generators are still required. Distinct cells, not repeated arm rows, count capability.')
    (P/'SOURCE_OOD_V1_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['arms','worst_prediction_cells']},indent=2));print(json.dumps(arms,indent=2))
if __name__=='__main__':main()
