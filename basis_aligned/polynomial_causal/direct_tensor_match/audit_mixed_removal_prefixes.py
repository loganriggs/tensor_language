"""Prefix sensitivity of opened mixed-CP finite-removal results; no new fit."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
    data=json.loads((P/'MIXED_CP_REMOVAL_V1.json').read_text());old=json.loads((P/'ROOT_CASE_ABLATION_V1.json').read_text())
    baseline={(r['domain'],r['alpha'],r['condition']):r for r in old['summary']};rows=[]
    for summary in data['summary']:
        key=tuple(summary[k] for k in ['seed','domain','alpha','condition'])
        rr=[r for r in data['rows'] if tuple(r[k] for k in ['seed','domain','alpha','condition'])==key]
        total_ref=sum(r['reference_energy'] for r in rr);total_error=sum(r['error_energy'] for r in rr)
        reconstructed=math.sqrt(total_error/total_ref);assert abs(reconstructed-summary['error'])<1e-12
        leaves=[math.sqrt((total_error-r['error_energy'])/(total_ref-r['reference_energy'])) for r in rr if total_ref>r['reference_energy']]
        per=[dict(document=r['document'],n=r['n'],reference_energy=r['reference_energy'],relative_error=math.sqrt(r['error_energy']/r['reference_energy'])) for r in rr if r['reference_energy']>0]
        previous=baseline[key[1:]]['error']
        rows.append(dict(seed=key[0],domain=key[1],alpha=key[2],condition=key[3],aggregate_error=reconstructed,previous_error=previous,relative_error_reduction=1-reconstructed/previous,leave_one_prefix_min=min(leaves),leave_one_prefix_max=max(leaves),prefixes_below_10percent=sum(r['relative_error']<=.1 for r in per),prefixes=len(per),per_prefix=per))
    result=dict(rows=rows,all_cells_improve=all(r['aggregate_error']<r['previous_error'] for r in rows),aggregate_cells_below_10percent=sum(r['aggregate_error']<=.1 for r in rows),newline_leave_one_prefix_all_below_10percent=all(r['leave_one_prefix_max']<=.1 for r in rows if r['condition']=='newline'),scope='Post-hoc descriptive prefix sensitivity on same opened panels, not independent confidence intervals, new success gate or untouched confirmation. All sixteen original cells retained; native selectivity remains failed.')
    (P/'MIXED_CP_REMOVAL_PREFIX_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    for r in rows:print({k:v for k,v in r.items() if k!='per_prefix'})
if __name__=='__main__':main()
