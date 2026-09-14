"""Charge shared native maps once across the entire tested context portfolio."""
from pathlib import Path
from datetime import datetime,timezone
import json


def main():
    p=Path(__file__).resolve().parent;out=p/'COUPLED_SHARED_WEIGHT_PORTFOLIO_V1_RESULT.json'
    assert not out.exists()
    rows=json.loads((p/'DENOMINATOR_FACTORED_NATIVE_V1_RESULT.json').read_text())['prices']
    assert len(rows)==8
    d=1152
    qkv_bank=5*d*d*4
    common=d*d*4+4+2*4+d*8  # O, attention mixture, block10 lambdas, source bias
    source_per=(13*d+9)*8
    lazy_per=source_per+2*d*4  # x0 and first values
    projected_per=source_per+12*5*d*8+12*12*8+9*8+2*d*4
    checks=[]
    for r in rows:
        native=common+qkv_bank+r['tokens']*lazy_per
        projected=common+r['tokens']*projected_per
        checks.append(dict(row=r['row'],native_formula=native,native_recorded=r['independent_bytes'],
                           projected_formula=projected,projected_recorded=r['combined_bytes'],
                           missing_reentry_bytes=projected-r['combined_bytes']))
    plans=[]
    for mask in range(1<<len(rows)):
        need_bank=mask!=(1<<len(rows))-1
        total=common+(qkv_bank if need_bank else 0)
        total+=sum(r['tokens']*(projected_per if mask&(1<<i) else lazy_per) for i,r in enumerate(rows))
        plans.append(dict(projected_mask=mask,payload_bytes=total,qkv_bank_retained=need_bank))
    baseline=plans[0]['payload_bytes'];fully_projected=plans[-1]['payload_bytes']
    best=min(plans,key=lambda r:r['payload_bytes'])
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=all(r['native_formula']==r['native_recorded'] and r['missing_reentry_bytes']==8 for r in checks),
                pred_b=fully_projected<=.99*baseline,
                contexts=len(rows),total_tokens=sum(r['tokens'] for r in rows),
                shared_native_portfolio_bytes=baseline,shared_projected_portfolio_bytes=fully_projected,
                projected_to_native=fully_projected/baseline,best_storage_plan=best,
                shared_common_bytes=common,shared_qkv_bank_bytes=qkv_bank,lazy_bytes_per_token=lazy_per,projected_bytes_per_token=projected_per,
                per_context_checks=checks,assignments_checked=len(plans),
                scope='Exact retained-payload audit for eight existing fixed-context executors. Shared model maps/source bias/reentry counted once. Eight contexts are not eight circuits. Runtime/working-memory/fidelity not retested. Original prefix/model generators remain globally required; standalone single-context benefits do not establish shared-service weight compression.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_context_checks'}))


if __name__=='__main__':main()
