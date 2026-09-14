"""Score-free natural-prefix pairing and charged long-context price forecast."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json


def main():
    p=Path(__file__).resolve().parent
    rows_out=p/'COUPLED_NATURAL_LENGTH_V1_ROWS.json';out=p/'COUPLED_NATURAL_LENGTH_V1_FORECAST.json'
    assert not out.exists() and not rows_out.exists()
    source=p/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json'
    natural=json.loads(source.read_text())['natural'];rows=[]
    for pair,indices in enumerate([(0,1),(16,17)]):
        length=min(len(natural[i]['ids']) for i in indices)
        for i in indices:
            r=natural[i];ids=r['ids'][:length]
            target=r['ids'][length] if length<len(r['ids']) else r.get('newline_id',198)
            rows.append(dict(row_id=len(rows),pair=pair,original_panel_row=i,source_row=r['source_row'],
                             original_length=len(r['ids']),ids=ids,target_id=target,
                             target_origin='next token within original prefix' if length<len(r['ids']) else 'original newline target'))
    assert len(rows)==4 and all(len(rows[i]['ids'])==len(rows[i^1]['ids']) for i in range(4))
    rows_out.write_text(json.dumps(dict(rows=rows,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                       scope='Deterministic existing natural text pairs, common shorter prefix, correct next-token targets. No scores/filtering or corpus independence; donors are different natural texts, not controlled semantic counterfactuals.'),indent=2)+'\n')
    prices=json.loads((p/'COUPLED_ATTENTION10_NATIVE_V2_RESULT.json').read_text())['prices']
    a,b=prices[0],prices[2];dt=b['tokens']-a['tokens'];assert dt==1
    coeff={}
    for key in ['independent_bytes','combined_bytes']:
        per=(b[key]-a[key])//dt;coeff[key]=dict(fixed=a[key]-a['tokens']*per,per_token=per)
    forecasts=[]
    for r in rows:
        values={k:v['fixed']+len(r['ids'])*v['per_token'] for k,v in coeff.items()}
        forecasts.append(dict(row=r['row_id'],tokens=len(r['ids']),**values,ratio=values['combined_bytes']/values['independent_bytes']))
    saving_lengths=[n for n in range(1,1025) if coeff['combined_bytes']['fixed']+n*coeff['combined_bytes']['per_token']<=coeff['independent_bytes']['fixed']+n*coeff['independent_bytes']['per_token']]
    result=dict(utc=datetime.now(timezone.utc).isoformat(),forecasts=forecasts,coefficients=coeff,
                maximum_saving_length_under1025=max(saving_lengths),predicted_price_gate=all(r['ratio']<=1 for r in forecasts),
                body_forwards_registered=204,scope='Exact payload formula extrapolated from native18/19-token tensor layouts including source residual program/x0. No execution, accuracy or latency claim at these new lengths.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
