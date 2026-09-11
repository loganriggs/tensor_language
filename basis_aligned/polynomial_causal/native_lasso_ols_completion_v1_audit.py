"""Frozen oblique features: OLS fills low-penalty Lasso's missing support slots."""
import hashlib,json,math,time
from pathlib import Path
import torch
from native_reader_msp_generalization_v1 import P,CK
from lasso_reader_encoding_v1 import encode as legacy
from lasso_ols_completion_v1 import encode as completed

@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    assert all(json.loads((P/'LASSO_OLS_COMPLETION_V1_CONTROL.json').read_text())['predictions'].values())
    previous=json.loads((P/'NATIVE_OBLIQUE_ENCODER_V1_AUDIT.json').read_text());source=previous['source']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu');basis=saved['analysis_basis']
    selected=torch.tensor(previous['selected_products'][:32]);ids=torch.cat((selected,selected+4608))
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    readers=torch.cat([sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right')])[ids]
    rows=[]
    for name,encoder,penalty in [('legacy',legacy,.05),('scaled',legacy,.05*math.sqrt(12/1152)),('ols_completion',completed,.05)]:
        started=time.perf_counter();support,values,report=encoder(basis,readers,k=128,penalty=penalty)
        prediction=torch.einsum('nk,nkd->nd',values,basis[support])
        capture=1-((prediction-readers)/readers.norm(dim=1,keepdim=True)).square().sum(1)
        row=dict(name=name,penalty=penalty,mean_capture=float(capture.mean()),per_reader_capture=capture.tolist(),
                 encoding=report,seconds=time.perf_counter()-started)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('per_reader_capture','encoding')}),flush=True)
    gains=torch.tensor(rows[2]['per_reader_capture'])-torch.tensor(rows[0]['per_reader_capture'])
    result=dict(predictions=dict(pred_a_instrument=all(r['encoding']['converged'] and
        r['encoding']['maximum_code_kkt']<=1e-5 and r['encoding']['support_ls_normal_residual']<=1e-8 for r in rows),
        pred_b_legacy_gain=rows[2]['mean_capture']>=rows[0]['mean_capture']+.01,
        pred_c_scaled_gain=rows[2]['mean_capture']>=rows[1]['mean_capture']+.005),
        fraction_improved_over_legacy=float((gains>0).double().mean()),minimum_gain=float(gains.min()),
        arms=rows,selected_products=selected.tolist(),source=source,
        scope='64 already-inspected weight readers, fixed features,128-term price, no full-U or corpus score, no changed live L1 source.')
    with (P/'NATIVE_LASSO_OLS_COMPLETION_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],fraction_improved=result['fraction_improved_over_legacy'])),flush=True)

if __name__=='__main__':main()
