"""Convergence of the same frozen 128-product sparse-support conditional fit."""
import hashlib
import json
from pathlib import Path
import time
import torch
from native_reader_msp_generalization_v1 import P, CK
from folded_support_reader_v1 import decode, project, solve
from symmetric_product_als_v1 import native_rhs, normal_operator
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    started=time.perf_counter()
    prior=json.loads((P/'FOLDED_SUPPORT_READER_V1_AUDIT.json').read_text())
    assert all(prior['predictions'].values())
    for item in (prior['source'],prior['cache']):
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    saved=torch.load(prior['source']['path'],weights_only=True,map_location='cpu')
    first=torch.load(prior['cache']['path'],weights_only=True,map_location='cpu')
    sel=first['selected_products']; basis=saved['analysis_basis']; ids=saved['code_indices'].long()
    values=saved['code_values'].clone(); values[sel]=first['left_values']; values[sel+4608]=first['right_values']
    readers=torch.cat([decode(values[i:i+64],basis[ids[i:i+64]]) for i in range(0,len(values),64)])
    a,b=readers[:4608].clone(),readers[4608:].clone(); initial_a=a.clone();initial_b=b.clone()
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram']
    eigen,vectors=torch.linalg.eigh(metric); writers=(eigen.clamp_min(0).sqrt()[:,None]*vectors.T)@d
    chosen=writers[:,sel]; gram=chosen.T@chosen; dw=d[:,sel]
    atoms=[basis[ids[sel]],basis[ids[sel+4608]]]
    total=json.loads((P/'FULLU_TRACE_METRIC_V1_AUDIT.json').read_text())['full_coefficient_metric']['total_energy']
    history=[]; cumulative=0.; maximum_normal=0.; maximum_increase=0.; converged=False
    def rhs_for(partner,variable):
        other=partner[sel]
        return (native_rhs(l,r,d,other,dw,metric)-native_rhs(a,b,d,other,dw,metric)+
                normal_operator(variable[sel],other,gram))
    def residuals():
        out=[]
        for n,(variable,partner) in enumerate(((a,b),(b,a))):
            rhs=rhs_for(partner,variable)
            grad=project(normal_operator(variable[sel],partner[sel],gram)-rhs,atoms[n])
            out.append(float(grad.norm()/project(rhs,atoms[n]).norm().clamp_min(1e-30)))
        return out
    initial_residuals=residuals(); fit_start=time.perf_counter()
    for sweep in range(1,101):
        for n,(variable,partner) in enumerate(((a,b),(b,a))):
            vi=sel+n*4608; other=partner[sel]; rhs=rhs_for(partner,variable)
            fitted,report=solve(atoms[n],other,gram,rhs,values[vi])
            maximum_normal=max(maximum_normal,report['true_relative_residual'])
            change=decode(fitted,atoms[n])-variable[sel]
            loss_change=float((change*normal_operator(change,other,gram)).sum()+
                2*(change*(normal_operator(variable[sel],other,gram)-rhs)).sum())/total
            maximum_increase=max(maximum_increase,loss_change); cumulative+=loss_change
            variable[sel]+=change; values[vi]=fitted
        residual=residuals()
        row=dict(sweep=sweep,relative_error=1-prior['capture_after']+cumulative,
                 projected_residuals=residual,seconds=time.perf_counter()-fit_start)
        history.append(row)
        if len(history)>=6:
            row['five_sweep_relative_change']=abs(history[-6]['relative_error']-row['relative_error'])/row['relative_error']
            converged=max(residual)<=1e-7 and row['five_sweep_relative_change']<=1e-9
        if sweep%5==0 or converged:print(json.dumps(row),flush=True)
        if converged or time.perf_counter()-fit_start>=60:break
    delta=(torch.cat((a[sel],initial_a[sel])),torch.cat((b[sel],initial_b[sel])),torch.cat((chosen,-chosen),1))
    independent=float(inner(delta,delta)+2*inner((initial_a,initial_b,writers),delta)-2*inner((l,r,writers),delta))/total
    replay=abs(independent-cumulative)
    output=dict(initial_projected_residuals=initial_residuals,history=history,converged=converged,
        stop='converged' if converged else 'budget',maximum_normal_residual=maximum_normal,
        maximum_relative_increase=maximum_increase,independent_error_replay=replay,
        capture_before=prior['capture_after'],capture_after=prior['capture_after']-independent,
        additional_gain=-independent,predictions=dict(
            pred_a_instrument=maximum_normal<=1e-9 and maximum_increase<=1e-10 and replay<=1e-9,
            pred_b_convergence=converged,pred_c_gain=-independent>=.0001),
        source=prior['cache'],seconds=time.perf_counter()-started,
        scope='Local convergence only for 128 fixed-support product reader coefficients, all other readers and writers fixed; no joint dictionary convergence or circuit validation.')
    path=Path('/dev/shm/bilin18_folded_support_continue_v1.pt');assert not path.exists()
    torch.save(dict(selected_products=sel,left_values=values[sel],right_values=values[sel+4608],source=prior['cache']),path)
    output['cache']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (P/'FOLDED_SUPPORT_CONTINUE_V1_RESULT.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
