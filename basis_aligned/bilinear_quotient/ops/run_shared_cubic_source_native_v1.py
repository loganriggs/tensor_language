#!/usr/bin/env python3
# BQGATE:0bodyforwards;2x4000steps/300sec;1024coefficientprobes/position;800sec.
"""pred_a exactcontrols/nativeFD<=1e-4/evaluation<=2sec;
pred_b bothstationary and >=20%capturegain;
pred_c >=4source matches>=.9, >=4multiheadatoms/arm, cancellation<=2.
Null: this small nonlinear source dictionary does not resolve stable reuse.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from scipy.optimize import linear_sum_assignment
from sparse_path_stability_atlas_v1 import digest
from folded_normalized_router_v1 import rotary
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram,capture
from shared_cubic_source_fit_v1 import fit
from joint_routing_value_polynomial_v1 import contract
STEM='SHARED_CUBIC_SOURCE_NATIVE_V1'
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'SHARED_CUBIC_SOURCE_PROJECTION_V1_CONTROL.json').read_text());fitcontrol=json.loads((P/'SHARED_CUBIC_SOURCE_FIT_V1_CONTROL.json').read_text());assert max(control.values())<=1e-10 and fitcontrol['all_recovered'] and fitcontrol['all_stationary']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('0 body forwards; exact shared cubic variable projection, two bounded starts');return
    out=P/(STEM+'_RESULT.json');artifact=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not artifact.exists();signal.alarm(800);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def w(k):return sd[k].double().cuda()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mix=float(sd['transformer.h.17.attn.lamb']);v=torch.cat([(1-mix)*w('transformer.h.17.attn.c_v.weight'),mix*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1))
    o=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128)*g0[None,:,None];qr=rotary(8,128).cuda();weights={}
    for pos in (7,0):
        rotation=qr.T@rotary(pos,128).cuda();ka=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kb=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1)
        weights[pos]=(q1,ka,q2,kb,v,o)
    gen=torch.Generator(device='cuda').manual_seed(73330);initial=[]
    for r in range(16):
        h=r%9;initial.append(torch.stack([torch.randn(128,device='cuda',dtype=torch.float64,generator=gen)@matrix[h] for matrix in (weights[7][1],weights[7][3],v)]))
    aligned=torch.stack(initial);aligned=aligned/aligned.norm(dim=-1,keepdim=True)
    gen.manual_seed(73331);random=torch.randn(aligned.shape,device='cuda',dtype=torch.float64,generator=gen);random=random/random.norm(dim=-1,keepdim=True);starts=[aligned,random]
    test=aligned.detach().requires_grad_(True);torch.cuda.synchronize();start=time.perf_counter();cap=capture(test,*weights[7]);grad=torch.autograd.grad(cap,test)[0];torch.cuda.synchronize();price=time.perf_counter()-start
    direction=grad/grad.norm();eps=1e-5
    with torch.no_grad():numeric=(capture(aligned+eps*direction,*weights[7])-capture(aligned-eps*direction,*weights[7]))/(2*eps);analytic=(grad*direction).sum();fd=float(abs(numeric-analytic)/analytic.abs().clamp_min(1e-30))
    result={'pred_a':fd<=1e-4 and price<=2,'pred_b':None,'pred_c':None}
    if not result['pred_a']:
        result.update(native_fd_error=fd,evaluation_seconds=price,fits_started=False,source_shas=binding)
        out.write_text(json.dumps(result,indent=2)+'\n');print('native preflight failed');return
    fitted=[];reports=[];histories=[]
    for arm,atoms in enumerate(starts):
        divisor=float(capture(atoms,*weights[7]));held_initial=float(capture(atoms,*weights[0]))
        def callback(row,b,n,mix):
            if row['iteration']%100==0:print(json.dumps(dict(arm=arm,**row)),flush=True)
        best,history,reason=fit(atoms,weights[7],divisor,max_steps=4000,max_seconds=300,callback=callback)
        with torch.no_grad():
            g=atom_gram(best);kernel=query_output_gram(cross_factors(best,*weights[7]));inv=torch.linalg.inv(g)
            energies=(inv[None]@kernel@inv[None]).diagonal(dim1=-2,dim2=-1)*g.diagonal()[None]
            fractions=energies/energies.sum(0,keepdim=True);shared=int(((fractions>=.1).sum(0)>=2).sum());final=float(capture(best,*weights[7]));ratio=float(energies.sum()/final)
            reports.append(dict(arm=arm,initial_capture=divisor,final_capture=final,gain_ratio=final/divisor,held_initial_capture=held_initial,held_final_capture=float(capture(best,*weights[0])),reason=reason,gradient=history[-1]['projected_gradient_norm'],steps=len(history),shared_atoms=shared,cancellation_ratio=ratio,gram_condition=float(torch.linalg.cond(g))))
            fitted.append(best.cpu());histories.append(history)
        print(json.dumps(reports[-1]),flush=True)
    with torch.no_grad():
        combined=torch.cat(fitted).cuda();g=atom_gram(combined);corr=g[:16,16:]/(g.diagonal()[:16,None]*g.diagonal()[None,16:]).sqrt()
        rows,cols=linear_sum_assignment(-corr.abs().cpu().numpy());matched=corr[rows,cols].abs();matches=int((matched>=.9).sum())
        gen.manual_seed(73332);identity=torch.eye(1152,device='cuda',dtype=torch.float64).reshape(1152,9,128);og=torch.einsum('ohk,ohl->hkl',o,o);coverage=[]
        for pos in (7,0):
            energies=[]
            for _ in range(16):
                q=torch.randint(0,2,(64,2,1152),device='cuda',generator=gen).double()*2-1;s=torch.randint(0,2,(64,3,2304),device='cuda',generator=gen).double()*2-1
                raw=contract(q,s,*weights[pos][:5],identity).reshape(64,9,128);energies.append(torch.einsum('nhk,hkl,nhl->n',raw,og,raw).cpu())
            e=torch.cat(energies);mean=float(e.mean());se=float(e.std()/len(e)**.5);coverage.append(dict(position=pos,total_energy_estimate=mean,standard_error=se,capture_fractions=[report['final_capture' if pos==7 else 'held_final_capture']/mean for report in reports]))
    torch.save(dict(atoms=fitted,histories=histories,source_correlations=corr.cpu(),head_reference_scales=g0.cpu()),artifact)
    result.update(pred_b=all(z['reason']=='stationary' and z['gain_ratio']>=1.2 for z in reports),pred_c=matches>=4 and all(z['shared_atoms']>=4 and z['cancellation_ratio']<=2 for z in reports))
    result.update(native_fd_error=fd,evaluation_seconds=price,reports=reports,matched_source_atoms=matches,matched_cosines=matched.tolist(),coverage=coverage,execution_seconds=time.perf_counter()-tic,artifact_sha=digest(artifact),source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
