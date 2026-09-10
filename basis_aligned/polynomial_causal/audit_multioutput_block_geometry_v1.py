"""CPU-only geometry audit of saved weight fit; no activation data."""
from pathlib import Path
import json,torch
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
torch.set_num_threads(2)
state=torch.load(P/'WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_CHECKPOINT.pt',weights_only=False,map_location='cpu')
model=MultioutputQuadraticBlocks(1152);model.load_state_dict(state['best']['model']);writer=state['best']['writer']
e,c=model.components();e=e.detach();c=c.detach()
q,r=torch.linalg.qr(e.transpose(-1,-2),mode='reduced')
cores=r[:,None]@c@r[:,None].transpose(-1,-2)
u=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)['lm_head.weight']
y=torch.cat([chunk.double()@writer for chunk in u.split(2048)])
gw=y.T@y
rows=[];energy=[]
def spectrum(a):
    vals=torch.linalg.eigvalsh((a+a.T)/2).flip(0);scale=float(vals.sum());neg=float(vals.min())
    assert neg>=-1e-10*max(scale,1),neg
    vals=vals.clamp_min(0);prob=vals/vals.sum()
    return dict(top_fraction=float(prob[0]),participation_ratio=float(1/prob.square().sum()),dimensions_for_90_percent=int(torch.searchsorted(prob.cumsum(0),.9))+1),vals
for g in range(16):
    w=gw[g*4:(g+1)*4,g*4:(g+1)*4];h=cores[g]
    gh=h.flatten(1)@h.flatten(1).T;ev,vec=torch.linalg.eigh(w);ws=(vec*ev.clamp_min(0).sqrt())@vec.T
    os,_=spectrum(ws@gh@ws)
    usage=torch.einsum('mn,mij,njk->ik',w,h,h);ins,_=spectrum(usage)
    en=float((w*gh).sum());energy.append(en)
    trace_relative=abs(float(usage.trace())-en)/en
    assert trace_relative<1e-10
    rows.append(dict(block=g,output_function_spectrum=os,input_one_mode_spectrum=ins,energy=en,trace_relative_error=trace_relative,input_bank_condition=float(torch.linalg.cond(e[g]))))
overlap=torch.einsum('gri,hsi->ghrs',e,e)
transport=torch.einsum('ghik,hnkl,ghjl->ghnij',overlap,c,overlap)
gram=torch.einsum('gmij,ghnij->gmhn',c,transport).reshape(64,64)
whole=float((gw*gram).sum())
def overlap_summary(bases):
    fractions=[];largest=[]
    for g in range(16):
        for h in range(g):
            sv=torch.linalg.svdvals(bases[g].T@bases[h]);fractions.append(float(sv.square().sum()/16));largest.append(float(sv[0]))
    return dict(mean_shared_subspace_fraction=sum(fractions)/len(fractions),maximum_shared_subspace_fraction=max(fractions),mean_largest_principal_cosine=sum(largest)/len(largest),maximum_principal_cosine=max(largest))
torch.manual_seed(410);control=MultioutputQuadraticBlocks(1152);ce,_=control.components();cq,_=torch.linalg.qr(ce.detach().transpose(-1,-2),mode='reduced')
hist=state['history'];last=hist[-1];lb=[v for v in hist if v['phase']=='lbfgs'];prior=lb[max(0,len(lb)-101)]
result=dict(scope='Saved unconverged weight function. Orthogonal QR is only a coordinate change within each existing subspace; it imposes no cross-block orthogonality. Random geometry control is an independent CPU initialization, not the exact GPU initial state. No semantic/causal identification.',blocks=rows,learned_overlap=overlap_summary(q),random_geometry_control=overlap_summary(cq),sum_block_energy_over_full_function_energy=sum(energy)/whole,output_top_fraction_mean=sum(x['output_function_spectrum']['top_fraction'] for x in rows)/16,output_top_fraction_min=min(x['output_function_spectrum']['top_fraction'] for x in rows),input_90_percent_dimensions=[x['input_one_mode_spectrum']['dimensions_for_90_percent'] for x in rows],redteam=dict(narrow_failure='First540s blockfit neither converges nor gains1percentagepoint capture over productbaseline',structural_negative=False,still_improving_objective=prior['optimization_loss']-last['optimization_loss'],window_lbfgs_updates=last['lbfgs_done']-prior['lbfgs_done'],final_relative_stationarity=last['relative_stationarity'],remaining_explanations=['Incomplete nonlinear convergence','Only one initialization','Outputrank64 and16dimensional blocks constrain capacity','Penalty is imposed on each quadratic feature, with cross-family granularity differences']))
(P/'MULTIOUTPUT_BLOCK_GEOMETRY_V1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='blocks'}))
