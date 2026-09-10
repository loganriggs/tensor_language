"""Exact common vocabulary-output summand; preserve it before native tanh."""
from pathlib import Path
import json,torch
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
torch.set_num_threads(2);sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
u=sd['lm_head.weight'].double();mean=u.mean(0)
l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
coef=mean@d;raw=(l.T*coef)@r;q=(raw+raw.T)/2
energy=q.square().sum();ev,vec=torch.linalg.eigh(q);ordered=ev.square().sort(descending=True).values
pos=torch.where(ev>0)[0][-128:].flip(0);neg=torch.where(ev<0)[0][:128]
pa=vec[:,pos]*ev[pos].sqrt();nb=vec[:,neg]*(-ev[neg]).sqrt();aa=pa+nb;bb=pa-nb;paired=(aa@bb.T+bb@aa.T)/2
paired_error=(q-paired).square().sum()/energy;predicted_error=1-(ev[pos].square().sum()+ev[neg].square().sum())/energy
assert abs(float(paired_error-predicted_error))<=1e-10
native_total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
s=torch.load(P/'WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_CHECKPOINT.pt',weights_only=False,map_location='cpu');m=MultioutputQuadraticBlocks(1152);m.load_state_dict(s['best']['model']);e,c=m.components();e=e.detach();c=c.detach();w=s['best']['writer'];cw=(mean@w).reshape(16,4);hc=(cw[:,:,None,None]*c).sum(1)
fit=torch.einsum('gri,grs,gsj->ij',e,hc,e)
# Independent polynomial evaluations check folding and factor reconstruction.
torch.manual_seed(782);x=torch.randn(64,1152,dtype=torch.float64)
via_weights=((x@l.T)*(x@r.T))@coef;via_q=torch.einsum('ni,ij,nj->n',x,q,x)
projection=mean@w;via_fit=((torch.einsum('gri,ni->ngr',e,x)[:,:,:,None]*torch.einsum('gri,ni->ngr',e,x)[:,:,None,:])[:, :,None,:,:]*c[None,:,:,:,:]).sum((-1,-2)).flatten(1)@projection
fit_eval=torch.einsum('ni,ij,nj->n',x,fit,x)
bridge=float((via_weights-via_q).norm()/via_weights.norm());fitbridge=float((via_fit-fit_eval).norm()/fit_eval.norm())
assert max(bridge,fitbridge)<=1e-10
result=dict(schema='common.output.quadratic.v1',native_common_energy_fraction=float(len(u)*energy/native_total),native_centered_energy_fraction=float(1-len(u)*energy/native_total),native_common_best_signed_square_capture={str(k):float(ordered[:k].sum()/energy) for k in [1,8,32,64,128,256,512,1152]},native_common_best_real_product_capture={str(k):float((ev[ev>0].square().sort(descending=True).values[:k].sum()+ev[ev<0].square().sort(descending=True).values[:k].sum())/energy) for k in [1,8,32,64,128,256]},block_fit_centered_relative_squared_error=float((s['best']['diagnostics']['squared_relative_error']-len(u)*(q-fit).square().sum()/native_total)/(1-len(u)*energy/native_total)),native_common_rank90=int(torch.searchsorted(ordered.cumsum(0)/energy,.9))+1,positive_eigenvalues=int((ev>0).sum()),negative_eigenvalues=int((ev<0).sum()),block_fit_common_relative_squared_error=float((q-fit).square().sum()/energy),rank128_product_construction_squared_error=float(paired_error),rank128_product_bound_replay_error=float(abs(paired_error-predicted_error)),native_fold_relative_error=bridge,block_fold_relative_error=fitbridge,scope='Exact all-vocabulary output mean plus centered remainder, no discarded channels and no data. Signed eigendecomposition solves the isolated scalar quadratic exactly, not the remaining token tensor. Uniform pre-tanh shift affects native saturated logits and must be preserved. Rank counts concern one quadratic input matrix, not total circuit rank.')
(P/'COMMON_OUTPUT_QUADRATIC_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
