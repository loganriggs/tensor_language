"""Exact two-reader attention producer fold; CPU only, no task data fitting."""
from pathlib import Path
import json,hashlib,itertools
import torch
from residual_payload_unroll_v1 import coefficients
P=Path(__file__).resolve().parent
CHECKPOINT=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fc4278f4240/pytorch_model.bin')
# The actual snapshot identifier includes the fe before 4278.
CHECKPOINT=CHECKPOINT.parent.parent/'ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'/'pytorch_model.bin'
@torch.no_grad()
def main():
    out=P/'REGIONAL_PRODUCER_OV_FOLD_V1_RESULT.json'; art=P/'REGIONAL_PRODUCER_OV_FOLD_V1_ARTIFACT.pt'
    assert not out.exists() and not art.exists()
    torch.set_num_threads(2);torch.manual_seed(621)
    sd=torch.load(CHECKPOINT,mmap=True,weights_only=True,map_location='cpu')
    C=torch.load(P/'REGIONAL_PAYLOAD_MLP16_FOLD_V1_ARTIFACT.pt',weights_only=True)['current_readers'].double()
    _,scales=coefficients(torch.stack([sd[f'transformer.h.{j}.lambdas'].double() for j in range(18)]))
    x=torch.randn(13,1152,dtype=torch.float64);first=torch.randn_like(x);gamma=torch.randn(13,9,dtype=torch.float64)
    v0=sd['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152)
    programs={};errors={};spans={}
    for layer in (8,9,13):
        prefix=f'transformer.h.{layer}.attn.'
        O=sd[prefix+'c_proj.weight'].double().reshape(1152,9,128)
        V=sd[prefix+'c_v.weight'].double().reshape(9,128,1152);mix=float(sd[prefix+'lamb'])
        F=scales[layer]*torch.einsum('ad,dhk->ahk',C,O)
        current=(1-mix)*torch.einsum('ahk,hkd->ahd',F,V)
        base=mix*torch.einsum('ahk,hkd->ahd',F,v0)
        values=(1-mix)*torch.einsum('nd,hkd->nhk',x,V)+mix*torch.einsum('nd,hkd->nhk',first,v0)
        native=scales[layer]*torch.einsum('nhk,nh,dhk,ad->na',values,gamma,O,C)
        folded=torch.einsum('nh,nd,ahd->na',gamma,x,current)+torch.einsum('nh,nd,ahd->na',gamma,first,base)
        errors[str(layer)]=float((native-folded).norm()/native.norm())
        programs[str(layer)]=dict(projected_output=F,current_value_readers=current,first_value_readers=base,value_mix=mix,residual_scale=float(scales[layer]))
        joined=torch.cat([current,base],-1).reshape(18,2304)
        _,s,vh=torch.linalg.svd(joined,full_matrices=False);rank=int((s>s[0]*1e-10).sum());spans[layer]=vh[:rank]
        programs[str(layer)]['reader_singular_values']=s
    overlap={f'{a}_{b}':torch.linalg.svdvals(spans[a]@spans[b].T).tolist() for a,b in itertools.combinations(spans,2)}
    assert max(errors.values())<1e-12
    torch.save(programs,art)
    result=dict(relative_replay_errors=errors,cross_layer_value_reader_principal_cosines=overlap,stored_tensor_floats=sum(v.numel() for p in programs.values() for v in p.values() if isinstance(v,torch.Tensor)),scope='Exact producer numerators for arbitrary live per-head routing, current/first inputs. Both QK factors, positions, normalizers and upstream state remain required. Reader overlap is descriptive weight geometry, not causal reuse or task independence.',artifact_sha=hashlib.sha256(art.read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
