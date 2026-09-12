"""CPU suffix replay and joint candidate from already scanned producer ports."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
    torch.set_num_threads(2);rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_V2_ROWS.json').read_text())['rows'];data=torch.load(P/'REGIONAL_PAYLOAD_ATLAS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');b=json.loads((P/'REGIONAL_PAYLOAD_ATLAS_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in b if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    l,r,d=[sd['transformer.h.17.mlp.'+n+'.weight'].float() for n in ('Left','Right','Down')];bias=sd['transformer.h.17.mlp.Down_bias'].float();ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids']] for r in rows]);u=sd['lm_head.weight'][ids].float();pre=data['pre'];writes=data['write_vertices'];group=[10,11,15];delta=(writes[:,group]-writes[:,0,None]).sum(1)
    def margins(change):
        z=pre+change.float();x=F.rms_norm(z,(1152,));h=z+((x@l.T)*(x@r.T))@d.T+bias;logits=30*torch.tanh(torch.einsum('nd,nkd->nk',F.rms_norm(h,(1152,)),u)/30);return torch.stack([logits[:,0]-logits[:,1],logits[:,2]-logits[:,3]],-1).double()
    base=margins(torch.zeros_like(pre));joint=margins(delta);reference=data['arm_margins'][0];error=float((base-reference).norm()/reference.norm());assert error<=1e-5
    cells=[]
    for family in (0,1):
        uk=[i for i,rw in enumerate(rows) if rw['family']==family and rw['cue']=='British'];us=[i+1 for i in uk];native=base[uk]-base[us];transfer=(native-(joint[uk]-joint[us]))/2;old=data['arm_margins'];full=(old[0,uk,0]-old[0,us,0]-old[38,uk,0]+old[38,us,0])/2
        cells.append(dict(family=family,joint_transfer=float(transfer[:,0].mean()),whole_child_transfer=float(full.mean()),fraction=float(transfer[:,0].mean()/full.mean()),positive=int((transfer[:,0]>0).sum()),unrelated_meanabs=float(transfer[:,1].abs().mean())))
    out=P/'REGIONAL_PRODUCER_GROUP_V1_AUDIT.json';assert not out.exists();result=dict(native_baseline_relative_error=error,group=['attention8','attention9','attention13'],cells=cells,scope='Post-selection joint audit from existing rows. Numerator-swap write deltas add exactly; nonlinear suffix recomputed. Not held-out group identification.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
