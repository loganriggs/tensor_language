"""Compare shared linear versus quadratic parent collapses of producer secants."""
from pathlib import Path
import torch,json
from folded_producer_cubic_weights_v1 import weights as load_weights
from folded_normalized_router_v1 import rotary
from cubic_secant_block_v1 import gram_kernel
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'FOLDED_PRODUCER_SHARED_PARENT_V1_RESULT.json';assert not out.exists()
 bind=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in bind if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu');C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'];(q1,k1,q2,k2,v,o),_,_=load_weights(sd,C,'cpu')
 saved=torch.load(P/'FOLDED_PRODUCER_CUBIC_SECANT_V1_ARTIFACT.pt',weights_only=True);records=[];artifacts=[]
 for arm,a in enumerate(saved):
  original=a['components'];mix=a['mix'];direction=a['direction'];axis_order=direction.norm(dim=-1).argsort().tolist()
  for nshared,name in [(1,'shared_linear_parent'),(2,'shared_quadratic_parent')]:
   comp=original.clone();axes=axis_order[:nshared]
   for mask in range(8):
    for axis in axes:
     if mask&(1<<axis):comp[14+mask,axis]=0
   artifacts.append(dict(arm=arm,kind=name,components=comp,mix=mix,shared_axes=axes))
   combo=torch.cat([original,comp]);mm=torch.block_diag(mix,mix)
   for pos in (7,0):
    r=rotary(8,128).T@rotary(pos,128);weights=(q1,torch.cat([torch.einsum('ab,hbd->had',r,k1),torch.zeros_like(k1)],-1),q2,torch.cat([torch.einsum('ab,hbd->had',r,k2),torch.zeros_like(k2)],-1),v,o)
    g,k=gram_kernel(combo,mm,weights);go,gn,h=g[:16,:16],g[16:,16:],g[:16,16:];ko,kn,kcross=k[:,:16,:16].sum(0),k[:,16:,16:].sum(0),k[:,:16,16:].sum(0);io,inn=torch.linalg.inv(go),torch.linalg.inv(gn)
    old=torch.trace(io@ko);new=torch.trace(inn@kn);cross=(h*(io@kcross@inn)).sum();difference=(old+new-2*cross).clamp_min(0);nopair=torch.trace(torch.linalg.solve(go[:14,:14],ko[:14,:14]));energies=(inn[None]@k[:,16:,16:]@inn[None]).diagonal(dim1=-2,dim2=-1)*gn.diagonal()[None]
    records.append(dict(arm=arm,kind=name,position=pos,shared_axes=axes,source_reader_count=6-nshared,full_projection_relative_error=float((difference/old).sqrt()),marginal_capture_retained=float((new-nopair)/(old-nopair)),original_capture=float(old),new_capture=float(new),gram_condition=float(torch.linalg.cond(gn)),block_layer_energies=energies[:,-2:].sum(-1).reshape(3,9).sum(1).tolist()))
 result=dict(records=records,scope='Unfitted weight-only collapse of one or two tiny reader differences in an exact secant representation. Errors concern the entire projected producer function, not isolated block-native behavior. Layer energies are descriptive coefficient energies in the fixed reference metric.')
 torch.save(artifacts,P/'FOLDED_PRODUCER_SHARED_PARENT_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
