"""Fold native head8.2 key/value readers through MLP7's output map."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
 binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files']
 checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
 assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==binding[checkpoint]
 weights=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
 head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
 W=torch.cat([head[k] for k in ['k1','k2','current_value']]).float()
 left=weights['transformer.h.7.mlp.Left.weight'].float();right=weights['transformer.h.7.mlp.Right.weight'].float()
 down=weights['transformer.h.7.mlp.Down.weight'].float();bias=weights['transformer.h.7.mlp.Down_bias'].float()
 exact=W.double()@down.double();offset=W.double()@bias.double()
 program={'left':left,'right':right,'folded_down':exact.float(),'folded_bias':offset.float()}
 gen=torch.Generator().manual_seed(18095200);x=F.rms_norm(torch.randn(40,1152,generator=gen),(1152,)).double()
 hidden=(x@left.double().T)*(x@right.double().T)
 unfused=(hidden@down.double().T+bias.double())@W.double().T
 fused=hidden@exact.T+offset
 fp64=float((unfused-fused).norm()/unfused.norm())
 rounded=hidden@program['folded_down'].double().T+program['folded_bias'].double()
 rounding=float((rounded-unfused).norm()/unfused.norm())
 assert fp64<=1e-10 and rounding<=1e-4
 torch.save(program,P/'CITY_MLP7_READERS_V1_PROGRAM.pt')
 count=sum(v.numel() for v in program.values());baseline=sum(v.numel() for v in [left,right,down,bias,W])
 result={'fp64_identity_relative':fp64,'rounded_program_relative':rounding,'floating_scalars':count,'fp32_bytes':4*count,'unfolded_floating_scalars':baseline,'unfolded_fp32_bytes':4*baseline,'checkpoint_sha256':binding[checkpoint],
         'scope':'Weight identity plus synthetic CPU control. One normalized native MLP7 city input required; native context, denominators, query generation and suffix not included. No causal or OOD claim.'}
 (P/'CITY_MLP7_READERS_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
