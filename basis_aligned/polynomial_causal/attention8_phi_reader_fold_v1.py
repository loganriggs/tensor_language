"""Exact head-wise OV folds for the four MLP8 eigenreaders."""
from pathlib import Path
import json,time,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(24171);tic=time.perf_counter();gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);u=gen['eigenvectors'][:,:4]
 files=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in files if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);base='transformer.h.8.attn.';O=sd[base+'c_proj.weight'].double();V=sd[base+'c_v.weight'].double();V0=sd['transformer.h.0.attn.c_v.weight'].double();mu=sd[base+'lamb'].double();head=torch.stack([u.T@O[:,128*h:128*(h+1)] for h in range(9)]);cur=torch.stack([(1-mu)*head[h]@V[128*h:128*(h+1)] for h in range(9)]);first=torch.stack([mu*head[h]@V0[128*h:128*(h+1)] for h in range(9)])
 x=torch.randn(32,1152,dtype=torch.float64);x0=torch.randn_like(x);values=((1-mu)*(x@V.T)+mu*(x0@V0.T)).reshape(32,9,128);reference=torch.einsum('bhk,hik->bhi',values,head);fold=torch.einsum('bd,hid->bhi',x,cur)+torch.einsum('bd,hid->bhi',x0,first);full=(values.reshape(32,1152)@O.T)@u;rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));program=dict(head_output_readers=head,current_value_readers=cur,first_value_readers=first,mixing=mu)
 result=dict(head_value_fold_error=rel(fold,reference),sum_heads_error=rel(reference.sum(1),full),head_output_reader_shape=list(head.shape),each_value_reader_shape=list(cur.shape),program_scalars=sum(t.numel() for t in program.values()),seconds=time.perf_counter()-tic,scope='Exact linear value-to-fourreader maps for9attention8heads. Current/first normalizedvalue inputs andbothQK routing factors remain external. Firstinput is nativeattention0normalizedinput, not an assumed rawembedding. No nativehead causal ranking or speedupclaim.')
 assert result['head_value_fold_error']<1e-12 and result['sum_heads_error']<1e-12
 torch.save(program,P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt');(P/'ATTENTION8_PHI_READER_FOLD_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
