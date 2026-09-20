from pathlib import Path
import torch,json
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(4);torch.set_grad_enabled(False);source=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);A=source['A'].double();B=source['B'].double()
# Product gauge: balance column norms before choosing shared input spans.
scale=(B.norm(dim=0)/A.norm(dim=0)).sqrt();A=A*scale;B=B/scale
Ua,sa,Vha=torch.linalg.svd(A,full_matrices=False);Ub,sb,Vhb=torch.linalg.svd(B,full_matrices=False);records=[];programs={}
for rank in [2,4,8,12,16]:
 Pn=Ua[:,:rank];Pm=Ub[:,:rank];Tn=Pn.T@A;Tm=Pm.T@B;Ah=Pn@Tn;Bh=Pm@Tm;errors=[]
 for g in range(4):
  idx=slice(4*g,4*g+4);K=A[:,idx]@B[:,idx].T;Kh=Ah[:,idx]@Bh[:,idx].T;errors.append(float((Kh-K).norm()/K.norm()))
 records.append(dict(shared_input_rank_each=rank,reader_coefficients=2*1152*rank+2*rank*16,products=16,coefficient_error_per_scalar=errors,scope='Coefficient error relative to confirmed approximate scalar program, not original teacher or native functional error.'))
 programs[rank]=dict(Pn=Pn.float(),Pm=Pm.float(),Tn=Tn.float(),Tm=Tm.float(),offset=source['offset'],readout=source['readout'],scalar_readers=source['scalar_readers'],reduced_writers=source['reduced_writers'])
assert max(records[-1]['coefficient_error_per_scalar'])<1e-12
out=p/'MIDPOINT_SHARED_READERS_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='CPU structural proposal only. Full-span replay checked; reduced-span programs require native evaluation. No confirmation data used to choose spans.'),indent=2)+'\n');torch.save(dict(programs=programs),p/'MIDPOINT_SHARED_READERS_V1.pt');print(json.dumps(records,indent=2))
