"""Structural norm-information control in unconstrained bilinear hidden space."""
import hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
 files=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files']
 checkpoint=next(k for k in files if k.endswith('pytorch_model.bin'))
 weights=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
 D=weights['transformer.h.7.mlp.Down.weight'].double();bias=weights['transformer.h.7.mlp.Down_bias'].double()
 head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
 W=torch.cat([head[k] for k in ['k1','k2','current_value']]).double();C=W@D
 Q,_=torch.linalg.qr(C.T,mode='reduced')
 generator=torch.Generator().manual_seed(18090600)
 raw=torch.randn(D.shape[1],generator=generator,dtype=torch.float64)
 v=raw-Q@(Q.T@raw);v=v/v.norm()
 read=C@v;output=D@v
 residual=D-(D@Q)@Q.T
 null_error=float(read.norm()/(C.norm()*v.norm()))
 energy=float(output.square().sum())
 assert null_error<=1e-10 and energy>1e-12
 # h=t*v gives the same reader values Wb for every t, but a nonconstant norm.
 norms={str(t):float((bias+t*output).square().sum()) for t in [-1.,0.,1.]}
 result={'reader_null_relative':null_error,'output_norm_squared_coefficient':energy,
         'output_norm_squared_by_hidden_scale':norms,'down_component_outside_reader_rowspan_ratio':float(residual.norm()/D.norm()),
         'reader_rows':C.shape[0],'hidden_dimension':C.shape[1],
         'down_floating_scalars':D.numel(),'explicit_hidden_gram_floating_scalars':D.shape[1]**2,'folded_reader_floating_scalars':C.numel(),
         'universal_hidden_norm_factorization_through_readers':False,
         'scope':'Counterexample for arbitrary hidden h. Does not establish reachability under h=(Lz)*(Rz), actual token states, or impossibility of computing RMS from full upstream inputs/other weights. It rules out simply replacing the native hidden-space norm by a function of C h alone.',
         'checkpoint_sha256_from_provenance':files[checkpoint],
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 torch.save({'hidden_null_direction':v,'output_direction':output,'reader_direction':read},P/'CITY_MLP7_NORM_INFORMATION_V1_WITNESS.pt')
 (P/'CITY_MLP7_NORM_INFORMATION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
