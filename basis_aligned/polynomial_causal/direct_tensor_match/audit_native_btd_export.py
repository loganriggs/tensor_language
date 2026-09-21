from pathlib import Path
import torch,json
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match')
torch.set_num_threads(2)
d=torch.load(p/'MIDPOINT_NATIVE_BTD_V1.pt',weights_only=True); e=d['program'];w=d['output_mix'].double()
r=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True)
n=r['n'].flatten(0,1).double();m=r['m'].flatten(0,1).double()
A=e['A'].double().reshape(1152,4,4);B=e['B'].double().reshape(1152,4,4)
blocks=torch.einsum('igr,jgr->gij',A,B)
flat=((n@e['A'].double())*(m@e['B'].double()))@e['readout'].double()-e['offset']
dense=torch.einsum('ti,gij,tj->tg',n,blocks,m)-e['offset']
replay=float((flat-dense).norm()/flat.norm())
write1=flat@e['reduced_writers'].T;write2=(flat@w.T)@old['reduced_writers'].T
write_replay=float((write1-write2).norm()/write1.norm())
result={'export_dense_replay_relative_error':replay,'output_coordinate_change_replay_relative_error':write_replay,'products':16,'input_coefficients':36864,'native_interventions_tested':False}
assert replay<1e-10 and write_replay<1e-10
out=p/'MIDPOINT_NATIVE_BTD_EXPORT_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
