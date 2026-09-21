from pathlib import Path
import torch,json,time
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);K=torch.load(p/'MIDPOINT_FACTOR_PROGRAMS_V1.pt',weights_only=True)['K'].double();asymmetry=float((K-K.transpose(1,2)).norm()/K.norm());assert asymmetry<1e-6;K=(K+K.transpose(1,2))/2;rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();truth=rows['y'].flatten(0,1).double()@old['scalar_readers']-old['offset'].double();direct=torch.einsum('bi,gij,bj->bg',n,K,m)-old['offset'].double();target_replay=float((direct-truth).norm()/truth.norm());assert target_replay<1e-5;Mn=n.T@n/len(n);Mm=m.T@m/len(m);moment=(Mn/Mn.trace()+Mm/Mm.trace())/2
# Trace-balanced common moment; isotropic comparison has identical tying constraint.
e,U=torch.linalg.eigh(moment+1e-6*moment.trace()/1152*torch.eye(1152));S=(U*e.sqrt()[None,:])@U.T;I=(U*e.rsqrt()[None,:])@U.T;eye=torch.eye(1152,dtype=torch.float64);programs={};records=[];checks=[]
for metric in ['isotropic','common_moment']:
 s,inv=(eye,eye) if metric=='isotropic' else (S,I);allfactors=[]
 for k in K:
  weighted=s@k@s;vals,V=torch.linalg.eigh((weighted+weighted.T)/2);order=vals.abs().argsort(descending=True);vals=vals[order];V=V[:,order];directions=(inv@V)*vals.abs().sqrt()[None,:];sign=vals.sign();full=(directions*sign[None,:])@directions.T;checks.append(float((full-k).norm()/k.norm()));allfactors.append((directions,sign,vals))
 for rank in [1,2,4,8,16]:
  directions=torch.cat([d[:,:rank] for d,sign,vals in allfactors],dim=1);readout=torch.zeros(4*rank,4,dtype=torch.float64)
  for g,(_,sign,_) in enumerate(allfactors):readout[g*rank:(g+1)*rank,g]=sign[:rank]
  pred=((n@directions)*(m@directions))@readout-old['offset'].double();err=(pred-truth).square().sum(0);den=truth.square().sum(0);records.append(dict(metric=metric,rank_per_feature=rank,products=4*rank,input_coefficients=1152*4*rank,aggregate_calibration_error=float((err.sum()/den.sum()).sqrt()),individual_calibration_error=(err/den).sqrt().tolist(),weighted_matrix_error_per_feature=[float((vals[rank:].square().sum()/vals.square().sum()).sqrt()) for _,_,vals in allfactors]));programs[f'{metric}_{rank}']=dict(directions=directions.float(),readout=readout.float(),offset=old['offset'],scalar_readers=old['scalar_readers'],reduced_writers=old['reduced_writers'])
 # Eigen directions are common-metric orthogonal; rotations generally destroy diagonality.
 print(metric,'done',flush=True)
assert max(checks)<1e-8
result=dict(records=records,teacher_symmetry_error=asymmetry,archived_teacher_replay=target_replay,full_eigendecomposition_replay=max(checks),seconds=time.perf_counter()-start,scope='Signed symmetric bilinear forms in normalized residual-space n,m. Same reader evaluated on both inputs, stored once. Calibration-only common moment; exact native scalar target, noheldfitting. Scalar featuredefinitions unchanged, individual reader semanticidentity unproven.')
out=p/'MIDPOINT_TIED_READERS_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(programs=programs,common_moment=moment),p/'MIDPOINT_TIED_READERS_V1.pt');print(json.dumps(result,indent=2))
