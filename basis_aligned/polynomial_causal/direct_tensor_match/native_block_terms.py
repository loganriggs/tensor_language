from pathlib import Path
import torch,json,time,sys
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');sys.path.insert(0,str(p));from block_term_fit import fit,evaluate,block_cosines
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter();old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);K=torch.load(p/'MIDPOINT_FACTOR_PROGRAMS_V1.pt',weights_only=True)['K'].double();roots=torch.load(p/'MIDPOINT_SHARED_FIT_PROGRAMS_V1.pt',weights_only=True)['regularized_calibration_roots'];Sn=roots['n'];Sm=roots['m'];F=Sn@K@Sm;normalizer=F.norm();F=F/normalizer;left=F.permute(1,0,2).reshape(1152,-1);right=F.reshape(-1,1152);_,U=torch.linalg.eigh(left@left.T);_,V=torch.linalg.eigh(right.T@right);U=U.flip(1)[:,:32];V=V.flip(1)[:,:32];T=U.T@F@V;outside=float((F-U@T@V.T).norm());projection_identity=abs(float(T.square().sum())+outside**2-1);assert projection_identity<1e-10
A=[];B=[]
for t in T:
 a,s,b=torch.linalg.svd(t,full_matrices=False);A.append(a[:,:4]*s[:4].sqrt()[None,:]);B.append(b[:4].T*s[:4].sqrt()[None,:])
fixed=[torch.eye(4),torch.stack(A,dim=1),torch.stack(B,dim=1)];fixederror=float((evaluate(fixed)-T).norm());fits=[]
for init in ['random','fixed']:
 for seed in ([0,1,2] if init=='random' else [0]):
  for lr in [.003,.01]:
   r=fit(T,4,4,261149+seed,lr,1800,initial=None if init=='random' else fixed);fits.append(dict(initialization=init,seed=seed,lr=lr,error=r['error'],factors=r['factors']));print(init,seed,lr,r['error'],flush=True)
best=min(fits,key=lambda z:z['error']);w,a,b=best['factors'];colnorm=w.norm(dim=0);w=w/colnorm[None,:];a=a*colnorm.sqrt()[None,:,None];b=b*colnorm.sqrt()[None,:,None];cond=float(torch.linalg.cond(w));assert cond<1e8
physical_left=torch.linalg.solve(Sn,U);physical_right=torch.linalg.solve(Sm,V)
def export(factors):
 w,a,b=factors;A=physical_left@a.reshape(32,16)*normalizer.sqrt();B=physical_right@b.reshape(32,16)*normalizer.sqrt();inverse=torch.linalg.inv(w);readers=old['scalar_readers']@inverse.T;writers=old['reduced_writers']@w;offset=old['offset'].double()@inverse.T
 return dict(A=A.float(),B=B.float(),readout=old['readout'],offset=offset,scalar_readers=readers,reduced_writers=writers)
program=export((w,a,b));fixedprogram=export(fixed);rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();truth=rows['y'].flatten(0,1).double()@old['scalar_readers']-old['offset'].double()
def cal(pr,w):
 h=((n@pr['A'].double())*(m@pr['B'].double()))@pr['readout'].double()-pr['offset'];return float((h@w.T-truth).norm()/truth.norm())
calmixed=cal(program,w);calfixed=cal(fixedprogram,torch.eye(4));duality=float((program['scalar_readers'].T@program['reduced_writers']-torch.eye(4)).norm());assert duality<1e-8
byseed=[min([r for r in fits if r['initialization']=='random' and r['seed']==seed],key=lambda z:z['error']) for seed in [0,1,2]];agreement=[block_cosines(byseed[0]['factors'],r['factors']) for r in byseed[1:]]
result=dict(output_modes=4,input_modes=32,blocks=4,rank_per_block=4,projection_error=outside,projection_pythagorean_error=projection_identity,fixed_core_error=fixederror,best_core_error=best['error'],fixed_full_weighted_error=(outside**2+fixederror**2)**.5,best_full_weighted_error=(outside**2+best['error']**2)**.5,best={k:v for k,v in best.items() if k!='factors'},output_direction_condition=cond,reader_writer_duality=duality,calibration_fixed=calfixed,calibration_mixed=calmixed,random_restart_block_cosines=agreement,fits=[{k:v for k,v in r.items() if k!='factors'} for r in fits],seconds=time.perf_counter()-start,scope='Originalweightderived four-readout tensor beforeindependentoutputrank truncation. Common32inputprojectionerror explicit. Learned4outputgroupings vsfixed4 at16products. Noheldfitting. Stability/circuitidentification unproven.')
out=p/'MIDPOINT_NATIVE_BTD_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(program=program,fixed_program=fixedprogram,output_mix=w),p/'MIDPOINT_NATIVE_BTD_V1.pt');print(json.dumps({k:v for k,v in result.items() if k!='fits'},indent=2))
