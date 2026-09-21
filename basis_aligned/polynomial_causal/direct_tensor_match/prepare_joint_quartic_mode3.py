"""Prepare exact full-input quartic teacher and low-rank source student frames."""
from pathlib import Path
import torch,json
from quartic_pair_metric import inner
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);cal=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);mode=torch.load(p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True);z=cal['z'].flatten(0,1).double();mu=z.mean(0);delta=z-mu;cov=delta.T@delta/len(delta);v,V=torch.linalg.eigh(cov);assert v.min()>0;root=(V*v.sqrt())@V.T;inv=(V*v.rsqrt())@V.T;d=z.shape[1];D=d+3
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin';state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();Down=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'].double()[0];a=mode['A'][:,2];b=mode['B'][:,2];alpha=mode['mean_n']@a;beta=mode['mean_m']@b
Qs=[];sources={}
for key,reader in [('a',a),('b',b)]:
 channel=lam*(Down.T@reader);raw=L.T@(channel[:,None]*R);Q=(raw+raw.T)/2;Qs.append(Q);eig,U=torch.linalg.eigh(root@Q@root);ids=eig.abs().argsort(descending=True)[:16];sources[key]=dict(linear=2*Q@mu,constant=mu@Q@mu+torch.trace(cov@Q),initial=U[:,ids]*eig[ids].abs().sqrt(),sign=eig[ids].sign())
scale=cal['recipient_scale'].flatten().double();n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();t=((n+.5*m)@a)*scale;truth=torch.stack([((z@Q)*z).sum(1) for Q in Qs],1);phi=((t-.5*truth[:,0])/scale-alpha)*(truth[:,1]/scale-beta)
fixedA=torch.zeros(D,D,dtype=torch.float64);fixedB=torch.zeros_like(fixedA)
fixedA[:d,-1]=fixedA[-1,:d]=-.25*sources['a']['linear'];fixedA[d,-1]=fixedA[-1,d]=.5;fixedA[d+1,-1]=fixedA[-1,d+1]=-.5*alpha;fixedA[-1,-1]=-.5*sources['a']['constant']
fixedB[:d,-1]=fixedB[-1,:d]=.5*sources['b']['linear'];fixedB[d+1,-1]=fixedB[-1,d+1]=-.5*beta;fixedB[-1,-1]=sources['b']['constant']
teacherA=fixedA.clone();teacherB=fixedB.clone();teacherA[:d,:d]-=.5*Qs[0];teacherA[-1,-1]+=.5*torch.trace(cov@Qs[0]);teacherB[:d,:d]+=Qs[1];teacherB[-1,-1]-=torch.trace(cov@Qs[1])
joint=torch.cat([delta,(t-t.mean())[:,None],(scale-scale.mean())[:,None]],1);jcov=joint.T@joint/len(joint);ev,U=torch.linalg.eigh(jcov);jroot=(U*ev.clamp_min(0).sqrt())@U.T;frames={};checks=[]
for metric in ['isotropic','covariance']:
 H=torch.eye(D,dtype=torch.float64)
 if metric=='covariance':H[:-1,:-1]=jroot
 H[d,-1]=t.mean();H[d+1,-1]=scale.mean();A=H.T@teacherA@H;B=H.T@teacherB@H;entry=dict(A=A,B=B,J=H.T[:,:d]@inv,unit=H[-1].clone(),teacher_norm=inner(A,B,A,B))
 for key,M in [('a',fixedA),('b',fixedB)]:
  F=H.T@M@H;ev0,U0=torch.linalg.eigh(F);ids=ev0.abs().argsort(descending=True)[:2];entry[key+'_fixed_vectors']=U0[:,ids].clone();entry[key+'_fixed_values']=ev0[ids].clone();checks.append(float((F-(U0[:,ids]*ev0[ids])@U0[:,ids].T).norm()/F.norm()))
 # Check complete student-factor frame against direct affine/quadratic matrices.
 for key,factor in [('a',-.5),('b',1.)]:
  info=sources[key];u=info['initial'];sign=info['sign'];P=inv@u;Q=(P*sign)@P.T;raw=(fixedA if key=='a' else fixedB).clone();raw[:d,:d]+=factor*Q;raw[-1,-1]-=factor*(u.square().sum(0)*sign).sum();X=torch.cat([entry['J']@u,entry[key+'_fixed_vectors'],entry['unit'][:,None]],1);weights=torch.cat([factor*sign,entry[key+'_fixed_values'],(-factor*(u.square().sum(0)*sign).sum()).reshape(1)]);checks.append(float((H.T@raw@H-(X*weights)@X.T).norm()/(H.T@raw@H).norm()))
 frames[metric]=entry
pred=[]
for key in ['a','b']:
 info=sources[key];P=inv@info['initial'];pred.append(info['constant']+delta@info['linear']+(((delta@P).square()-info['initial'].square().sum(0))*info['sign']).sum(1))
ph=((t-.5*pred[0])/scale-alpha)*(pred[1]/scale-beta);baseline=float((ph-phi).norm()/(phi-phi.mean()).norm());assert abs(baseline-.3771067718710561)<1e-7 and max(checks)<1e-8
common=dict(mu=mu,source_inverse_root=inv,sources=sources,h_reader=a,residual_writer=torch.linalg.solve(mode['R_U'],mode['writer']),alpha=alpha,beta=beta,t=t,scale=scale,true_phi=phi)
torch.save(dict(common=common,frames=frames),p/'JOINT_QUARTIC_MODE3_INPUTS_V1.pt');result=dict(frame_replay=max(checks),initial_native_mode_variation_error=baseline,rank_per_source=16,augmented_dimension=D,parameter_count=2*d*16,negative_joint_covariance_eigenvalues=int((ev<0).sum()),scope='Original mode3 numerator. Bothmetrics use native calibration centering and exactaffine branches, common covariance-preconditionedsource parameters; isotropicmetriccontrol isnot wholly datafree. Covariance frame includes jointdelta-z/t/s secondmoments, notfullGaussianmomentM. ExplicitRMS denominator excludedfromcoefficientloss and used for nativecalibration probes.');(p/'JOINT_QUARTIC_MODE3_PREP_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
