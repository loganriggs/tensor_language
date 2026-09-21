"""Greedy product removal for modes1/2, preserving mode3's private program."""
from pathlib import Path
import torch,json,time
from shared_mixed_source_graph import component_scalars,source_reads
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);fit=json.loads((P/'MIXED_PRODUCT_NATIVE_FIT_V1.json').read_text());p=torch.load(P/'MIXED_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True)[fit['winner']]
root=torch.linalg.inv(d['inverse_root']);A=root@p['left_reader'];B=root@p['right_reader'];K=.5*((A.T@A)*(B.T@B)+(A.T@B)*(B.T@A));norm=K.diag().sqrt();A=A/norm.sqrt();B=B/norm.sqrt();left=p['left_reader']/norm.sqrt();right=p['right_reader']/norm.sqrt();K=.5*((A.T@A)*(B.T@B)+(A.T@B)*(B.T@A));rhs=torch.einsum('ir,oij,jr->ro',A,d['teacher'][:4],B)
inverse=torch.linalg.inv(K);weights=inverse@rhs;ids=torch.arange(len(K));costsum=0.;initial=float(d['teacher'][:4].square().sum()-(weights*rhs).sum())
while len(ids)>256:
 costs=weights.square().sum(1)/inverse.diag();i=int(costs.argmin());keep=torch.arange(len(ids))!=i;col=inverse[keep,i];diag=inverse[i,i];costsum+=float(costs[i]);weights=weights[keep]-col[:,None]/diag*weights[i];inverse=inverse[keep][:,keep]-torch.outer(col,col)/diag;ids=ids[keep]
exact=torch.linalg.solve(K[ids[:,None],ids[None,:]],rhs[ids]);weight_replay=float((weights-exact).norm()/exact.norm());assert weight_replay<1e-8;weights=exact
L=left[:,ids];R=right[:,ids];readout=weights*d['scales'][None,:4];raw=torch.einsum('ir,ro,jr->oij',L,readout,R);Qs=.5*(raw+raw.transpose(-1,-2));lin=[];bias=[]
for j,pair in enumerate(d['pairs'][:2]):
 for k,Q in enumerate(pair['Qs']):
  pos=2*j+k;linear=2*Q@d['mu'];constant=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q)
  lin.append(linear-2*Qs[pos]@d['mu']);bias.append(constant-d['mu']@linear+d['mu']@Qs[pos]@d['mu']-torch.trace(d['old_covariance']@Qs[pos]))
shared=dict(left_reader=L,right_reader=R,product_weights=readout,source_linear=torch.stack(lin,1),source_bias=torch.stack(bias),h_readers=p['h_readers'][:,:2],alpha=p['alpha'][:2],beta=p['beta'][:2])
private=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True)['2'];writer=private.pop('residual_writer');assert torch.equal(writer,p['residual_writer'])
program=dict(shared_mixed=shared,private_pair=private,residual_writer=writer)
scalars=component_scalars(d['z'],d['h'],program);errors=[]
for j,pair in enumerate(d['pairs']):
 truth=pair['truth'][d['indices']];errors.append(float((scalars[d['indices'],j]-truth).norm()/(truth-truth.mean()).norm()))
reads=source_reads(d['z'],program);dense=torch.einsum('ni,oij,nj->no',d['z'],Qs,d['z'])+d['z']@shared['source_linear']+shared['source_bias'];replay=float((reads[:,:4]-dense).norm()/dense.norm());assert replay<1e-8
metric=torch.stack([root@Q@root for Q in Qs])/d['scales'][:4,None,None];coefficient=float((metric-d['teacher'][:4]).norm()/d['teacher'][:4].norm());prediction=((initial+costsum)/float(d['teacher'][:4].square().sum()))**.5;assert abs(coefficient-prediction)<1e-8
floats=writer.numel()+sum(v.numel() for pr in [shared,private] for v in pr.values() if v.is_floating_point());base=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text());ratios=[a/b for a,b in zip(errors,base['baseline_errors'])]
out=dict(parent_winner=fit['winner'],shared_products=256,private_products=256,total_products=512,stored_float_scalars=floats,stored_integer_indices=private['product_indices'].numel(),per_mode_errors=errors,ratios_to_baseline=ratios,first_two_coefficient_error=coefficient,greedy_predicted_loss_replay=abs(coefficient-prediction),schur_weights_replay=weight_replay,dense_source_replay=replay,predictions=dict(pred_a_instrument=replay<1e-8 and weight_replay<1e-8,pred_b_fidelity=max(errors)<=.15 and max(ratios)<=1.1,pred_c_simplicity=floats<=1.01*897804),seconds=time.perf_counter()-start,scope='Opened seven-prefix diagnostic. Product deletion uses only exact coefficient objective; third component unchanged except numerical RMS replay. Allthree componenttargets retained; no native/OOD adoption yet.')
(P/'PARTIAL_MIXED_GRAPH_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(program,P/'PARTIAL_MIXED_GRAPH_V1.pt');print(json.dumps(out,indent=2))
