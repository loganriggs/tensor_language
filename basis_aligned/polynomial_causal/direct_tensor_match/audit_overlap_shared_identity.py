"""Gauge-invariant shared branch and input-span agreement, plus centered removal.
Conditional scalar utility is not semantic selectivity or native logit intervention.
"""
from pathlib import Path
import json,torch
from local_shared_reader_graph import expand,decode,source_reads
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True);meta=json.loads((P/'JOINT_OVERLAP_V1.json').read_text());S=torch.linalg.inv(d['inverse_root']);I=torch.eye(1152,dtype=S.dtype);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();trueQ=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);branches={};totals={};records=[]
for key,program in programs.items():
 bundle=expand(program);shared=[];total_forms=[];splits=[]
 for j in range(3):
  p=bundle[str(j)];sel=program['pairs'][str(j)]['shared_indices'];mask=torch.zeros(384,dtype=torch.bool);mask[sel]=True;i,k,_=p['product_indices'];assert (mask[i]==mask[k]).all()
  left=dict(p);right=dict(p);left['shared_reader']=p['shared_reader']*mask[None];right['shared_reader']=p['shared_reader']*(~mask)[None];Qsh=decode(left);Qpr=decode(right);Q=decode(p);splits.append(float((Qsh+Qpr-Q).norm()/Q.norm()));shared.append(Qsh);total_forms.append(Q)
 shared=torch.cat(shared);branches[key]=shared;totals[key]=torch.cat(total_forms);centered=z-d['mu'];delta=torch.einsum('ni,oij,nj->no',centered,shared,centered)-torch.einsum('ij,oji->o',d['old_covariance'],shared);reads=source_reads(z,program)-delta;errors=[]
 for j in range(3):
  p=program['pairs'][str(j)];phi=((h@p['h_reader']-.5*reads[:,2*j])/scale-p['alpha'])*(reads[:,2*j+1]/scale-p['beta']);truth=d['pairs'][j]['truth'][ids];errors.append(float((phi-truth).norm()/(truth-truth.mean()).norm()))
 assert max(splits)<1e-10
 records.append(dict(key=key,shared_private_coefficient_replay=max(splits),centered_shared_removal_component_errors=errors))
comparisons=[]
for geometry in meta['winners']:
 keys=[geometry+'_0',geometry+'_26301'];transform=S if geometry=='calibration_shaped' else I;teacher=transform@trueQ@transform;scales=teacher.square().sum((-1,-2)).reshape(3,2).sum(1).sqrt().repeat_interleave(2)
 A,B=[(transform@branches[k]@transform)/scales[:,None,None] for k in keys];cos=float((A*B).sum()/A.norm()/B.norm());relative=float((A-B).norm()/A.norm());bases=[torch.linalg.qr(transform@programs[k]['input_basis'],mode='reduced').Q for k in keys];principal=torch.linalg.svdvals(bases[0].T@bases[1]);pair_cos=[]
 for j in range(3):
  a,b=A[2*j:2*j+2],B[2*j:2*j+2];pair_cos.append(float((a*b).sum()/a.norm()/b.norm()))
 TA,TB=[(transform@totals[k]@transform)/scales[:,None,None] for k in keys];total_cos=float((TA*TB).sum()/TA.norm()/TB.norm());total_difference=float((TA-TB).norm()/TA.norm())
 comparisons.append(dict(total_source_coefficient_cosine=total_cos,total_source_relative_difference=total_difference,geometry=geometry,shared_branch_equal_pair_coefficient_cosine=cos,shared_branch_relative_difference=relative,pair_shared_branch_cosines=pair_cos,input_span_principal_cosine_min=float(principal.min()),input_span_principal_cosine_median=float(principal.median()),input_span_cosines_at_least_point9=int((principal>=.9).sum()),dimensions=len(principal)))
# Five invertible internal-basis gauges preserve the actual shared/private program.
p=programs[meta['winners']['calibration_shaped']];reference=source_reads(z[:32],p);gauges=[]
for seed in range(5):
 g=torch.Generator().manual_seed(29000+seed);U=torch.linalg.qr(torch.randn(128,128,dtype=z.dtype,generator=g)).Q;change=U*torch.linspace(1,5,128,dtype=z.dtype);new=dict(input_basis=p['input_basis']@change,pairs={})
 for key,pair in p['pairs'].items():q=dict(pair);q['shared_map']=torch.linalg.solve(change,pair['shared_map']);new['pairs'][key]=q
 error=float((source_reads(z[:32],new)-reference).norm()/reference.norm());assert error<1e-8;gauges.append(dict(seed=seed,execution_replay=error))
(P/'OVERLAP_SHARED_IDENTITY_V1.json').write_text(json.dumps(dict(records=records,comparisons=comparisons,gauge_controls=gauges,scope='Two fitted restarts per geometry; compare whole shared quadratic branch and shared input subspace, not individual basis vectors. Centered shared-source removal measures conditional scalar utility on opened448, not fresh selectivity, standalone extraction or native logit effects. No stability/adoption threshold selected after observing results.'),indent=2)+'\n');print(json.dumps(dict(records=records,comparisons=comparisons),indent=2))
