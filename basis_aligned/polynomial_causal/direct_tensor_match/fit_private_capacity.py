"""Allocate the unused portion of the original arithmetic budget from weights."""
from pathlib import Path
import itertools,json,time,torch
from orthogonal_private_varpro import OrthogonalPrivateMetric
from pairwise_reader_graph import GROUPS
from pairwise_graph_assessment import Assessment
from export_orthogonal_private import export
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();plan=json.loads((P/'PRIVATE_CAPACITY_PLAN_V1.json').read_text())
data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);audit=Assessment(data);parents=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True);saved=torch.load(P/'PENCIL_JOINT_REFIT_STATES_V1.pt',weights_only=True);rows=[];graphs={};budget=plan['additional_private_directions'];widths=list(range(0,budget+1,plan['increments']))
for geometry in plan['metrics']:
 key=geometry+'_inherited';raw=saved[key];template=parents[key];A,inv=(audit.S,data['inverse_root']) if geometry=='calibration_shaped' else (audit.I,audit.I);shared=[torch.cat([raw[a],raw[b]],1) for a,b in GROUPS];private=[torch.linalg.qr(v,mode='reduced').Q for v in raw[3:]];metric=OrthogonalPrivateMetric(A@audit.Q@A,shared);base,states=metric.loss(private);individual=[];banks=[];spectra=[]
 for j,(S,D,W,C,B) in enumerate(states):
  H=D@C@D.T+S@(B-W@C@W.T)@S.T;R=metric.targets[j]-H;individual.append(float(R.square().sum()));U=torch.linalg.qr(torch.cat([S,D],1),mode='reduced').Q;M=(R@R).sum(0);M=M-U@(U.T@M);M=M-(M@U)@U.T;M=(M+M.T)/2;e,V=torch.linalg.eigh(M);V=V[:,-budget:];V=V-U@(U.T@V);V=torch.linalg.qr(V,mode='reduced').Q;# strongest modes first
  banks.append(V.flip(1));spectra.append(e[-budget:].flip(0).tolist())
 profiles=[]
 for j in range(3):
  profile=[]
  for count in widths:
   trial=list(private);trial[j]=torch.cat([private[j],banks[j][:,:count]],1);loss,_=metric.loss(trial);error=3*float(loss)-sum(individual[k] for k in range(3) if k!=j);assert error>-1e-9;profile.append(dict(added=count,pair_squared_error=error))
  assert all(b['pair_squared_error']<=a['pair_squared_error']+1e-8 for a,b in zip(profile,profile[1:]));profiles.append(profile)
 allocations=[v for v in itertools.product(widths,repeat=3) if sum(v)<=budget];lookup=[{r['added']:r['pair_squared_error'] for r in profile} for profile in profiles];allocation=min(allocations,key=lambda v:sum(lookup[j][v[j]] for j in range(3)))
 params=[torch.cat([private[j],banks[j][:,:allocation[j]]],1) for j in range(3)];loss,components=metric.loss(params);dense=metric.loss(params,dense=True)[0];assert abs(float(loss-dense))<1e-8
 row=dict(geometry=geometry,initial_error=float(base.sqrt()),error=float(dense.sqrt()),allocation=list(allocation),profiles=profiles,residual_spectra=spectra)
 try:
  graph,diag=export(components,metric.scales,template,A,inv);graph=audit.correct(graph);scores=audit.assess(graph);field='covariance_error' if geometry=='calibration_shaped' else 'native_error';assert abs(scores[field]-float(dense.sqrt()))<1e-8
  expected_cost=1047648+plan['source_cost_per_direction']*sum(allocation);expected_storage=1058124+plan['stored_float_cost_per_direction']*sum(allocation);assert scores['source_total_multiplications']==expected_cost<=plan['source_ceiling'];assert scores['stored_floats']==scores['physical_storage_floats']==expected_storage
  graphs[geometry]=graph;row.update(instrument=True,compiler=diag,**scores)
 except (ValueError,RuntimeError,AssertionError) as error:row.update(instrument=False,failure=str(error))
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('compiler','profiles','residual_spectra')}),flush=True)
primary=rows[0];valid=primary['instrument'];base=plan['baseline'];pred=dict(pred_a_instrument=all(r['instrument'] for r in rows),pred_b_fidelity=valid and all(primary[k]<=1.1*base[k] for k in ('native_error','covariance_error')) and all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base['per_mode_errors'])) and all(a<=1.1*b for a,b in zip(primary['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])),pred_c_arithmetic=all(r['instrument'] and r['source_total_multiplications']<=plan['source_ceiling'] for r in rows))
torch.save(graphs,P/'PRIVATE_CAPACITY_PROGRAMS_V1.pt');(P/'PRIVATE_CAPACITY_V1.json').write_text(json.dumps(dict(plan=plan,records=rows,predictions=pred,seconds=time.monotonic()-start),indent=2)+'\n');print(pred,flush=True)
