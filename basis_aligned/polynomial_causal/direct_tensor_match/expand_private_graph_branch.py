"""Fixed shared computations, increased private source capacity. Primary320.
Controls288/384; no fitting or selection using native outcomes. Same capacity
increase supplied to the separate baseline. Frozen primary awaits fresh panel.
"""
from pathlib import Path
import torch,json,time
from quadratic_pair_blocks import compile_pair
from shared_mixed_source_graph import component_scalars
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);old=torch.load(P/'PARTIAL_GRAPH_FROZEN_V1.pt',weights_only=True);base=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);pair=d['pairs'][2];M=[root@Q@root for Q in pair['Qs']];G=sum(m@m for m in M);ev,V=torch.linalg.eigh(G);order=ev.argsort(descending=True);records=[];graphs={};baselines={}
for width in [288,320,384]:
 basis=V[:,order[:width]];projection=d['inverse_root']@basis;cores=[basis.T@m@basis for m in M];compiled=compile_pair(*cores);private={k:pair[k] for k in ['alpha','beta']};private.update(shared_reader=projection@compiled['input_transform'],product_indices=compiled['product_indices'],product_weights=compiled['product_weights'],h_reader=pair['a'])
 for key,Q,core in zip(['a','b'],pair['Qs'],cores):
  smallmean=d['mu']@projection;linear=2*Q@d['mu'];constant=d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q)
  private[key+'_linear']=linear-2*projection@(core@smallmean)
  private[key+'_bias']=constant-d['mu']@linear+smallmean@core@smallmean-torch.trace((projection.T@d['old_covariance']@projection)@core)
 graph={**old,'private_pair':private};phi=component_scalars(d['z'],d['h'],graph);errors=[]
 for j,p in enumerate(d['pairs']):
  true=p['truth'][d['indices']];errors.append(float((phi[d['indices'],j]-true).norm()/(true-true.mean()).norm()))
 # Independent centered dense-core replay for the changed branch.
 coords=(d['z']-d['mu'])@projection;reads=[]
 for Q,core in zip(pair['Qs'],cores):reads.append(d['mu']@Q@d['mu']+torch.trace(d['old_covariance']@Q)+(d['z']-d['mu'])@(2*Q@d['mu'])+((coords@core)*coords).sum(1)-torch.trace((projection.T@d['old_covariance']@projection)@core))
 scale=(d['h'].square().mean(1)+torch.finfo(torch.float32).eps).sqrt();direct=((d['h']@pair['a']-.5*reads[0])/scale-pair['alpha'])*(reads[1]/scale-pair['beta']);replay=float((phi[:,2]-direct).norm()/direct.norm());assert replay<1e-8
 separate={**base,'2':{**private,'residual_writer':old['residual_writer']}};floats=old['residual_writer'].numel()+sum(v.numel() for part in [graph['shared_mixed'],private] for v in part.values() if v.is_floating_point());basefloats=sum(v.numel() for part in separate.values() for v in part.values() if v.is_floating_point())-2*old['residual_writer'].numel();assert floats==basefloats
 graphs[str(width)]=graph;baselines[str(width)]=separate;records.append(dict(private_width=width,graph_source_products=256+width,baseline_source_products=512+width,matched_float_coefficients=floats,graph_integer_indices=private['product_indices'].numel(),per_mode_opened_errors=errors,dense_private_replay=replay,compiler_replay=compiled['diagnostics']['matrix_replay']))
 print(records[-1],flush=True)
out=dict(primary_width=320,records=records,scope='Private branch capacity increase after a fresh subgroup failure. Old seven-prefix errors are opened diagnostics; no claims about repairing that subgroup until new native evaluation. Same branch in baseline and graph; shared branch unchanged.',seconds=time.perf_counter()-start)
(P/'PRIVATE_GRAPH_CAPACITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(graphs,P/'PRIVATE_GRAPH_CAPACITY_PROGRAMS_V1.pt');torch.save(baselines,P/'PRIVATE_GRAPH_CAPACITY_BASELINES_V1.pt')
torch.save(graphs['320'],P/'PARTIAL_GRAPH_FROZEN_V2.pt');torch.save(baselines['320'],P/'PARTIAL_GRAPH_BASELINES_V2.pt')
