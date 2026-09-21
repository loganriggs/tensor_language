"""Re-execute consensus joint tensor without claiming cheaper atom count."""
from pathlib import Path
import copy,json,torch
from shared_mixed_source_graph import component_scalars,source_reads
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
c=torch.load(P/'PROFILED_CONSENSUS_V1.pt',weights_only=True)
allp=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)
p=copy.deepcopy(allp[c['reference']]);s=p['shared_mixed'];s['product_weights']*=c['atom_coefficients'][:,None]
raw=torch.einsum('ir,ro,jr->oij',s['left_reader'],s['product_weights'],s['right_reader']);Qhat=(raw+raw.transpose(-1,-2))/2
for o,Q in enumerate([q for pair in d['pairs'][:2] for q in pair['Qs']]):
    delta=Q-Qhat[o]
    s['source_linear'][:,o]=2*delta@d['mu']
    s['source_bias'][o]=torch.trace(d['old_covariance']@delta)-d['mu']@delta@d['mu']
dense=torch.einsum('ni,oij,nj->no',d['z'],Qhat,d['z'])+d['z']@s['source_linear']+s['source_bias']
replay=float((source_reads(d['z'],p)[:,:4]-dense).norm()/dense.norm());assert replay<1e-8
phi=component_scalars(d['z'],d['h'],p);errors=[]
for i,pair in enumerate(d['pairs']):
    truth=pair['truth'][d['indices']];errors.append(float((phi[d['indices'],i]-truth).norm()/(truth-truth.mean()).norm()))
baseline=[.0305840973,.0275584497,.1194180748]
out=dict(per_component_error=errors,baseline=baseline,dense_source_replay=replay,source_products=512,
    predictions=dict(pred_a_replay=replay<1e-8,pred_b_absolute=max(errors)<=.15,pred_c_relative=all(a<=1.1*b for a,b in zip(errors,baseline))),
    scope='Previously opened states; preserves original centered linear/mean terms. Three consensus joint tensor directions still use256shared products, plus256private. No new fresh native or semantic result.')
(P/'PROFILED_CONSENSUS_EXECUTION_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
