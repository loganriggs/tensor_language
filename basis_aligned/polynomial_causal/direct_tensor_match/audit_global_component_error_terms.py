"""Exact two-read value-error decomposition, retaining signs and cancellations."""
from pathlib import Path
import json,torch
from global_mixed_source_graph import source_reads as global_reads
from shared_mixed_source_graph import source_reads as partial_reads
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);truthreads=torch.einsum('ni,oij,nj->no',z,Q,z)
parentmeta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[parentmeta['winner']]
reads={'partial_parent':partial_reads(z,parent)}
if (P/'GLOBAL_SOURCE_BALANCE_V1.json').exists():
 meta=json.loads((P/'GLOBAL_SOURCE_BALANCE_V1.json').read_text());programs=torch.load(P/'GLOBAL_SOURCE_BALANCE_PROGRAMS_V1.pt',weights_only=True)
 for key in meta['winners'].values():reads[key]=global_reads(z,programs[key])
rows=[]
for name,read in reads.items():
 for j,pair in enumerate(d['pairs']):
  A=(h@pair['a']-.5*truthreads[:,2*j])/s-pair['alpha'];B=truthreads[:,2*j+1]/s-pair['beta']
  da=-.5*(read[:,2*j]-truthreads[:,2*j])/s;db=(read[:,2*j+1]-truthreads[:,2*j+1])/s
  terms=torch.stack([da*B,A*db,da*db]);total=terms.sum(0);truth=A*B;den=(truth-truth.mean()).norm()
  direct=(A+da)*(B+db)-truth;replay=float((direct-total).norm()/(total.norm()+1e-30));assert replay<1e-10
  gram=terms@terms.T/den.square();sq=float(total.square().sum()/den.square())
  rows.append(dict(program=name,component=j+1,total_error=float(total.norm()/den),term_errors=(terms.norm(dim=1)/den).tolist(),signed_cross_contributions=[float(2*gram[0,1]),float(2*gram[0,2]),float(2*gram[1,2])],term_energy_fractions=(gram.diag()/sq).tolist(),total_squared_error=sq,identity_replay=replay,exact_a_read_error=float((terms[1]).norm()/den),exact_b_read_error=float(terms[0].norm()/den)))
out=dict(records=rows,term_order=['first_read_error_times_true_second_factor','second_read_error_times_true_first_factor','error_product'],scope='Opened448sites and fixed native h; exact read replacement is a diagnostic oracle, not a deployable cost-matched candidate. Earlier/later native inputs and all3targets remain.')
(P/'GLOBAL_COMPONENT_ERROR_TERMS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
