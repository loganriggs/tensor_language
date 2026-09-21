"""Separate Gaussian fourth-moment closure from downstream sensitivity independence."""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)

def gaussian_error(D,l,b,mu,cov):
 mean=torch.trace(D@cov)+mu@D@mu+l@mu+b
 linear=2*D@mu+l
 return 2*torch.trace((D@cov)@(D@cov))+linear@cov@linear+mean.square()

# Small Gaussian positive controls with explicit random probes and MC uncertainty.
toys=[]
for seed in range(5):
 g=torch.Generator().manual_seed(9200+seed);dtype=torch.float64
 A=torch.randn(7,7,generator=g,dtype=dtype);D=(A+A.T)/2;L=torch.randn(7,7,generator=g,dtype=dtype);cov=L@L.T+.2*torch.eye(7,dtype=dtype);mu=torch.randn(7,generator=g,dtype=dtype);l=torch.randn(7,generator=g,dtype=dtype);b=torch.randn((),generator=g,dtype=dtype)
 z=torch.randn(100000,7,generator=g,dtype=dtype)@torch.linalg.cholesky(cov).T+mu;y=(torch.einsum('ni,ij,nj->n',z,D,z)+z@l+b).square();exact=gaussian_error(D,l,b,mu,cov);se=y.std()/len(y)**.5;standard=float(abs(y.mean()-exact)/se);assert standard<5
 toys.append(dict(seed=seed,analytic=float(exact),empirical=float(y.mean()),standard_errors=standard))

d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);T=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);n=len(d['z']);z=d['z'];h=d['h'];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();q=torch.einsum('ni,oij,nj->no',z,T,z)
# Decode compact winner and partial parent independently to native Q,l,b.
meta=json.loads((P/'SHARED_PRIVATE_DIRECTIONS_V1.json').read_text());compact=torch.load(P/'SHARED_PRIVATE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True)[meta['winners']['16']]
raw=torch.einsum('ir,ro,jr->oij',compact['left_reader'],compact['product_weights'],compact['right_reader']);CQ=(raw+raw.transpose(-1,-2))/2;CQ[5]+=(compact['square_reader']*compact['square_weights'])@compact['square_reader'].T
pm=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[pm['winner']];part=parent['shared_mixed'];raw=torch.einsum('ir,ro,jr->oij',part['left_reader'],part['product_weights'],part['right_reader']);PQ=(raw+raw.transpose(-1,-2))/2
b=parent['private_pair'];i,j,k=b['product_indices'].long();u=b['shared_reader'][:,i];v=b['shared_reader'][:,j];left=torch.where((k==1)[None],u+v,u);right=torch.where((k==1)[None],u-v,v);raw=torch.einsum('ir,ro,jr->oij',left,b['product_weights'],right);PQ=torch.cat([PQ,(raw+raw.transpose(-1,-2))/2])
PL=torch.cat([part['source_linear'],torch.stack([b['a_linear'],b['b_linear']],1)],1);PB=torch.cat([part['source_bias'],torch.stack([b['a_bias'],b['b_bias']])])
programs={'partial':(PQ,PL,PB),'compact':(CQ,compact['source_linear'],compact['source_bias'])}
from shared_mixed_source_graph import source_reads as partial_source
from compact_source_graph import source_reads as compact_source
decode=[]
for name,program,fn in [('partial',parent,partial_source),('compact',compact,compact_source)]:
 Q,l,bias=programs[name];x=z[:32];actual=fn(x,program);dense=torch.einsum('ni,oij,nj->no',x,Q,x)+x@l+bias;rel=float((actual-dense).norm()/actual.norm());assert rel<1e-10;decode.append(dict(program=name,replay=rel))
weights=[]
for j,pair in enumerate(d['pairs']):
 A=(h@pair['a']-.5*q[:,2*j])/s-pair['alpha'];B=q[:,2*j+1]/s-pair['beta'];weights.extend([(-.5*B/s).square(),(A/s).square()])
rows=[]
for panel,ids in [('train',torch.arange(1536)),('opened',d['indices'])]:
 x=z[ids];mu=x.mean(0);xc=x-mu;cov=xc.T@xc/len(x)
 for name,(Q,l,bias) in programs.items():
  D=Q-T;errors=torch.einsum('ni,oij,nj->no',x,D,x)+x@l+bias
  for o in range(6):
   squared=errors[:,o].square();w=weights[o][ids];actual=(w*squared).mean();independent=w.mean()*squared.mean();gaussian=gaussian_error(D[o],l[:,o],bias[o],mu,cov)
   energy=w*squared;chunkids=ids//64;chunks=torch.stack([energy[chunkids==i].sum() for i in chunkids.unique()])
   rows.append(dict(program=name,panel=panel,component=o//2+1,read='first' if o%2==0 else 'second',empirical_source_mse=float(squared.mean()),matched_gaussian_source_mse=float(gaussian),gaussian_to_empirical_source_ratio=float(gaussian/squared.mean()),independent_to_exact_weighted_ratio=float(independent/actual),gaussian_independent_to_exact_ratio=float(w.mean()*gaussian/actual),exact_weighted_error_energy=float(actual),sensitivity_effective_sites=float(w.sum().square()/w.square().sum()),top_one_percent_weighted_error_share=float(energy.topk(max(1,round(.01*len(ids)))).values.sum()/energy.sum()),max_chunk_weighted_error_share=float(chunks.max()/chunks.sum())))
out=dict(native_decode_replays=decode,gaussian_controls=toys,records=rows,predictions=dict(pred_a_gaussian_formula=max(r['standard_errors'] for r in toys)<5,pred_b_covariance_sufficiency=all(abs(r['gaussian_to_empirical_source_ratio']-1)<=.1 for r in rows),pred_c_sensitivity_independence=all(abs(r['independent_to_exact_weighted_ratio']-1)<=.1 for r in rows)),scope='Diagnostic on train1536 and opened448historical sites, not fresh data. Matched Gaussian uses each panel mean/covariance, therefore isolates fourth-moment closure rather than covariance estimation shift. Weighted terms are first-order component-value error contributions, not full product error; signed cross terms remain in separate exact audit.')
(P/'VALUE_METRIC_GEOMETRY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['predictions']))
