"""Exact outer-factor mixing with common writer; never changes fixed component gates."""
from pathlib import Path
import math,json,torch
from global_mixed_source_graph import source_reads as global_reads
from shared_mixed_source_graph import source_reads as partial_reads
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);ids=d['indices'];z=d['z'][ids];h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None]
Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);true_reads=torch.einsum('ni,oij,nj->no',z,Q,z);a=torch.stack([pair['a'] for pair in d['pairs']],1);alpha=torch.stack([pair['alpha'] for pair in d['pairs']]);beta=torch.stack([pair['beta'] for pair in d['pairs']]);A=(h@a-.5*true_reads[:,::2])/s-alpha;B=true_reads[:,1::2]/s-beta;true=A*B;total=true.sum(1);den=(total-total.mean()).norm()
meta=json.loads((P/'SHARED_PRIVATE_DIRECTIONS_V1.json').read_text());programs=torch.load(P/'SHARED_PRIVATE_DIRECTIONS_PROGRAMS_V1.pt',weights_only=True);readers={}
for key in meta['winners'].values():
 p=programs[key];q=global_reads(z,p);q[:,5]+=(z@p['square_reader']).square()@p['square_weights'];readers[key]=q
pm=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[pm['winner']];readers['partial_parent']=partial_reads(z,parent)
rotations={'identity':torch.eye(3,dtype=z.dtype)}
O=torch.eye(3,dtype=z.dtype);O[1:,1:]=torch.tensor([[1.,-1.],[1.,1.]],dtype=z.dtype)/math.sqrt(2);rotations['last_two_45deg']=O
for seed in [9021,9022]:
 g=torch.Generator().manual_seed(seed);rotations[f'orthogonal_seed{seed}']=torch.linalg.qr(torch.randn(3,3,dtype=z.dtype,generator=g))[0]
rows=[]
for name,q in readers.items():
 Ah=(h@a-.5*q[:,::2])/s-alpha;Bh=q[:,1::2]/s-beta;pred=Ah*Bh;parent_error=float((pred.sum(1)-total).norm()/den)
 for label,O in rotations.items():
  t=(A@O)*(B@O);v=(Ah@O)*(Bh@O)
  replay=max(float((t.sum(1)-total).norm()/total.norm()),float((v.sum(1)-pred.sum(1)).norm()/pred.sum(1).norm()))
  variation=(t-t.mean(0)).norm(dim=0);errors=((v-t).norm(dim=0)/variation).tolist();change=float((t-true).norm()/true.norm());aggregate=float((v.sum(1)-t.sum(1)).norm()/den)
  assert replay<1e-10 and abs(aggregate-parent_error)<1e-10
  rows.append(dict(program=name,rotation=label,per_component_errors=errors,combined_variation_error=aggregate,original_component_change=change,total_replay=replay))
out=dict(records=rows,predictions=dict(pred_a_exact_total=max(r['total_replay'] for r in rows)<1e-10,pred_b_nontrivial_components=any(r['rotation']!='identity' and r['original_component_change']>.1 for r in rows)),scope='Exact algebraic factor gauge for three components with one common writer. Rotated components have different meanings. Original per-component gates remain binding and failed; total preservation neither identifies semantic units nor validates this local approximation OOD. No products removed.')
(P/'OUTER_COMPONENT_GAUGE_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
