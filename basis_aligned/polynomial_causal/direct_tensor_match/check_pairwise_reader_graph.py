from pathlib import Path
import json,torch
import pairwise_reader_graph as new
from local_shared_reader_graph import source_reads as old_reads,expand as old_expand,decode
from pack_reader_graph_artifacts import packed
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 rng=torch.Generator().manual_seed(34000+case);r=2*(1+case%3);n=2*r+4;k=r+2;d=n+2
 rand=lambda *s:torch.randn(*s,dtype=torch.float64,generator=rng)
 basis=rand(d,r);pairs={};i=torch.arange(0,n,2).repeat_interleave(2);indices=torch.stack([i,i+1,torch.tensor([1,2]*(n//2))])
 for j in range(3):
  pairs[str(j)]=dict(shared_indices=torch.arange(k),private_indices=torch.arange(k,n),shared_map=rand(r,k),private_reader=rand(d,n-k),product_indices=indices.clone(),product_weights=rand(n,2),a_linear=rand(d),a_bias=rand(()),b_linear=rand(d),b_bias=rand(()))
 old=dict(input_basis=basis,pairs=pairs);graph=packed(new.from_common(old));z=rand(17,d)
 execution=float((new.source_reads(z,graph)-old_reads(z,old)).norm()/old_reads(z,old).norm());assert execution<1e-10
 targets=torch.cat([decode(p) for p in old_expand(old).values()]);templates=[graph['pairs'][str(j)] for j in range(3)];metric=new.PairwiseOverlapMetric(targets,templates);params=new.parameters_from_program(graph,torch.eye(d,dtype=torch.float64))
 oracle=float(metric.loss(params,dense=True)[0]);assert oracle<1e-7
 perturbed=[(p+.02*rand(*p.shape)).requires_grad_() for p in params];losses=[metric.loss(perturbed)[0],metric.loss(perturbed,dense=True)[0],metric.loss(perturbed,detach=False)[0]];grads=[torch.autograd.grad(loss,perturbed) for loss in losses]
 discrepancy=max(float((a-b).norm()/(1+a.norm())) for other in grads[1:] for a,b in zip(grads[0],other));assert discrepancy<1e-8
 rows.append(dict(case=case,execution_replay=execution,oracle_regularized_squared_error=oracle,dense_envelope_gradient_replay=discrepancy))
# Full native execution and physical price, all four unchanged parent functions.
parents=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True);data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);z=data['z'][data['indices']];native=[]
for key,old in parents.items():
 graph=packed(new.from_common(old));replay=float((new.source_reads(z,graph)-old_reads(z,old)).norm()/old_reads(z,old).norm());cost=new.price(graph)
 assert replay<1e-8 and cost['stored_floats']==cost['physical_storage_floats']==1070604 and cost['source_total_multiplications']==1060224
 native.append(dict(key=key,execution_replay=replay,**cost))
(P/'PAIRWISE_READER_GRAPH_PREFLIGHT_V1.json').write_text(json.dumps(dict(toys=rows,native=native,scope='Exact graph rewrite and literal physical price; no optimization or accuracy gain. Five planted product programs verify variable-projected loss and gradients.'),indent=2)+'\n');print('PASS five toy gradient/replay controls and four full native rewrites with physical cost')
