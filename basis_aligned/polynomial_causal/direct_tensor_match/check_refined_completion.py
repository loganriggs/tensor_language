from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from compile_shared_private_pair import compile_common_private
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[]
for case in range(5):
 T,bases,private,_,_=fixture(case);rng=torch.Generator().manual_seed(40000+case)
 for j,(a,b) in enumerate(GROUPS):
  shared=torch.cat([bases[a],bases[b]],1);U=torch.linalg.qr(torch.cat([shared,private[j]],1),mode='reduced').Q;noise=torch.randn(2,U.shape[1],U.shape[1],dtype=T.dtype,generator=rng);noise=(noise+noise.transpose(-1,-2))/2;Q=T[2*j:2*j+2]+.02*T[2*j:2*j+2].norm()*(U@noise@U.T)/noise.norm()
  _,old,_=compile_common_private(Q,shared,private[j]);block,H,info=compile_common_private(Q,shared,private[j],refine_steps=10);before=float((old-Q).norm()/Q.norm());after=float((H-Q).norm()/Q.norm());assert after<=before+1e-10
  reader=torch.cat([shared@block['shared_map'],block['private_reader']],1);decoded=decode(dict(shared_reader=reader,product_indices=block['product_indices'],product_weights=block['product_weights']));replay=float((decoded-H).norm()/H.norm());assert replay<1e-8
  rows.append(dict(case=case,pair=j+1,initial_error=before,refined_error=after,execution_replay=replay,iterations=info['refinement']['iterations']))
(P/'REFINED_COMPLETION_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Fifteen perturbed planted pairs exercise changed private forms, monotone refinement, full residual accounting and executable compiled replay.'),indent=2)+'\n');print('PASS fifteen perturbed-pair refinements and compiled replays')
