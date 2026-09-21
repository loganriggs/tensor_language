from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from compile_shared_private_pair import compile_common_private
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 targets,bases,private,maps,templates=fixture(case);replays=[]
 for j,(a,b) in enumerate(GROUPS):
  shared=torch.cat([bases[a],bases[b]],1);Q=targets[2*j:2*j+2]
  block,H,diag=compile_common_private(Q,shared,private[j])
  reader=torch.cat([shared@block['shared_map'],block['private_reader']],1)
  decoded=decode(dict(shared_reader=reader,product_indices=block['product_indices'],product_weights=block['product_weights']))
  replay=float((decoded-Q).norm()/Q.norm());assert replay<1e-8
  replays.append(replay)
 rows.append(dict(case=case,exact_planted_pair_replay=replays))
(P/'SHARED_PRIVATE_COMPLETION_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Five planted whole-block pairwise-sharing graphs,15quadraticpairs. Common cross map and two signed pair compilers recover their exact functions. Native approximation may retain cross residual.'),indent=2)+'\n');print('PASS exact completion and executable decode for fifteen planted quadratic pairs')
