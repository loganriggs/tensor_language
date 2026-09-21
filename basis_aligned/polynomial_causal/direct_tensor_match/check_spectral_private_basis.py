from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from compile_shared_private_pair import compile_common_private
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[]
for case in range(5):
 targets,bases,private,_,_=fixture(case)
 for j,(a,b) in enumerate(GROUPS):
  S=torch.cat([bases[a],bases[b]],1);Q=targets[2*j:2*j+2];block,H,info=compile_common_private(Q,S,private[j],refine_steps=10,private_strategy='spectral');reader=torch.cat([S@block['shared_map'],block['private_reader']],1);decoded=decode(dict(shared_reader=reader,product_indices=block['product_indices'],product_weights=block['product_weights']));error=float((decoded-Q).norm()/Q.norm());assert error<1e-8
  rows.append(dict(case=case,pair=j+1,exact_planted_replay=error,private_bottom_condition=info['private_choice']['private_bottom_condition']))
(P/'SPECTRAL_PRIVATE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Fifteen plantedpairs recover fromfullspace spectralprivate directions thenalternating symmetric-corefit and executablecompilation.'),indent=2)+'\n');print('PASS fifteen full-space private reconstructions and compiled replays')
