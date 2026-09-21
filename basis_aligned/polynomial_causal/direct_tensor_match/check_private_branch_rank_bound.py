from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[]
for case in range(5):
 targets,bases,private,_,_=fixture(case)
 for j,(a,b) in enumerate(GROUPS):
  shared=torch.cat([bases[a],bases[b]],1);U=torch.linalg.qr(shared,mode='complete').Q;r=shared.shape[1];T=U.T@targets[2*j:2*j+2]@U;B=T[:,:r,r:];C=T[:,r:,r:];M=torch.cat([2**.5*torch.cat(list(B),1),torch.cat(list(C),1)],0);p=private[j].shape[1];sv=torch.linalg.svdvals(M);tail=float(sv[p:].square().sum()/targets[2*j:2*j+2].square().sum());assert tail<1e-20
  rows.append(dict(case=case,pair=j+1,planted_private_width=p,normalized_rank_tail=tail))
(P/'PRIVATE_BRANCH_RANK_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Fifteen plantedquadraticpairs with arbitrary private directions and shared quadraticbranches satisfy the claimed rank restriction.'),indent=2)+'\n');print('PASS fifteen planted private-branch rank controls')
