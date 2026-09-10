"""Canonical output modes of each fitted tensor block, without token labels in fit."""
from pathlib import Path
import json,torch,tiktoken
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
torch.set_num_threads(2)
s=torch.load(P/'WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_CHECKPOINT.pt',map_location='cpu',weights_only=False)
m=MultioutputQuadraticBlocks(1152);m.load_state_dict(s['best']['model']);e,c=m.components();e=e.detach();c=c.detach()
q,r=torch.linalg.qr(e.transpose(-1,-2));h=r[:,None]@c@r[:,None].transpose(-1,-2)
u=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)['lm_head.weight'];w=s['best']['writer'];y=torch.cat([v.double()@w for v in u.split(2048)])
tok=tiktoken.get_encoding('gpt2');modes=[];rows=[]
for g in range(16):
    core=h[g].flatten(1);gh=core@core.T;ev,vec=torch.linalg.eigh(gh);root=(vec*ev.clamp_min(0).sqrt())@vec.T
    factor=y[:,g*4:g*4+4]@root;ev,vec=torch.linalg.eigh(factor.T@factor);ev=ev.flip(0);vec=vec.flip(1);left=factor@vec/ev.clamp_min(1e-30).sqrt()[None,:]
    if left[:,0].abs().argmax()<tok.n_vocab and left[left[:,0].abs().argmax(),0]<0:left[:,0]*=-1
    modes.append(left[:,0]);descriptions=[]
    for j in range(2):
        axis=left[:tok.n_vocab,j]
        def top(sign):
            ids=torch.topk(sign*axis,10).indices.tolist()
            return [dict(id=i,text=tok.decode([i]),loading=float(axis[i])) for i in ids]
        descriptions.append(dict(mode=j,energy_fraction=float(ev[j]/ev.sum()),uniform_output_energy=float(left[:,j].sum().square()/len(left)),outside_gpt2_vocab_energy=float(left[tok.n_vocab:,j].square().sum()),largest_absolute_ids=left[:,j].abs().topk(5).indices.tolist(),positive=top(1),negative=top(-1)))
    rows.append(dict(block=g,modes=descriptions))
allm=torch.stack(modes);cos=allm@allm.T;tri=cos[torch.triu_indices(16,16,1).unbind()].abs();order=torch.argsort(tri,descending=True);idx=torch.triu_indices(16,16,1)
centered=allm-allm.mean(1,keepdim=True);centered=centered/centered.norm(dim=1,keepdim=True);cc=centered@centered.T;ct=cc[idx.unbind()].abs()
result=dict(mean_absolute_centered_leading_output_cosine=float(ct.mean()),maximum_absolute_centered_leading_output_cosine=float(ct.max()),scope='Canonical left singular modes of each learned block tensor. Sign arbitrary; top tokens describe weights only, not behavioral tasks. Fit unconverged, no independent restart.',mean_absolute_leading_output_cosine=float(tri.mean()),minimum_absolute_leading_output_cosine=float(tri.min()),maximum_absolute_leading_output_cosine=float(tri.max()),strongest_pairs=[dict(blocks=idx[:,i].tolist(),absolute_cosine=float(tri[i])) for i in order[:8]],blocks=rows)
(P/'MULTIOUTPUT_BLOCK_OUTPUT_MODES_V1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='blocks'}))
for row in rows[:4]:print(json.dumps(dict(block=row['block'],top_positive=[v['text'] for v in row['modes'][0]['positive']],top_negative=[v['text'] for v in row['modes'][0]['negative']])))
