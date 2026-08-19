"""Stage D prep: for each certified ownership circuit, dump a human-readable
evidence pack -- sampled contexts with the target token marked, top target
tokens, top preceding tokens, induction (target-seen-earlier) rate, mean
position -- into circuits_stageD/pack_*.json in batches of 19 for parallel
story-writing. Mechanical, no predictions to register (data prep)."""
import json, sys, os, torch, collections
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
os.makedirs(PT+'circuits_stageD',exist_ok=True)
R0=300
reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
disc=reg['disc_mask']; a=reg['assign_disc']
idx_disc=disc.nonzero().squeeze(1)   # global token idx of discovery tokens
def locate(gi):
    row=R0+gi//256; pos=gi%256
    return row,pos
packs=[]; cur=[]
for ci,c in enumerate(reg['certified']):
    j=c['id']
    toks=idx_disc[(a==j).nonzero().squeeze(1)]
    tgt=collections.Counter(); prv=collections.Counter()
    ind=0; possum=0; examples=[]
    sel=toks[torch.randperm(len(toks))[:24]] if len(toks)>24 else toks
    for gi in toks.tolist():
        row,pos=locate(gi)
        t=int(FW[row,pos+1]); p=int(FW[row,pos])
        tgt[t]+=1; prv[p]+=1
        possum+=pos
        if t in FW[row,:pos+1].tolist(): ind+=1
    for gi in sel.tolist():
        row,pos=locate(gi)
        ctx=enc.decode(FW[row,max(0,pos-30):pos+1].tolist())
        target=enc.decode([int(FW[row,pos+1])])
        examples.append(f'...{ctx}[->{target}]')
    entry={'circuit_id':int(j),'owners':c['top'][:3],
           'n':len(toks),
           'top_targets':[enc.decode([t]) for t,_ in tgt.most_common(6)],
           'top_prev':[enc.decode([t]) for t,_ in prv.most_common(6)],
           'induction_rate':round(ind/len(toks),2),
           'mean_pos':round(possum/len(toks),1),
           'examples':examples[:16]}
    cur.append(entry)
    if len(cur)==19:
        packs.append(cur); cur=[]
if cur: packs.append(cur)
for i,p in enumerate(packs):
    json.dump(p,open(PT+f'circuits_stageD/pack_{i}.json','w'),indent=1)
print(f'{len(reg["certified"])} circuits -> {len(packs)} packs')
