"""Wave 3 prep: mixed-window evidence packs. For each unconfirmed circuit
(and the 6 confirmed, excluded), gather 10 example contexts from window A
(rows 300-512 discovery half) AND 10 from window C (rows 120-300, third
atlas) for the SAME cluster, plus per-window top targets/prevs and class
stats. Stories must describe what is common to BOTH windows. Mechanical
prep."""
import json, sys, os, torch, collections
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import FW
import tiktoken
enc=tiktoken.get_encoding('gpt2')
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
DEV='cuda'
reg=torch.load(PT+'circuits_registry.pt',weights_only=False)
big=torch.load(PT+'circuit_atlas_big.pt',weights_only=False)
thr=torch.load(PT+'circuit_atlas_third.pt',weights_only=False)
keys=reg['keys']
Mb=torch.stack([big['fingerprints'][k].float().to(DEV) for k in keys])
disc=reg['disc_mask'].to(DEV)
mu=Mb[:,disc].mean(1,keepdim=True); sd=Mb[:,disc].std(1,keepdim=True).clamp_min(1e-8)
M3=torch.stack([thr['fingerprints'][k].float().to(DEV) for k in keys])
base3=thr['base'].float().to(DEV)
wp3=base3<=base3.median()
Z3=((M3-mu)/sd)[:,wp3].T; Z3=Z3/Z3.norm(dim=1,keepdim=True).clamp_min(1e-8)
C=reg['centroids'].to(DEV)
a3=(Z3@C.T).argmax(1).cpu()
idx3=wp3.nonzero().squeeze(1).cpu()
ad=reg['assign_disc']; idxA=reg['disc_mask'].nonzero().squeeze(1)
def ctx(gi,R0):
    row=R0+gi//256; pos=gi%256
    return (f"...{enc.decode(FW[row,max(0,pos-28):pos+1].tolist())}"
            f"[->{enc.decode([int(FW[row,pos+1])])}]")
def stats(gis,R0):
    tgt=collections.Counter(); ind=0
    for gi in gis:
        row=R0+gi//256; pos=gi%256
        t=int(FW[row,pos+1]); tgt[enc.decode([t])]+=1
        if t in FW[row,:pos+1].tolist(): ind+=1
    return ([t for t,_ in tgt.most_common(5)],
            round(ind/max(len(gis),1),2))
confirmed={c['circuit_id'] for c in json.load(open(PT+'circuits_FINAL_certified.json'))}
entries=[]
for c in reg['certified']:
    j=c['id']
    if j in confirmed: continue
    tA=idxA[(ad==j).nonzero().squeeze(1)]
    tC=idx3[(a3==j).nonzero().squeeze(1)]
    if len(tC)<8 or len(tA)<8: continue
    gA=tA[torch.randperm(len(tA))[:10]].tolist()
    evenC=tC[((tC//256)%2==0)]
    if len(evenC)<5: continue
    gC=evenC[torch.randperm(len(evenC))[:10]].tolist()
    tgA,indA=stats(tA.tolist()[:200],300); tgC,indC=stats(tC.tolist()[:200],120)
    entries.append({'circuit_id':int(j),'owners':c['top'][:3],
        'windowA_examples':[ctx(g,300) for g in gA],
        'windowC_examples':[ctx(g,120) for g in gC],
        'windowA_top_targets':tgA,'windowC_top_targets':tgC,
        'induction_rate_A':indA,'induction_rate_C':indC})
os.makedirs(PT+'circuits_wave3',exist_ok=True)
n=0
for i in range(0,len(entries),20):
    json.dump(entries[i:i+20],open(PT+f'circuits_wave3/pack_{n}.json','w'),
              indent=1); n+=1
print(f'{len(entries)} circuits -> {n} packs')
