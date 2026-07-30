"""Causally verify the KEY_newline heads are BOUNDARY ANCHORS: knockout the cluster and measure the
capital-target CE damage SPLIT by whether the capitalized target is soon-after-newline (within 6
tokens of a preceding newline = post-boundary) vs not. Boundary-anchor hypothesis: damage
concentrates post-newline. Held slice FW[448:600], paired per-token SEs.
"""
import json, sys, numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18'); NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))[448:600][:, :129]
tok=AutoTokenizer.from_pretrained('gpt2')
CAP=torch.zeros(V,dtype=torch.bool); NLK=torch.zeros(V,dtype=torch.bool)
for i in range(50257):
    s=tok.convert_ids_to_tokens(i)
    if s is None: continue
    core=s.replace('Ġ',''); lead=s.startswith('Ġ')
    if lead and len(core) and core[0].isupper(): CAP[i]=True
    if 'Ċ' in s: NLK[i]=True
CAP,NLK=CAP.to(DEV),NLK.to(DEV)
NLHEADS=[(c['layer'],c['head']) for c in json.load(open('qk_selection_census_v2.json'))['programmatic'] if c['top_predicate']=='KEY_newline']

@torch.no_grad()
def run(idx, keep, MEAN, collect=False):
    B,T=idx.shape; x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cb,sb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool)); means={}
    for li in range(NL):
        b=m.transformer.h[li]; a=b.attn; x=b.lambdas[0]*x+b.lambdas[1]*x0 if li else (b.lambdas[0]+b.lambdas[1])*x0
        h=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(h).view(B,T,NH,HD),(HD,)); return apply_rot(z,cb,sb)
        v=a.c_v(h).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        pat=((torch.einsum('bqhd,bkhd->bhqk',q,k)/HD)*(torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD)).masked_fill(~mask,0.0)
        yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        if collect: means[('h',li)]=yh.mean((0,1))
        if keep is not None:
            for hh in range(NH):
                if ('h',li,hh) not in keep: yh[:,:,hh,:]=MEAN[('h',li)][hh]
        x=x+a.c_proj(yh.reshape(B,T,-1)); x=x+b.mlp(F.rms_norm(x,(D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30).float(), means

_,MEAN=run(FW[:8].to(DEV),None,None,True)
ALL=[('h',li,hh) for li in range(NL) for hh in range(NH)]
keep=set(ALL)-{('h',li,hh) for (li,hh) in NLHEADS}
# per-token CE real vs knockout; classify capital targets by post-newline distance
def collect_ce(kp):
    ces=[]; capm=[]; postnl=[]
    for i in range(0,len(FW),4):
        idx=FW[i:i+4].to(DEV); lg=run(idx[:,:-1],kp,MEAN)[0] if kp is not None else run(idx[:,:-1],None,None)[0]
        tgt=idx[:,1:]; ce=F.cross_entropy(lg.reshape(-1,V),tgt.reshape(-1),reduction='none').view(idx.shape[0],-1)
        # post-newline: any newline key in the preceding 6 positions of the QUERY position
        isnl=NLK[idx[:,:-1]].float()  # (B,T)
        win=torch.zeros_like(isnl)
        for d in range(1,7): win[:,d:]+=isnl[:,:-d]
        pn=(win>0)
        ces.append(ce.reshape(-1).cpu()); capm.append(CAP[tgt].reshape(-1).cpu()); postnl.append(pn.reshape(-1).cpu())
    return torch.cat(ces),torch.cat(capm),torch.cat(postnl)
ce_r,capm,pn=collect_ce(None)
ce_k,_,_=collect_ce(keep)
d=ce_k-ce_r
def stat(mask):
    dm=d[mask]; return {'n':int(mask.sum()),'dCE':round(float(dm.mean()),5),'SE':round(float(dm.std()/np.sqrt(dm.numel())),6)}
res={'cluster':[list(x) for x in NLHEADS],
     'capital_post_newline': stat(capm & pn),
     'capital_not_post_newline': stat(capm & ~pn),
     'noncapital_post_newline': stat(~capm & pn),
     'all_capital': stat(capm)}
for k,v in res.items():
    if isinstance(v,dict) and 'dCE' in v: print(f"{k}: dCE +{v['dCE']} (SE {v['SE']}, n={v['n']})",flush=True)
json.dump(res,open(f'{QK}/qk_newline_anchor.json','w'),indent=2)
print("QK NEWLINE ANCHOR DONE",flush=True)
