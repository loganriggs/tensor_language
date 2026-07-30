"""Follow-up to qk_gtwo_patch.py: is the surviving greater-of-two accuracy under heavy
ablation a STATIC MAGNITUDE PRIOR (fixed digit-logit ordering, no per-pair comparison)
rather than real comparison? Test:
  (A) all-attention mean-ablated -> are final-position logits ~constant across the 72
      prompts? (if so, acc2 there is pure static ordering, not computation)
  (B) static digit ordering: rank the 9 digit tokens by their all-attn-ablated final
      logit; acc2_static = frac of pairs where larger digit outranks smaller.
  (C) the "true comparison" signal = baseline margin minus the static-prior margin.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18')
NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']; NL=len(m.transformer.h)
tok=AutoTokenizer.from_pretrained('gpt2')
def tid(s):
    t=tok(s)['input_ids']; assert len(t)==1; return t[0]
DT={d:tid(' '+str(d)) for d in range(1,10)}
FS="7 3 -> 7\n2 8 -> 8\n"
prompts,meta=[],[]
for a in range(1,10):
    for b in range(1,10):
        if a==b: continue
        prompts.append(FS+f"{a} {b} ->"); meta.append((a,b))
IDS=torch.stack([tok(p,return_tensors='pt')['input_ids'][0] for p in prompts]).to(DEV)
NP,T=IDS.shape; CHUNK=8

@torch.no_grad()
def forward(idx, means=None, ablate_all_attn=False, collect_attn=None):
    B,Tt=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(Tt,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(Tt,Tt,device=DEV,dtype=torch.bool))
    for li in range(NL):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hc=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hc).view(B,Tt,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hc).view(B,Tt,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        attn=a.c_proj(yh.reshape(B,Tt,-1))
        if collect_attn is not None: collect_attn[li]+=attn.sum(0)
        if ablate_all_attn: attn=means[li].unsqueeze(0).expand(B,-1,-1)
        x=x+attn; x=x+blk.mlp(F.rms_norm(x,(D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)

# collect per-position attn means
msum=[torch.zeros(T,D,device=DEV) for _ in range(NL)]
for i in range(0,NP,CHUNK): forward(IDS[i:i+CHUNK],collect_attn=msum)
means=[t/NP for t in msum]

# baseline + all-attn-ablated final logits
def final_logits(ablate):
    out=[]
    for i in range(0,NP,CHUNK): out.append(forward(IDS[i:i+CHUNK],means=means,ablate_all_attn=ablate)[:,-1].float())
    return torch.cat(out)
Lbase=final_logits(False); Labl=final_logits(True)

digit_ids=[DT[d] for d in range(1,10)]
# (A) constancy across prompts of the ablated final logits (over digit tokens)
sub_abl=Labl[:,digit_ids]                       # (72,9)
std_across=float(sub_abl.std(0).mean())         # ~0 => constant
prof=sub_abl.mean(0)                             # static digit-logit profile (1..9)
# (B) static ordering acc + margins
def pair_stats(L):
    accs=0; margins=[]
    for i,(a,b) in enumerate(meta):
        la,lb=float(L[i,DT[a]]),float(L[i,DT[b]]); g=max(a,b); lo=min(a,b)
        accs+= (float(L[i,DT[g]])>float(L[i,DT[lo]]))
        margins.append(float(L[i,DT[g]])-float(L[i,DT[lo]]))
    return accs/len(meta), float(np.mean(margins))
acc_base,mar_base=pair_stats(Lbase)
acc_abl,mar_abl=pair_stats(Labl)
# static profile as a single fixed vector applied to all pairs
acc_static=np.mean([ (float(prof[max(a,b)-1])>float(prof[min(a,b)-1])) for a,b in meta])
mar_static=np.mean([ (float(prof[max(a,b)-1])-float(prof[min(a,b)-1])) for a,b in meta])

res={
 'baseline':{'acc2':round(acc_base,3),'margin':round(mar_base,3)},
 'all_attn_ablated':{'acc2':round(acc_abl,3),'margin':round(mar_abl,3),
                     'final_digit_logit_std_across_prompts':round(std_across,4)},
 'static_prior':{'acc2':round(float(acc_static),3),'margin':round(float(mar_static),3),
                 'digit_logit_profile_1to9':[round(float(v),3) for v in prof]},
 'true_comparison_margin (baseline - all_attn_ablated)':round(mar_base-mar_abl,3),
}
print(json.dumps(res,indent=2),flush=True)
json.dump(res,open(f'{QK}/qk_gtwo_static.json','w'),indent=2)
print('DONE',flush=True)
