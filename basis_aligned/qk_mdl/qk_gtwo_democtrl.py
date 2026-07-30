"""§40 red-team control: is the greater-of-two "static magnitude prior" (the all-
attention-ablated final-position digit-logit profile) actually IN-CONTEXT COPYING of
the few-shot demo answers?

Baseline (qk_gtwo_static.py) uses demos "7 3 -> 7\n2 8 -> 8\n" whose answers are 7,8 —
EXACTLY where the ablated profile peaks. Test: swap the demo answers to other digits,
keeping format identical, and re-measure the ablated profile. If the peak MOVES to the
new demo answers -> it's in-context copying. If it stays magnitude-ordered/peaked at the
top regardless -> genuine magnitude prior. Also a zero-shot (no demo) variant.

forward + prompt construction copied VERBATIM from qk_gtwo_static.py.
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

CHUNK=8
meta=[]
for a in range(1,10):
    for b in range(1,10):
        if a==b: continue
        meta.append((a,b))
digit_ids=[DT[d] for d in range(1,10)]

def run_header(FS, name):
    """Return the all-attention-ablated final-position digit-logit profile (digits 1..9)
    plus baseline and static-prior pair stats, for a given few-shot header string FS."""
    prompts=[FS+f"{a} {b} ->" for (a,b) in meta]
    IDS=torch.stack([tok(p,return_tensors='pt')['input_ids'][0] for p in prompts]).to(DEV)
    NP,T=IDS.shape
    # collect per-position attn means
    msum=[torch.zeros(T,D,device=DEV) for _ in range(NL)]
    for i in range(0,NP,CHUNK): forward(IDS[i:i+CHUNK],collect_attn=msum)
    means=[t/NP for t in msum]
    def final_logits(ablate):
        out=[]
        for i in range(0,NP,CHUNK): out.append(forward(IDS[i:i+CHUNK],means=means,ablate_all_attn=ablate)[:,-1].float())
        return torch.cat(out)
    Lbase=final_logits(False); Labl=final_logits(True)
    sub_abl=Labl[:,digit_ids]                       # (72,9)
    std_across=float(sub_abl.std(0).mean())         # ~0 => constant static profile
    prof=sub_abl.mean(0)                            # static digit-logit profile (1..9)
    def pair_stats(L):
        accs=0; margins=[]
        for i,(a,b) in enumerate(meta):
            g=max(a,b); lo=min(a,b)
            accs+=(float(L[i,DT[g]])>float(L[i,DT[lo]]))
            margins.append(float(L[i,DT[g]])-float(L[i,DT[lo]]))
        return accs/len(meta), float(np.mean(margins))
    acc_base,mar_base=pair_stats(Lbase)
    acc_abl,mar_abl=pair_stats(Labl)
    acc_static=float(np.mean([(float(prof[max(a,b)-1])>float(prof[min(a,b)-1])) for a,b in meta]))
    prof_l=[round(float(v),3) for v in prof]
    peak=int(np.argmax(prof.cpu().numpy()))+1
    top2=list(np.argsort(prof.cpu().numpy())[::-1][:2]+1)
    return {'name':name,'FS':FS,'T':T,
            'profile_1to9':prof_l,'peak_digit':peak,'top2_digits':[int(x) for x in top2],
            'std_across_prompts':round(std_across,4),
            'baseline_acc2':round(acc_base,3),'ablated_acc2':round(acc_abl,3),
            'static_acc2':round(acc_static,3),
            'baseline_margin':round(mar_base,3),'ablated_margin':round(mar_abl,3)}

# header variants. demo answers noted in comment.
HEADERS=[
 ("7 3 -> 7\n2 8 -> 8\n", "standard_demos_ans_7_8"),   # answers 7,8 (= baseline peak)
 ("4 1 -> 4\n2 5 -> 5\n", "swapped_demos_ans_4_5"),     # answers 4,5
 ("3 1 -> 3\n1 2 -> 2\n", "swapped_demos_ans_3_2"),     # answers 3,2 (small)
 ("5 1 -> 5\n1 4 -> 4\n", "swapped_demos_ans_5_4"),     # answers 5,4 (order mirror of standard)
 ("", "zero_shot_no_demos"),                            # no demonstrated answers
]
results=[run_header(FS,name) for FS,name in HEADERS]

# quantify: correlation of each ablated profile with (i) magnitude ramp 1..9 and
# (ii) a "demo indicator" that is +1 at the two demo-answer digits, 0 elsewhere.
mag=np.arange(1,10,dtype=float); mag=(mag-mag.mean())/mag.std()
def demo_indicator(ans):
    v=np.zeros(9);
    for d in ans: v[d-1]=1.0
    return (v-v.mean())/v.std()
DEMO_ANS={'standard_demos_ans_7_8':[7,8],'swapped_demos_ans_4_5':[4,5],
          'swapped_demos_ans_3_2':[3,2],'swapped_demos_ans_5_4':[5,4]}
for r in results:
    p=np.array(r['profile_1to9']); pz=(p-p.mean())/(p.std()+1e-9)
    r['corr_with_magnitude']=round(float(np.dot(pz,mag)/9),3)
    if r['name'] in DEMO_ANS:
        ind=demo_indicator(DEMO_ANS[r['name']])
        r['corr_with_demo_answers']=round(float(np.dot(pz,ind)/9),3)
        # partial: how much of the profile's demo-position bump is above the magnitude baseline
        # residual after regressing out magnitude, correlated with demo indicator
        beta=np.dot(pz,mag)/np.dot(mag,mag); resid=pz-beta*mag
        rz=(resid-resid.mean())/(resid.std()+1e-9)
        r['corr_demo_after_removing_magnitude']=round(float(np.dot(rz,ind)/9),3)

out={'results':results,'note':'profile = all-attention mean-ablated final-position digit logits, mean over 72 pairs'}
print(json.dumps(out,indent=2),flush=True)
json.dump(out,open(f'{QK}/qk_gtwo_democtrl.json','w'),indent=2)
print('DONE',flush=True)
