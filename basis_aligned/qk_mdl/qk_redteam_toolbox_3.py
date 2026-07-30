"""RED-TEAM ATTACK 3 (§61): copy-family redundancy ratio 3.86 (joint dCE +0.430 vs
sum-of-solos 0.111), minimal 4-head subset {L8H3,L5H5,L7H3,L8H4} recovers 87%, z=24.9 vs
random head-sets.

Independent reproduction + apples-to-apples audit of the random control:
  (1) Reproduce joint dCE, sum-of-solos, redundancy ratio, and greedy minimal subset with
      an INDEPENDENT implementation (forward compute lines copied VERBATIM from
      qk_unsup_redundant.py / qk_bracket_patch.py). Same held slice, mean-ablation.
  (2) Audit the random control for hidden advantages to the family:
      (a) Same-count random (as §61): 6 heads drawn from all 18 layers.
      (b) SAME-LAYER-BAND random: 6 heads drawn only from L5-L14 (the family's own layer
          span) -> removes any "deep vs shallow layer" capacity confound.
      (c) Family on RANDOM positions: ablate the family on a random valid position set
          (not its own top-activating union) -> tests whether the joint dCE is inflated by
          evaluating on positions cherry-picked for the family.
  (3) Double-counting check: recompute sum-of-solos on the identical shared posset and
      confirm the ratio.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import json, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); np.random.seed(0)
DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18')
NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h)
tok=AutoTokenizer.from_pretrained('gpt2')
FINEWEB=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL=128
HELD=FINEWEB[448:600,:SEQL].to(DEV); NHELD=HELD.shape[0]; BATCH=4
KPOS=150; NRAND=40
def pid(hn): return int(hn[1:hn.index('H')]), int(hn[hn.index('H')+1:])
_special={tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL=np.array(sorted(_special))
COPY_FAMILY=['L8H3','L8H4','L13H0','L14H7','L7H3','L5H5']
FAMILY_HEADS=set(pid(h) for h in COPY_FAMILY)
FAMILY_LAYERS=sorted(set(li for (li,h) in FAMILY_HEADS))   # [5,7,8,13,14]
LAYER_BAND=range(min(FAMILY_LAYERS),max(FAMILY_LAYERS)+1)  # 5..14 inclusive

# ============ VERBATIM forward (set ablation + collect) from qk_unsup_redundant.py ============
@torch.no_grad()
def forward(idx, ablate=None, want_yh=False, collect=False):
    B,Tt=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(Tt,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(Tt,Tt,device=DEV,dtype=torch.bool))
    acc={'yh':[]} if want_yh else None; out={} if collect else None; ab={}
    if ablate is not None and ablate[0]=='heads':
        for (li,h) in ablate[1]: ab.setdefault(li,[]).append(h)
    for li in range(NL):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hc=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hc).view(B,Tt,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hc).view(B,Tt,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        if collect:
            Wp=a.c_proj.weight.view(D,NH,HD)
            comp=torch.einsum('bthc,dhc->bthd',yh,Wp)
            out[li]={'hnorm':comp.norm(dim=-1).cpu().numpy()}; del comp
        if li in ab:
            yh=yh.clone()
            for h in ab[li]: yh[:,:,h]=means_yh[li][:,h].unsqueeze(0)
        if want_yh: acc['yh'].append(yh.sum(0))
        x=x+a.c_proj(yh.reshape(B,Tt,-1))
        x=x+blk.mlp(F.rms_norm(x,(D,)))
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    if want_yh: return logits,acc
    if collect: return logits,out
    return logits

print("PASS1 means + hnorm...", flush=True)
YH_SUM={li:torch.zeros(SEQL,NH,HD,device=DEV) for li in range(NL)}
hnorm=np.zeros((NL,NH,NHELD,SEQL),np.float32)
for i in range(0,NHELD,BATCH):
    _,acc=forward(HELD[i:i+BATCH],want_yh=True)
    for li in range(NL): YH_SUM[li]+=acc['yh'][li]
    _,out=forward(HELD[i:i+BATCH],collect=True); b=out[0]['hnorm'].shape[0]
    for li in range(NL): hnorm[li,:,i:i+b]=out[li]['hnorm'].transpose(2,0,1)
means_yh={li:YH_SUM[li]/NHELD for li in range(NL)}
held_np=HELD.cpu().numpy()
pos_th=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)
is_special_h=np.isin(held_np,SPECIAL)
valid_h=(pos_th>0)&(pos_th<SEQL-1)&~is_special_h

def build_posset(heads,kpos=KPOS):
    best={}
    for (li,h) in heads:
        a=hnorm[li,h].reshape(-1).copy(); a[~valid_h.reshape(-1)]=-1e30
        tk=np.argpartition(a,-kpos)[-kpos:]
        for f in tk:
            s,p=divmod(int(f),SEQL); hv=float(hnorm[li,h,s,p])
            if (s,p) not in best or hv>best[(s,p)]: best[(s,p)]=hv
    seqs=np.array([k[0] for k in best],np.int64); poss=np.array([k[1] for k in best],np.int64)
    tgts=held_np[seqs,poss+1].astype(np.int64)
    return {'seqs':seqs,'poss':poss,'tgts':tgts,'n':len(seqs)}

def random_posset(nP,seed=0):
    rng=np.random.RandomState(seed); vf=np.where(valid_h.reshape(-1))[0]
    pick=rng.choice(vf,size=nP,replace=False); seqs=pick//SEQL; poss=pick%SEQL
    return {'seqs':seqs,'poss':poss,'tgts':held_np[seqs,poss+1].astype(np.int64),'n':nP}

def se(x): return float(np.asarray(x).std(ddof=1)/math.sqrt(len(x)))
@torch.no_grad()
def eval_ce(ps,ablate_heads):
    seqs,poss,tgts=ps['seqs'],ps['poss'],ps['tgts']; N=ps['n']; ce=np.zeros(N)
    ab=('heads',set(ablate_heads)) if ablate_heads else None
    by_seq={}
    for j,s in enumerate(seqs): by_seq.setdefault(int(s),[]).append(j)
    uniq=np.array(sorted(by_seq))
    for i in range(0,len(uniq),BATCH):
        sb=uniq[i:i+BATCH]; logits=forward(HELD[sb],ablate=ab).float()
        loc={int(s):bi for bi,s in enumerate(sb)}
        rows,ps_,tg_,jj_=[],[],[],[]
        for s in sb:
            for j in by_seq[int(s)]:
                rows.append(loc[int(s)]); ps_.append(int(poss[j])); tg_.append(int(tgts[j])); jj_.append(j)
        lg=logits[torch.tensor(rows,device=DEV),torch.tensor(ps_,device=DEV)]
        ce_b=F.cross_entropy(lg,torch.tensor(tg_,device=DEV),reduction='none').cpu().numpy()
        for kk,j in enumerate(jj_): ce[j]=ce_b[kk]
    return ce

heads=[pid(h) for h in COPY_FAMILY]
ps=build_posset(heads)
print(f"shared posset n={ps['n']}", flush=True)
base=eval_ce(ps,[])
solos={}
for hn,hp in zip(COPY_FAMILY,heads):
    d=eval_ce(ps,[hp])-base; solos[hn]={'dCE':round(float(d.mean()),4),'SE':round(se(d),4)}
    print(f"  solo {hn} {solos[hn]}", flush=True)
jd=eval_ce(ps,heads)-base
joint={'joint_dCE':round(float(jd.mean()),4),'joint_dCE_SE':round(se(jd),4)}
sum_solo=sum(max(solos[h]['dCE'],0.0) for h in COPY_FAMILY)
sum_solo_signed=sum(solos[h]['dCE'] for h in COPY_FAMILY)
redundancy=round(joint['joint_dCE']/sum_solo,2)
print(f"  JOINT {joint} sum_solo+={sum_solo:.4f} signed={sum_solo_signed:.4f} redundancy={redundancy}", flush=True)

# greedy minimal subset
remaining=list(COPY_FAMILY); chosen=[]; curve=[]
while remaining:
    best_h,best_d=None,-1e9
    for hn in remaining:
        d=float((eval_ce(ps,[pid(x) for x in chosen+[hn]])-base).mean())
        if d>best_d: best_d,best_h=d,hn
    chosen.append(best_h); remaining.remove(best_h)
    curve.append({'added':best_h,'set':list(chosen),'cum_dCE':round(best_d,4)})
    print(f"  greedy +{best_h} cum={best_d:.4f} set={chosen}", flush=True)
thr=0.8*joint['joint_dCE']; minimal=None
for c in curve:
    if c['cum_dCE']>=thr:
        minimal={'subset':c['set'],'size':len(c['set']),'cum_dCE':c['cum_dCE'],
                 'frac_of_joint':round(c['cum_dCE']/joint['joint_dCE'],2)}; break

# ---- controls ----
def random_control(pool,label,posset,seed=0):
    rng=np.random.RandomState(seed)
    allh=[(li,h) for li in pool for h in range(NH) if (li,h) not in FAMILY_HEADS]
    rd=[]
    for _ in range(NRAND):
        pick=[allh[i] for i in rng.choice(len(allh),len(heads),replace=False)]
        rd.append(float((eval_ce(posset,pick)-eval_ce(posset,[])).mean()) if posset is not ps
                  else float((eval_ce(posset,pick)-base).mean()))
    rd=np.array(rd)
    z=float((joint['joint_dCE']-rd.mean())/(rd.std(ddof=1)+1e-9))
    return {'label':label,'n_draws':NRAND,'mean':round(float(rd.mean()),4),
            'std':round(float(rd.std(ddof=1)),4),'max':round(float(rd.max()),4),
            'family_over_random_z':round(z,2),'family_exceeds_all':bool(joint['joint_dCE']>rd.max())}

print("random control (all layers)...", flush=True)
ctrl_all=random_control(range(NL),'6 heads, all 18 layers (as §61)',ps,seed=0)
print("random control (same layer band L5-L14)...", flush=True)
ctrl_band=random_control(LAYER_BAND,'6 heads, L5-L14 band only',ps,seed=1)

# family on RANDOM positions
rps=random_posset(ps['n'],seed=7)
rbase=eval_ce(rps,[])
fam_rand=eval_ce(rps,heads)-rbase
family_on_random_pos={'joint_dCE_on_random_positions':round(float(fam_rand.mean()),4),
                      'SE':round(se(fam_rand),4),
                      'note':'family ablated on a RANDOM valid position set (not its own top union)'}
print(f"  family on random positions: {family_on_random_pos}", flush=True)

RES={'attack':'§61 copy-family redundancy reproduction + control audit',
     'shared_posset_n':ps['n'],'solos':solos,'joint':joint,
     'sum_solo_positive':round(sum_solo,4),'sum_solo_signed':round(sum_solo_signed,4),
     'redundancy_ratio':redundancy,'minimal_subset':minimal,'greedy_curve':curve,
     'random_control_all_layers':ctrl_all,'random_control_same_layer_band':ctrl_band,
     'family_on_random_positions':family_on_random_pos,
     'note_§61':'joint 0.430±0.051, sum-solo 0.111, redundancy 3.86, minimal 4-head 87%, z=24.9'}
json.dump(RES,open(f'{QK}/qk_redteam_attack3.json','w'),indent=2)
print("\n===== ATTACK 3 SUMMARY =====", flush=True)
print(json.dumps(RES,indent=2), flush=True)
print("ATTACK3 DONE", flush=True)
