"""RED-TEAM ATTACK 2 (§63) + ATTACK 4 (§62).

ATTACK 2: h.L8.7 / h.L8.3 "attend-to-DIGIT" causal damage concentrates 11-20x on
digit-source positions. CONFOUND: is the trigger orthographic (digit CHARACTER) or
POSITIONAL (digits cluster in dates/lists at particular line-structure contexts)?
Position-matched control: compare digit-source vs non-digit-source damage WITHIN matched
distance-since-newline bins, and with a stratified dsn-matched resample of non-digit
positions. If the concentration survives position-matching -> orthographic (SURVIVES);
if it vanishes -> positional artifact (BROKEN).

ATTACK 4: §62 negative claim "there is NO distance-to-newline computational head" rests on
corr(per-position dCE, distance-since-newline) ~= 0. This is a NEGATIVE claim; test its
POWER. (a) Positive control: the position-0 sink h.L5.7 — does its damage in fact rise
with distance and does the Pearson-corr metric detect it? (b) Power calibration: inject a
KNOWN monotone distance signal of realistic magnitude and run the exact metric to see the
corr it would assign a genuine distance head. If the metric misses a real monotone-then-
saturating signature, the negative claim is underpowered.

Forward compute lines copied VERBATIM from qk_unsup_positional.py / qk_unsup_bytefrag.py.
HELD FW[448:600], mean-ablation, paired STANDARD ERRORS.
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
held_np=HELD.cpu().numpy()

_special={tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL=np.array(sorted(_special)); is_special_h=np.isin(held_np,SPECIAL)
HAS_DIGIT=np.array([any(c.isdigit() for c in tok.decode([t]).strip()) for t in range(V)])
NEWLINE_BOOL=np.array([('\n' in tok.decode([t])) for t in range(V)])
def dsn_grid(tokens):
    N,T=tokens.shape; nl=NEWLINE_BOOL[tokens]; seg=np.zeros((N,T),np.int64); cur=np.full(N,-1)
    for t in range(T):
        seg[:,t]=np.where(cur>=0,cur,0); cur=np.where(nl[:,t],t,cur)
    return np.arange(T)[None,:]-seg
HE_DSN=dsn_grid(held_np)
posmat=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)

# ============ VERBATIM forward w/ optional single-head ablation + source collection ============
@torch.no_grad()
def forward(idx, ablate=None, collect_src=None):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    src_out={}
    for li in range(NL):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hc=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hc).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hc).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        yh4=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        if collect_src:
            srcpos=pat.abs().argmax(-1)                                  # (B,NH,T)
            for (tli,th) in collect_src:
                if tli==li:
                    src_out[(tli,th)]=torch.gather(idx,1,srcpos[:,th,:]).cpu().numpy()
        if ablate is not None and ablate[0]==li:
            yh4=yh4.clone(); yh4[:,:,ablate[1]]=YHMEAN[li][:,ablate[1]].unsqueeze(0)
        x=x+a.c_proj(yh4.reshape(B,T,-1))
        mo=blk.mlp(F.rms_norm(x,(D,))); x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    return logits, src_out

# collect held YHMEAN (all heads) + source tokens for target heads
TARGETS=[(8,7),(8,3),(5,7)]
YH_SUM={li:torch.zeros(SEQL,NH,HD,device=DEV) for li in range(NL)}
src_held={t:np.zeros((NHELD,SEQL),np.int64) for t in TARGETS}
@torch.no_grad()
def collect():
    for i in range(0,NHELD,BATCH):
        idx=HELD[i:i+BATCH]; b=idx.shape[0]
        # yh means need a plain forward accumulation
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        cos,sin=rope_tables(SEQL,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(SEQL,SEQL,device=DEV,dtype=torch.bool))
        for li in range(NL):
            blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hc=F.rms_norm(x,(D,))
            def qk(l): z=F.rms_norm(l(hc).view(b,SEQL,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
            v=a.c_v(hc).view(b,SEQL,NH,HD)
            if v1 is None: v1=v
            v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
            q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
            s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
            pat=(s1*s2).masked_fill(~mask,0.0)
            yh4=torch.einsum('bhqk,bkhd->bqhd',pat,v)
            YH_SUM[li]+=yh4.sum(0)
            srcpos=pat.abs().argmax(-1)
            for (tli,th) in TARGETS:
                if tli==li: src_held[(tli,th)][i:i+b]=torch.gather(idx,1,srcpos[:,th,:]).cpu().numpy()
            x=x+a.c_proj(yh4.reshape(b,SEQL,-1)); x=x+blk.mlp(F.rms_norm(x,(D,)))
print("collect held means + source tokens...", flush=True)
collect()
YHMEAN={li:YH_SUM[li]/NHELD for li in range(NL)}

@torch.no_grad()
def ce_grid(ablate=None):
    ce=np.full((NHELD,SEQL),np.nan,np.float32)
    for i in range(0,NHELD,BATCH):
        idx=HELD[i:i+BATCH]; b=idx.shape[0]
        lg,_=forward(idx,ablate=ablate); lg=lg.float()
        lp=F.log_softmax(lg[:,:-1],-1); tgt=idx[:,1:]
        nll=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        ce[i:i+b,:-1]=nll.cpu().numpy()
    return ce
def se(x):
    x=np.asarray(x,float); n=len(x)
    return float(x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
def stat(d):
    d=np.asarray(d,float); n=len(d)
    return {'n':int(n),'dCE':round(float(d.mean()),5) if n else None,'SE':round(se(d),5) if n>1 else None}

valid=(posmat>0)&(posmat<SEQL-1)&~is_special_h
nxt_special=np.zeros_like(is_special_h); nxt_special[:,:-1]=is_special_h[:,1:]
valid=valid&~nxt_special

print("base CE grid...", flush=True)
base_ce=ce_grid(None)
DSN_BINS=[(1,1),(2,3),(4,7),(8,15),(16,31),(32,127)]

RESULTS={}
# =========================== ATTACK 2 ===========================
attack2={}
for (li,h) in [(8,7),(8,3)]:
    print(f"attack2 ablate h.L{li}.{h}...", flush=True)
    abl_ce=ce_grid((li,h)); dce=abl_ce-base_ce
    src=src_held[(li,h)]; src_digit=HAS_DIGIT[src]
    vm=valid&np.isfinite(dce)
    dcef=dce[vm]; dsnf=HE_DSN[vm]; sdf=src_digit[vm]
    overall_digit=stat(dcef[sdf]); overall_nondigit=stat(dcef[~sdf])
    ratio_raw=(overall_digit['dCE']/overall_nondigit['dCE']) if overall_nondigit['dCE'] else None
    # within-dsn-bin digit vs nondigit
    by_bin={}
    for lo,hi in DSN_BINS:
        bm=(dsnf>=lo)&(dsnf<=hi)
        dg=stat(dcef[bm&sdf]); ndg=stat(dcef[bm&~sdf])
        r=(dg['dCE']/ndg['dCE']) if (ndg['dCE'] and dg['dCE'] is not None and ndg['n']>=15 and dg['n']>=8) else None
        by_bin[f"{lo}-{hi}"]={'digit':dg,'nondigit':ndg,'ratio':round(r,2) if r else None}
    # stratified dsn-matched resample: match non-digit dsn distribution to digit's
    digit_idx=np.where(sdf)[0]; nondigit_idx=np.where(~sdf)[0]
    rng=np.random.default_rng(0)
    dsn_dig=dsnf[digit_idx]
    # bin-match on the 6 DSN bins
    def binof(x):
        for bi,(lo,hi) in enumerate(DSN_BINS):
            if lo<=x<=hi: return bi
        return len(DSN_BINS)-1
    dig_bins=np.array([binof(x) for x in dsn_dig])
    nd_bins=np.array([binof(x) for x in dsnf[nondigit_idx]])
    matched=[]
    for bi in range(len(DSN_BINS)):
        want=int((dig_bins==bi).sum()); pool=nondigit_idx[nd_bins==bi]
        if want>0 and len(pool)>0:
            matched.append(rng.choice(pool,size=min(want,len(pool)),replace=False))
    matched=np.concatenate(matched) if matched else np.array([],int)
    matched_nondigit=stat(dcef[matched])
    # digit restricted to same bins that had control coverage (fair)
    digit_matched=stat(dcef[digit_idx])
    ratio_matched=(digit_matched['dCE']/matched_nondigit['dCE']) if matched_nondigit['dCE'] else None
    attack2[f"h.L{li}.{h}"]={
        'raw_digit_source':overall_digit,'raw_nondigit_source':overall_nondigit,
        'raw_ratio':round(ratio_raw,2) if ratio_raw else None,
        'within_dsn_bin':by_bin,
        'dsn_matched_control':{'digit_source':digit_matched,
                               'dsn_matched_nondigit':matched_nondigit,
                               'position_matched_ratio':round(ratio_matched,2) if ratio_matched else None},
        'note':'§63 reported digit-source ~0.0316 vs 0.0029 off (~11x) for L8.7; ~4.6x for L8.3'}
RESULTS['attack2_digit_position_matched']=attack2

# =========================== ATTACK 4 ===========================
print("attack4 ablate h.L5.7 (pos-0 sink)...", flush=True)
abl_ce=ce_grid((5,7)); dce=abl_ce-base_ce
vm=valid&np.isfinite(dce); dcef=dce[vm]; dsnf=HE_DSN[vm].astype(float)
# metric as §62 used it: Pearson corr of per-position dCE vs raw dsn
pear=float(np.corrcoef(dcef,dsnf)[0,1])
# better-powered stats
from scipy.stats import spearmanr
spear=float(spearmanr(dcef,dsnf).correlation)
pear_log=float(np.corrcoef(dcef,np.log(dsnf+1))[0,1])
by_bin={}
for lo,hi in DSN_BINS:
    bm=(dsnf>=lo)&(dsnf<=hi); by_bin[f"{lo}-{hi}"]=stat(dcef[bm])
# paired-ish bin contrast dsn=1 vs dsn in [8,15]
b1=dcef[(dsnf>=1)&(dsnf<=1)]; b8=dcef[(dsnf>=8)&(dsnf<=15)]
contrast=float(b8.mean()-b1.mean()); contrast_se=math.sqrt(se(b8)**2+se(b1)**2)
# POWER calibration: inject known monotone distance signals, run the SAME Pearson metric.
# noise sd = residual within-bin sd of the real dCE (realistic per-token noise)
bin_ids=np.digitize(dsnf,[1.5,3.5,7.5,15.5,31.5])
resid=dcef.copy()
for b in np.unique(bin_ids): resid[bin_ids==b]-=dcef[bin_ids==b].mean()
noise_sd=float(resid.std())
rng=np.random.default_rng(1)
power={}
for shape,fn in [('linear',lambda x:x/64.0),('saturating_log',lambda x:np.log(x+1)/np.log(64))]:
    for amp in [0.01,0.02,0.05]:
        # signal spanning ~[0, amp] across the dsn range (matches a real distance head magnitude)
        sig=amp*fn(dsnf)
        syn=sig+rng.normal(0,noise_sd,size=len(dsnf))
        c=float(np.corrcoef(syn,dsnf)[0,1])
        # bin-mean spread the §62 metric also reported
        bm=[syn[(dsnf>=lo)&(dsnf<=hi)].mean() for lo,hi in DSN_BINS]
        power[f"{shape}_amp{amp}"]={'pearson_corr_metric':round(c,3),
                                    'bin_mean_range':round(float(max(bm)-min(bm)),4),
                                    'would_metric_flag(|corr|>0.05)':bool(abs(c)>0.05)}
RESULTS['attack4_distance_power']={
    'h.L5.7_pos0_sink':{
        'pearson_corr_raw_dsn(as_§62)':round(pear,3),
        'spearman_corr':round(spear,3),
        'pearson_corr_log_dsn':round(pear_log,3),
        'dCE_by_dsn_bin':by_bin,
        'contrast_dsn[8-15]_minus_dsn1':{'delta':round(contrast,5),'SE':round(contrast_se,5),
                                         'z':round(contrast/contrast_se,2) if contrast_se else None},
        'note':'§62 reported corr(dCE,dsn)~=0 and called damage UNIFORM across line structure'},
    'power_calibration_injected_signal':{
        'noise_sd_per_token':round(noise_sd,4),
        'note':'inject KNOWN monotone dsn signal of realistic amplitude, run §62 Pearson-corr metric',
        'results':power},
}

json.dump(RESULTS, open(f'{QK}/qk_redteam_attack24.json','w'), indent=2)
print("\n===== ATTACK 2+4 SUMMARY =====", flush=True)
print(json.dumps(RESULTS,indent=2), flush=True)
print("ATTACK24 DONE", flush=True)
