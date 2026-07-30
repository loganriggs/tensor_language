"""RED-TEAM ATTACK 1 (§64): is mlp.L15.d2 "punctuation -> capitalization" a DISTINCT
remap circuit, or is it redundant with the already-known capitalization mechanism
(line/sentence-start positional structure §62, punctuation head h.L13.8 §63, and the
newline->capital remap mlp.L16.d1 §64)?

Independent re-check on HELD FW[448:600], mean-ablation (in-distribution per-position
zero point), paired per-token STANDARD ERROR. Forward compute lines copied VERBATIM from
qk_unsup_decouple.py (which copies qk_unsup_discover.py). mlp directions recomputed
identically (eigh of TRAIN MLP-output gram, top-N_SVD descending).

Two independent breaks:
  (A) JOINT / MARGINAL ablation (the §61 redundancy logic applied to a remap): at
      mlp.L15.d2's active-punctuation positions, measure the capital-prob drop under each
      of {L15.d2}, {L13.8}, {L16.d1}, {L13.8+L16.d1}, {all three}. If L15.d2's MARGINAL
      contribution (joint3 minus {L13.8+L16.d1}) ~= its solo drop and >> 0, it is a
      DISTINCT/additive circuit; if the marginal collapses to ~0, it is redundant.
  (B) CONDITIONING split: does the capital boost survive at MID-SENTENCE punctuation
      (far from a line start) and does it require SENTENCE-ENDING punctuation? If it only
      fires at line-initial contexts, it is the same line-start capitalization §62 already
      attributes to positional structure.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128; N_SVD = 4
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)
HELD  = FINEWEB[448:600, :SEQL].to(DEV)
NHELD = HELD.shape[0]
BATCH = 4
KCAUSAL = 200

# --- lexical classes copied VERBATIM from qk_unsup_decouple.py ---
BRACKETS_OPEN=set("([{<"); BRACKETS_CLOSE=set(")]}>")
QUOTE_OPEN=set("“‘`"); QUOTE_CLOSE=set("”’"); QUOTE_STRAIGHT=set("\"'")
PUNCT=set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS={"and","or","but","nor","yet","so"}
DETERMINERS={"the","a","an","this","that","these","those","some","any","each","every","no","another","such"}
PRONOUNS={"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}
def lex1(s):
    if s=="": return 'other'
    if ('�' in s) or (s==tok.eos_token or '<|endoftext|>' in s): return 'special'
    if '\n' in s: return 'newline'
    body=s.strip(); low=body.lower()
    if body=="": return 'other'
    if all(ch in QUOTE_OPEN for ch in body): return 'quote_open'
    if all(ch in QUOTE_CLOSE for ch in body): return 'quote_close'
    if all(ch in QUOTE_STRAIGHT for ch in body): return 'quote'
    if all(ch in BRACKETS_OPEN for ch in body): return 'bracket_open'
    if all(ch in BRACKETS_CLOSE for ch in body): return 'bracket_close'
    if any(ch.isdigit() for ch in body): return 'digit'
    if all((ch in PUNCT or ch in QUOTE_STRAIGHT or ch in QUOTE_OPEN or ch in QUOTE_CLOSE
            or ch in BRACKETS_OPEN or ch in BRACKETS_CLOSE) for ch in body): return 'punct'
    if low in DETERMINERS: return 'determiner'
    if low in COORDINATORS: return 'coordinator'
    if low in PRONOUNS: return 'pronoun'
    if body[0].isupper(): return 'capital'
    lead=s.startswith(' ')
    if lead and body.isalpha() and len(body)>1: return 'word'
    if (not lead) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'
VOCAB_CLASS = np.array([lex1(tok.decode([t])) for t in range(V)], dtype=object)
CAP_MASK = torch.from_numpy((VOCAB_CLASS=='capital')).to(DEV)          # (V,) bool
SENT_END = set('.!?…')
_special={tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL=np.array(sorted(_special))

# newline token set + distance-since-newline
NEWLINE_BOOL = np.array([('\n' in tok.decode([t])) for t in range(V)])
def dsn_grid(tokens):
    N,T=tokens.shape; nl=NEWLINE_BOOL[tokens]
    seg=np.zeros((N,T),np.int64); cur=np.full(N,-1)
    for t in range(T):
        seg[:,t]=np.where(cur>=0,cur,0)
        cur=np.where(nl[:,t],t,cur)
    return np.arange(T)[None,:]-seg
held_np = HELD.cpu().numpy()
HE_DSN = dsn_grid(held_np)

# ============================ MLP directions (TRAIN gram) ============================
gram=[torch.zeros(D,D,device=DEV) for _ in range(NL)]
@torch.no_grad()
def pass_gram(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
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
        x=x+a.c_proj(yh4.reshape(B,T,-1))
        mo=blk.mlp(F.rms_norm(x,(D,)))
        gram[li]+=torch.einsum('btd,bte->de',mo,mo)
        x=x+mo
print("PASS gram (TRAIN)...", flush=True)
for i in range(0,256,BATCH): pass_gram(TRAIN[i:i+BATCH])
mlp_dirs=torch.zeros(NL,N_SVD,D,device=DEV)
for li in range(NL):
    _,evecs=torch.linalg.eigh(gram[li]); mlp_dirs[li]=evecs[:,-N_SVD:].T.flip(0)
del gram
print("MLP dirs done.", flush=True)

# candidate components
L15d2 = ('mlp',15,2); L16d1 = ('mlp',16,1); L138 = ('head',13,8)

# ============================ generalized forward (list ablation) ============================
# ablate_list entries: ('head',li,h) | ('mlp',li,k). mean-ablation via YHMEAN / PROJMEAN.
@torch.no_grad()
def forward(idx, ablate_list=None, collect=None):
    """collect: dict-of-lists spec. returns logits, coll. coll keys:
       ('yh',li)->sum over batch; ('mproj',li,k)->(B,T); ('hnorm',li,h)->(B,T)."""
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    ab_head={}; ab_mlp={}
    if ablate_list:
        for e in ablate_list:
            if e[0]=='head': ab_head.setdefault(e[1],[]).append(e[2])
            else: ab_mlp.setdefault(e[1],[]).append(e[2])
    coll={}
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
        if collect and ('yh',li) in collect: coll[('yh',li)]=yh4.sum(0)
        if collect:
            Wr=a.c_proj.weight.view(D,NH,HD)
            for (kind,tli,th) in [k for k in collect if k[0]=='hnorm']:
                if tli==li:
                    comp=torch.einsum('btc,oc->bto',yh4[:,:,th],Wr[:,th]); coll[('hnorm',tli,th)]=comp.norm(dim=-1)
        if li in ab_head:
            yh4=yh4.clone()
            for th in ab_head[li]: yh4[:,:,th]=YHMEAN[li][:,th].unsqueeze(0)
        x=x+a.c_proj(yh4.reshape(B,T,-1))
        mo=blk.mlp(F.rms_norm(x,(D,)))
        if collect:
            for (kind,tli,tk) in [k for k in collect if k[0]=='mproj']:
                if tli==li:
                    coll[('mproj',tli,tk)]=torch.einsum('btd,d->bt',mo,mlp_dirs[tli,tk])
        if li in ab_mlp:
            for tk in ab_mlp[li]:
                dvec=mlp_dirs[li,tk]; pr=torch.einsum('btd,d->bt',mo,dvec)
                mo=mo-(pr-PROJMEAN[(li,tk)].unsqueeze(0)).unsqueeze(-1)*dvec
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    return logits, coll

# ---- collect held means + activations ----
print("collect held means + activations...", flush=True)
YH_SUM={li:torch.zeros(SEQL,NH,HD,device=DEV) for li in range(NL)}
PROJ_SUM={(15,2):torch.zeros(SEQL,device=DEV),(16,1):torch.zeros(SEQL,device=DEV)}
act_l15=np.zeros((NHELD,SEQL),np.float32)
collect_spec={('yh',li):1 for li in range(NL)}
collect_spec[('mproj',15,2)]=1; collect_spec[('mproj',16,1)]=1
for i in range(0,NHELD,BATCH):
    idx=HELD[i:i+BATCH]; b=idx.shape[0]
    _,coll=forward(idx, collect=collect_spec)
    for li in range(NL): YH_SUM[li]+=coll[('yh',li)]
    PROJ_SUM[(15,2)]+=coll[('mproj',15,2)].sum(0); PROJ_SUM[(16,1)]+=coll[('mproj',16,1)].sum(0)
    act_l15[i:i+b]=coll[('mproj',15,2)].abs().cpu().numpy()
YHMEAN={li:YH_SUM[li]/NHELD for li in range(NL)}
PROJMEAN={k:PROJ_SUM[k]/NHELD for k in PROJ_SUM}

# ---- position selection: L15.d2-active punctuation positions ----
posmat=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)
held_special=np.isin(held_np,SPECIAL)
valid_next=(posmat>0)&(posmat<SEQL-1)&~held_special
cur_class=VOCAB_CLASS[held_np]
a=act_l15.copy(); a[~valid_next]=-1e30
flat=a.reshape(-1); order=np.argsort(-flat)[:KCAUSAL]
top_s,top_p=order//SEQL, order%SEQL
is_punct_top=(cur_class[top_s,top_p]=='punct')
A_s,A_p=top_s[is_punct_top],top_p[is_punct_top]                     # active punctuation
print(f"active-punct positions n={len(A_s)}", flush=True)
# inactive-punct control (punct positions, path below median activation)
punct_all=(cur_class.reshape(-1)=='punct')&(flat>-1e29)
med=np.median(flat[flat>-1e29])
ctrl_pool=np.where(punct_all&(flat<med))[0]
rng=np.random.default_rng(0)
ctrl=rng.choice(ctrl_pool,size=min(len(A_s),len(ctrl_pool)),replace=False)
C_s,C_p=ctrl//SEQL, ctrl%SEQL

# ---- compute base + per-config grids of P(capital at next) and CE ----
def grids(ablate_list):
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    for i in range(0,NHELD,BATCH):
        idx=HELD[i:i+BATCH]; b=idx.shape[0]
        lg,_=forward(idx, ablate_list=ablate_list); lg=lg.float()
        sm=torch.softmax(lg,-1)
        pc=sm[...,CAP_MASK].sum(-1)                                 # (b,T)
        lp=F.log_softmax(lg[:,:-1],-1); tgt=idx[:,1:]
        nll=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        Pcap[i:i+b]=pc.cpu().numpy(); CE[i:i+b,:-1]=nll.cpu().numpy()
    return Pcap,CE
def se(x):
    x=np.asarray(x,float); n=len(x)
    return float(x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
def paired_drop(base_grid,abl_grid,s,p):
    d=base_grid[s,p]-abl_grid[s,p]                                  # base - ablated (drop>0 = suppressed)
    return float(np.mean(d)), se(d), len(d)
def paired_dce(base_ce,abl_ce,s,p):
    d=abl_ce[s,p]-base_ce[s,p]
    return float(np.mean(d)), se(d), len(d)

print("base grid...", flush=True)
Pcap_base,CE_base=grids(None)
configs={
    'L15d2':[L15d2], 'L138':[L138], 'L16d1':[L16d1],
    'L138+L16d1':[L138,L16d1], 'all3':[L15d2,L138,L16d1],
}
cfg_grids={}
for name,al in configs.items():
    print(f"config {name}...", flush=True)
    cfg_grids[name]=grids(al)

RES={'attack':'§64 mlp.L15.d2 punctuation->capital DISTINCTNESS',
     'n_active_punct':int(len(A_s)),'n_control':int(len(C_s))}

# ---- (0) reproduce §64 headline: L15.d2 capital drop at active-punct vs inactive control ----
Pa,Ca=cfg_grids['L15d2']
drop_active=paired_drop(Pcap_base,Pa,A_s,A_p)
drop_ctrl=paired_drop(Pcap_base,Pa,C_s,C_p)
dce_active=paired_dce(CE_base,Ca,A_s,A_p)
RES['reproduce_headline']={
 'dP_capital_active_punct':[round(drop_active[0],5),round(drop_active[1],5),drop_active[2]],
 'dP_capital_inactive_control':[round(drop_ctrl[0],5),round(drop_ctrl[1],5),drop_ctrl[2]],
 'z_active':round(drop_active[0]/drop_active[1],2),
 'dCE_active_punct':[round(dce_active[0],4),round(dce_active[1],4),dce_active[2]],
 'note':'§64 reported dP=0.0068±0.0009, control 0.0003, z~7.7, dCE +0.0236±0.0083'}

# ---- (A) joint/marginal redundancy at active-punct positions ----
drops={}
for name in configs:
    Pg,_=cfg_grids[name]
    d=paired_drop(Pcap_base,Pg,A_s,A_p); drops[name]=(round(d[0],5),round(d[1],5))
solo=drops['L15d2'][0]; joint3=drops['all3'][0]; base_pair=drops['L138+L16d1'][0]
marginal_L15=joint3-base_pair
sum_solos=drops['L15d2'][0]+drops['L138'][0]+drops['L16d1'][0]
RES['joint_marginal']={
 'capital_drop_by_config':drops,
 'solo_L15d2':round(solo,5),
 'drop_L138+L16d1_only':round(base_pair,5),
 'joint_all3':round(joint3,5),
 'marginal_of_L15d2_added_last':round(marginal_L15,5),
 'sum_of_three_solos':round(sum_solos,5),
 'additivity_ratio_joint_over_sumsolo':round(joint3/sum_solos,2) if abs(sum_solos)>1e-9 else None,
 'interpretation':('DISTINCT/ADDITIVE if marginal_of_L15d2 ~= solo_L15d2 and >>0; '
                   'REDUNDANT if marginal ~= 0')}

# ---- (B) conditioning split of L15.d2 solo capital-drop ----
dsn_A=HE_DSN[A_s,A_p]
d_all=Pcap_base[A_s,A_p]-Pa[A_s,A_p]
def sub(mask):
    if mask.sum()<8: return {'n':int(mask.sum()),'dP':None,'SE':None}
    dd=d_all[mask]; return {'n':int(mask.sum()),'dP':round(float(dd.mean()),5),'SE':round(se(dd),5)}
line_init=dsn_A<=3; mid_line=dsn_A>=8
# sentence-ending vs other punctuation (decode current token)
cur_tok_A=held_np[A_s,A_p]
def is_sent_end(t):
    s=tok.decode([int(t)]).strip(); return len(s)>0 and all(ch in SENT_END for ch in s)
sent_mask=np.array([is_sent_end(t) for t in cur_tok_A])
RES['conditioning']={
 'by_line_position':{'line_initial_dsn<=3':sub(line_init),'mid_line_dsn>=8':sub(mid_line)},
 'by_punct_type':{'sentence_ending(.!?…)':sub(sent_mask),'other_punct':sub(~sent_mask)},
 'dsn_distribution_active_punct':{'median':int(np.median(dsn_A)),'frac_dsn<=3':round(float((dsn_A<=3).mean()),3)},
}

json.dump(RES, open(f'{QK}/qk_redteam_attack1.json','w'), indent=2)
print("\n===== ATTACK 1 SUMMARY =====", flush=True)
print(json.dumps(RES,indent=2), flush=True)
print("ATTACK1 DONE", flush=True)
