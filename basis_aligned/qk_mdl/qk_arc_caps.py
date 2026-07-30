"""OPTION-2 ALGORITHMIC ARC on the CAPITALIZATION circuit (§63/§64 unsupervised discovery).

Circuit components (unsupervised-discovered, red-team-confirmed):
  mlp.L15.d2 : fires on PUNCTUATION -> boosts capital (strongest remap, §64; concentrated mid-sentence)
  mlp.L16.d1 : fires on NEWLINE     -> boosts capital (§64)
  h.L13.8    : PUNCTUATION head (fires on word -> boosts punct, §63)  [emits the boundary MARKER]

Hypothesis: together they implement "capitalize the next word at a sentence/line boundary."
This arc tries to ESTABLISH or REFUTE that as a genuine minimal algorithm, with the discipline:
  1. VERIFY behaviour exists (baseline P(capital next) by boundary type + empirical prior).
  2. PATCH to minimal: mean-ablate all 7 non-empty subsets at boundary positions; dP(capital) and
     delta-cross-entropy with paired standard errors; find minimal subset + division of labour.
  3. RED-TEAM generalisation: does ablating the minimal set break capitalize-AFTER-BOUNDARY while
     LEAVING proper-noun / mid-sentence capitalization intact (genuine boundary algorithm), or kill
     all capitalization equally (generic capital-booster)? Plus the frequency-prior confound.
  4. CONTENT vs POSITION: does the circuit read the boundary via token IDENTITY (punct/newline) or
     via POSITION (fixed offset)? (connect to §62).

FORWARD lines copied VERBATIM from qk_redteam_toolbox.py (which copies qk_unsup_decouple.py /
qk_unsup_discover.py): bilin18 two-branch pattern (q1k1/HD)(q2k2/HD) masked UNNORMALISED, per-head
QK rms_norm THEN RoPE, v-lerp via a.lamb (block-0 v cache), 30*tanh logits. MLP directions =
eigh of TRAIN MLP-output gram, top-N_SVD descending. Do NOT rewrite the forward.

Discovery/verify all on held-back FW[448:600]. Batch<=8, footprint<4GB.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, math, itertools
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
tok = AutoTokenizer.from_pretrained('gpt2')   # model exposes none -- FLAGGED (matches §56/§63/§64)
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128; N_SVD = 4
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)          # only to recompute MLP directions (as discovery)
HELD  = FINEWEB[448:600, :SEQL].to(DEV)        # held-back slice, untouched by discovery
NHELD = HELD.shape[0]
BATCH = 4                                       # keeps softmax-over-V footprint <4GB
KCAUSAL = 200

# --- lexical classes copied VERBATIM from qk_unsup_decouple.py / qk_redteam_toolbox.py ---
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
CAP_MASK = torch.from_numpy((VOCAB_CLASS=='capital')).to(DEV)          # (V,) bool -- model "capital class" output
SENT_END = set('.!?…')
_special={tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL=np.array(sorted(_special))

# vocab-level orthographic masks
NEWLINE_BOOL = np.array([('\n' in tok.decode([t])) for t in range(V)])
def _is_sent_end_tok(t):
    s=tok.decode([int(t)]).strip(); return len(s)>0 and all(ch in SENT_END for ch in s)
SENTEND_BOOL = np.array([_is_sent_end_tok(t) for t in range(V)])
CAP_BOOL     = (VOCAB_CLASS=='capital')
PUNCT_BOOL   = (VOCAB_CLASS=='punct')
WORD_BOOL    = np.isin(VOCAB_CLASS, np.array(['word','subword']))

def dsn_grid(tokens):
    """distance-since-newline: positions elapsed since the last newline token."""
    N,T=tokens.shape; nl=NEWLINE_BOOL[tokens]
    seg=np.zeros((N,T),np.int64); cur=np.full(N,-1)
    for t in range(T):
        seg[:,t]=np.where(cur>=0,cur,0)
        cur=np.where(nl[:,t],t,cur)
    return np.arange(T)[None,:]-seg
held_np = HELD.cpu().numpy()
HE_DSN  = dsn_grid(held_np)

# ============================ MLP directions (TRAIN gram) -- VERBATIM ============================
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
COMPS = {'L15d2':('mlp',15,2), 'L16d1':('mlp',16,1), 'L138':('head',13,8)}

# ============================ generalized forward (list ablation) -- VERBATIM ============================
@torch.no_grad()
def forward(idx, ablate_list=None, collect=None):
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

# ---- collect held per-position means (for mean-ablation) + component activations ----
print("collect held means + activations...", flush=True)
YH_SUM={li:torch.zeros(SEQL,NH,HD,device=DEV) for li in range(NL)}
PROJ_SUM={(15,2):torch.zeros(SEQL,device=DEV),(16,1):torch.zeros(SEQL,device=DEV)}
act_l15=np.zeros((NHELD,SEQL),np.float32)     # signed mproj L15.d2
act_l16=np.zeros((NHELD,SEQL),np.float32)     # signed mproj L16.d1
act_h138=np.zeros((NHELD,SEQL),np.float32)    # comp-norm h.L13.8
collect_spec={('yh',li):1 for li in range(NL)}
collect_spec[('mproj',15,2)]=1; collect_spec[('mproj',16,1)]=1; collect_spec[('hnorm',13,8)]=1
for i in range(0,NHELD,BATCH):
    idx=HELD[i:i+BATCH]; b=idx.shape[0]
    _,coll=forward(idx, collect=collect_spec)
    for li in range(NL): YH_SUM[li]+=coll[('yh',li)]
    PROJ_SUM[(15,2)]+=coll[('mproj',15,2)].sum(0); PROJ_SUM[(16,1)]+=coll[('mproj',16,1)].sum(0)
    act_l15[i:i+b]=coll[('mproj',15,2)].cpu().numpy()
    act_l16[i:i+b]=coll[('mproj',16,1)].cpu().numpy()
    act_h138[i:i+b]=coll[('hnorm',13,8)].cpu().numpy()
YHMEAN={li:YH_SUM[li]/NHELD for li in range(NL)}
PROJMEAN={k:PROJ_SUM[k]/NHELD for k in PROJ_SUM}

# ============================ grids for P(capital next) and CE under any ablation set ============================
@torch.no_grad()
def grids(ablate_list):
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    for i in range(0,NHELD,BATCH):
        idx=HELD[i:i+BATCH]; b=idx.shape[0]
        lg,_=forward(idx, ablate_list=ablate_list); lg=lg.float()
        sm=torch.softmax(lg,-1)
        pc=sm[...,CAP_MASK].sum(-1)                              # P(model puts on capital class) (b,T)
        lp=F.log_softmax(lg[:,:-1],-1); tgt=idx[:,1:]
        nll=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        Pcap[i:i+b]=pc.cpu().numpy(); CE[i:i+b,:-1]=nll.cpu().numpy()
        del lg,sm,pc,lp,nll
    return Pcap,CE

def se(x):
    x=np.asarray(x,float); n=len(x)
    return float(x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
def stat(vals):
    vals=np.asarray(vals,float)
    return {'mean':round(float(vals.mean()),5) if len(vals) else None,
            'SE':round(se(vals),5) if len(vals)>1 else None, 'n':int(len(vals))}

# ============================ position-type grids over held slice ============================
posmat=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)
held_special=np.isin(held_np,SPECIAL)
cur_class=VOCAB_CLASS[held_np]
nxt = np.zeros_like(held_np); nxt[:,:-1]=held_np[:,1:]
next_special = np.zeros_like(held_special); next_special[:,:-1]=held_special[:,1:]
valid = (posmat>0)&(posmat<SEQL-1)&~held_special&~next_special

# current-token boundary flags
cur_sentend = SENTEND_BOOL[held_np]                 # . ! ? …
cur_newline = NEWLINE_BOOL[held_np]                 # any token containing newline
cur_midpunct= PUNCT_BOOL[held_np] & ~cur_sentend & ~cur_newline   # , ; : - etc (within-sentence)
cur_word    = WORD_BOOL[held_np]                    # content word / subword
# next-token flags
next_cap    = CAP_BOOL[nxt]

M_sentend = valid & cur_sentend
M_newline = valid & cur_newline
M_midpunct= valid & cur_midpunct
M_midword = valid & cur_word & (HE_DSN>=8)          # deep mid-sentence content position
M_boundary= (M_sentend | M_newline)

print(f"position-type counts: sentend={M_sentend.sum()} newline={M_newline.sum()} "
      f"midpunct={M_midpunct.sum()} midword={M_midword.sum()} boundary={M_boundary.sum()}", flush=True)

# ============================ STEP 1: VERIFY BEHAVIOUR ============================
print("STEP 1: baseline behaviour...", flush=True)
Pcap_base, CE_base = grids(None)
def base_behaviour(mask, label):
    return {'label':label,
            'model_P_capital_next':stat(Pcap_base[mask]),
            'empirical_freq_next_is_capital':stat(next_cap[mask].astype(float))}
STEP1 = {
    'after_sentence_punct(.!?…)': base_behaviour(M_sentend, 'after_sentence_punct'),
    'after_newline':              base_behaviour(M_newline, 'after_newline'),
    'mid_sentence_punct(,;:)':    base_behaviour(M_midpunct,'mid_sentence_punct'),
    'mid_sentence_word_control':  base_behaviour(M_midword, 'mid_sentence_word'),
    'all_valid_positions':        base_behaviour(valid,     'all_valid'),
}
print(json.dumps(STEP1, indent=2), flush=True)

# ============================ STEP 2: PATCH TO MINIMAL (7 subsets) ============================
print("STEP 2: 7-subset ablation grids...", flush=True)
keys=list(COMPS.keys())
subsets=[]
for r in range(1,len(keys)+1):
    for c in itertools.combinations(keys,r): subsets.append(list(c))
sub_grids={}
for sub in subsets:
    al=[COMPS[k] for k in sub]
    name='+'.join(sub)
    print(f"  ablate {name}...", flush=True)
    sub_grids[name]=grids(al)

def drop_stats(base_g, abl_g, mask):
    d = base_g[mask]-abl_g[mask]                    # base - ablated (drop>0 = capital suppressed)
    return stat(d)
def dce_stats(abl_ce, mask):
    d = abl_ce[mask]-CE_base[mask]                  # ablated - base (>0 = model got worse)
    return stat(d)

def all_subset_drops(mask):
    out={}
    for sub in subsets:
        name='+'.join(sub); Pg,Cg=sub_grids[name]
        out[name]={'dP_capital':drop_stats(Pcap_base,Pg,mask),'dCE':dce_stats(Cg,mask)}
    return out

STEP2 = {
    'at_boundary(sentence|newline)': all_subset_drops(M_boundary),
    'at_sentence_punct':             all_subset_drops(M_sentend),
    'at_newline':                    all_subset_drops(M_newline),
    'at_mid_sentence_punct':         all_subset_drops(M_midpunct),
}
# division of labour summary at boundary
def solo(name,mask): return drop_stats(Pcap_base,sub_grids[name][0],mask)['mean']
STEP2_summary={
 'boundary_solo_L15d2':solo('L15d2',M_boundary),
 'boundary_solo_L16d1':solo('L16d1',M_boundary),
 'boundary_solo_L138' :solo('L138', M_boundary),
 'boundary_all3':solo('L15d2+L16d1+L138',M_boundary),
 'boundary_pair_L15d2+L16d1':solo('L15d2+L16d1',M_boundary),
 'sentpunct_solo_L15d2':solo('L15d2',M_sentend),
 'sentpunct_solo_L16d1':solo('L16d1',M_sentend),
 'sentpunct_solo_L138' :solo('L138', M_sentend),
 'newline_solo_L15d2':solo('L15d2',M_newline),
 'newline_solo_L16d1':solo('L16d1',M_newline),
 'newline_solo_L138' :solo('L138', M_newline),
}
print("STEP2 summary:", json.dumps(STEP2_summary,indent=2), flush=True)

# ============================ STEP 3: RED-TEAM GENERALIZATION ============================
# minimal capital-carrying set = the two capital-boosters {L15d2,L16d1}; also test all3.
print("STEP 3: red-team generalization...", flush=True)
MIN_NAME='L15d2+L16d1'; ALL_NAME='L15d2+L16d1+L138'
Pg_min = sub_grids[MIN_NAME][0]; Pg_all = sub_grids[ALL_NAME][0]

# position sets: where a capital SHOULD come, split boundary vs mid-sentence(proper-noun)
M_bcap_sent  = M_sentend & next_cap                     # boundary (sentence) & next actually capital
M_bcap_nl    = M_newline & next_cap                     # boundary (newline)  & next actually capital
M_bcap       = M_boundary & next_cap
M_propernoun = valid & cur_word & (HE_DSN>=6) & next_cap & ~cur_sentend & ~cur_newline  # mid-sentence proper noun
M_anycap     = valid & next_cap                         # any capital-next position (generic reference)

def redteam_block(Pg,label):
    return {'config':label,
            'boundary_next_capital':      drop_stats(Pcap_base,Pg,M_bcap),
            'sentence_boundary_next_cap': drop_stats(Pcap_base,Pg,M_bcap_sent),
            'newline_boundary_next_cap':  drop_stats(Pcap_base,Pg,M_bcap_nl),
            'mid_sentence_propernoun':    drop_stats(Pcap_base,Pg,M_propernoun),
            'any_capital_next':           drop_stats(Pcap_base,Pg,M_anycap)}
STEP3 = {
    'counts':{'boundary_next_cap':int(M_bcap.sum()),'propernoun_next_cap':int(M_propernoun.sum()),
              'any_next_cap':int(M_anycap.sum())},
    'ablate_minimal(L15d2+L16d1)': redteam_block(Pg_min,'L15d2+L16d1'),
    'ablate_all3':                 redteam_block(Pg_all,'all3'),
}
# specificity ratio: boundary drop vs proper-noun drop under minimal set
b=STEP3['ablate_minimal(L15d2+L16d1)']['boundary_next_capital']['mean']
p=STEP3['ablate_minimal(L15d2+L16d1)']['mid_sentence_propernoun']['mean']
STEP3['boundary_over_propernoun_ratio_minimal']=round(b/p,2) if p and abs(p)>1e-9 else None

# --- frequency-prior confound ---
# how much of the boundary capitalization is "just" a bigram prior the circuit rides on?
# compare model P(capital) & empirical next-capital freq: sentence-ending vs within-sentence punct.
STEP3['frequency_prior'] = {
    'empirical_next_cap_freq':{
        'after_sentence_punct':stat(next_cap[M_sentend].astype(float)),
        'after_mid_sentence_punct':stat(next_cap[M_midpunct].astype(float)),
        'after_newline':stat(next_cap[M_newline].astype(float)),
        'mid_sentence_word':stat(next_cap[M_midword].astype(float)),
        'overall_base_rate':stat(next_cap[valid].astype(float)),
    },
    'circuit_capital_drop_tracks_prior':{
        # does L15.d2 drop scale with sentence-ending vs within-sentence punct? (a bigram-prior circuit
        # would fire on the specific ".<space>->Capital" bigram, i.e. much stronger at sentence-enders)
        'L15d2_drop_at_sentence_punct':solo('L15d2',M_sentend),
        'L15d2_drop_at_mid_sentence_punct':solo('L15d2',M_midpunct),
    },
    'note':('If the boundary capital drop is present at BOTH sentence-ending and within-sentence punct, '
            'the circuit is not a pure ".<space>->Capital" bigram; if it collapses to only sentence-enders '
            'it is closer to a learned boundary bigram. Empirical freq columns give the prior the circuit rides on.'),
}
print("STEP3:", json.dumps(STEP3, indent=2), flush=True)

# ============================ STEP 4: CONTENT vs POSITION ============================
# Does activation key on token IDENTITY (content) or on POSITION (fixed offset / dsn)?
print("STEP 4: content vs position...", flush=True)
vflat=valid.reshape(-1)
def act_by_class(act):
    a=act.reshape(-1); cc=cur_class.reshape(-1)
    out={}
    for c in ['punct','newline','word','subword','capital','digit','determiner','pronoun','quote','other']:
        mm=vflat&(cc==c)
        if mm.sum()>=20: out[c]=stat(np.abs(a[mm]))
    return out
def act_by_dsn_within(act, curbool):
    """within a fixed current-token class (curbool grid), mean |act| by distance-since-newline bucket.
    If flat across dsn -> content-driven (position doesn't matter given the token); if it ramps -> positional."""
    a=act; out={}
    for lab,lo,hi in [('dsn1-1',1,1),('dsn2-3',2,3),('dsn4-7',4,7),('dsn8-15',8,15),('dsn16+',16,999)]:
        mm=valid&curbool&(HE_DSN>=lo)&(HE_DSN<=hi)
        if mm.sum()>=20: out[lab]=stat(np.abs(a[mm]))
    return out
STEP4 = {
  'L15d2_abs_activation_by_current_class':act_by_class(act_l15),
  'L16d1_abs_activation_by_current_class':act_by_class(act_l16),
  'h138_activation_by_current_class':act_by_class(act_h138),
  'L15d2_within_punct_by_dsn':act_by_dsn_within(act_l15, cur_midpunct|cur_sentend),
  'L16d1_within_word_by_dsn': act_by_dsn_within(act_l16, cur_word),
  'note':('Content-driven if abs-activation concentrates on the trigger token class (punct for L15.d2, '
          'newline for L16.d1) AND is roughly flat across distance-since-newline WITHIN that class; '
          'position-driven if it ramps with dsn independent of token identity (connects to §62 '
          'fixed-offset vs content).'),
}
print("STEP4:", json.dumps(STEP4, indent=2), flush=True)

# ============================ SAVE ============================
OUT = {
 'meta':{'model':'bilin18','tokenizer':'gpt2 (model exposes none) -- FLAGGED',
         'held_slice':'FW[448:600,:128]','train_slice_for_mlp_dirs':'FW[0:256,:128]',
         'components':{k:list(v) for k,v in COMPS.items()},
         'metric_P_capital':'sum of softmax mass on VOCAB_CLASS==capital (proper-noun-ish leading-cap tokens; '
                            '"I" is class pronoun, acronyms are class capital)',
         'dP_capital':'base - ablated (positive = ablation suppressed capitalization)',
         'dCE':'ablated - base cross-entropy (positive = ablation hurt next-token prediction)',
         'ablation':'mean-ablation with held per-position mean (heads: yh mean; mlp: project-out to per-position projmean)',
         'batch':BATCH,'paired_SE':'per-position paired standard error'},
 'step1_verify_behaviour':STEP1,
 'step2_minimal_subset':{'per_subset':STEP2,'summary':STEP2_summary},
 'step3_redteam_generalization':STEP3,
 'step4_content_vs_position':STEP4,
}
json.dump(OUT, open(f'{QK}/qk_arc_caps.json','w'), indent=2)
print("\nSaved qk_arc_caps.json", flush=True)
print("QK ARC CAPS DONE", flush=True)
