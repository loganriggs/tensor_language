"""EDITING / CONTROL demonstration on the unsupervised-discovered CAPITAL SELECTOR
mlp.L17.d1 (§68/§69). The class-level detector + algorithmic arc verified this final-block
feed-forward direction is a GENUINE context-conditioned capital SELECTOR: it pushes the
capital token CLASS ~3.2x harder where a capital is due, and ablation hurts ~11x more there.

Here we close discovery -> verification -> CONTROL. We STEER the direction by scaling its
contribution to the layer-17 residual write:

    new_mo = mo + (alpha - 1) * (proj - projmean) * dir

where `proj` is the (signed) projection of the block-17 MLP output `mo` onto the unit
direction `dir = mlp_dirs[17,1]`, and `projmean` is the held per-position mean projection.
  alpha = 1  -> exactly the model (natural)
  alpha = 0  -> mean-ablated (IDENTICAL to the qk_unsup_classpush / qk_arc_caps ablation)
  alpha > 1  -> amplified deviation from the mean (up-steer)
  alpha < 0  -> reversed deviation (down-steer past ablation)
The parameterisation is sign-invariant in the arbitrary eigenvector sign because it scales
the model's OWN deviation from its mean, and reduces to mean-ablation at alpha=0.

Discipline (Logan spec): DOSE-RESPONSE -> REACH -> COLLATERAL/SPECIFICITY ->
CONTEXT-CONDITIONING RED-TEAM -> PLACEBO (random matched-norm final-block direction).

FORWARD + MLP-direction construction + mean baseline copied VERBATIM from qk_arc_caps.py /
qk_unsup_classpush.py. Capital metric = softmax mass on VOCAB_CLASS=='capital' (arc_caps).
Held-back FW[448:600,:128]. Batch<=8, footprint<4GB, GPU guard.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

# ---------------- GPU GUARD (verbatim from classpush) ----------------
def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 NL={NL} NH={NH} HD={HD} D={D} V={V}", flush=True)

FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128; N_SVD = 4
TRAIN = FINEWEB[0:256, :SEQL].to(DEV)          # only to recompute MLP directions (as discovery)
HELD  = FINEWEB[448:600, :SEQL].to(DEV)        # held-back slice, untouched by discovery
NHELD = HELD.shape[0]
BATCH = 6

TARGET_LI, TARGET_KK = 17, 1                    # mlp.L17.d1 -- the verified capital SELECTOR

# --- lexical classes copied VERBATIM from qk_arc_caps.py / qk_unsup_decouple.py ---
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
CAP_MASK = torch.from_numpy((VOCAB_CLASS=='capital')).to(DEV)          # (V,) bool -- capital output class
SENT_END = set('.!?…')
_special={tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL=np.array(sorted(_special))
NEWLINE_BOOL = np.array([('\n' in tok.decode([t])) for t in range(V)])
def _is_sent_end_tok(t):
    s=tok.decode([int(t)]).strip(); return len(s)>0 and all(ch in SENT_END for ch in s)
SENTEND_BOOL = np.array([_is_sent_end_tok(t) for t in range(V)])
CAP_BOOL     = (VOCAB_CLASS=='capital')
PUNCT_BOOL   = (VOCAB_CLASS=='punct')
WORD_BOOL    = np.isin(VOCAB_CLASS, np.array(['word','subword']))

def dsn_grid(tokens):
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

TARGET_DIR = mlp_dirs[TARGET_LI, TARGET_KK].clone()          # unit-norm capital selector direction
# PLACEBO: random final-block direction of MATCHED (unit) norm
gseed = torch.Generator(device=DEV).manual_seed(1234)
PLACEBO_DIR = torch.randn(D, device=DEV, generator=gseed); PLACEBO_DIR /= PLACEBO_DIR.norm()
print(f"TARGET_DIR norm {TARGET_DIR.norm().item():.4f}  PLACEBO_DIR norm {PLACEBO_DIR.norm().item():.4f}  "
      f"cos(target,placebo)={float(TARGET_DIR@PLACEBO_DIR):.4f}", flush=True)

# ============================ steerable forward (VERBATIM forward + one added term) ============================
@torch.no_grad()
def forward(idx, steer=None, collect_proj=None):
    """steer=(li, dir(D,), projmean(SEQL,), alpha): new_mo = mo + (alpha-1)*(proj-projmean)*dir.
       collect_proj=(li, dirs(ndir,D)) -> returns (logits, per-pos projection sums (SEQL,ndir))."""
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    proj_out=None
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
        if collect_proj is not None and li==collect_proj[0]:
            proj_out = torch.einsum('btd,nd->tn', mo, collect_proj[1])   # summed over batch -> (T,ndir)
        if steer is not None and li==steer[0]:
            dvec=steer[1]; pmean=steer[2]; alpha=steer[3]
            pr=torch.einsum('btd,d->bt', mo, dvec)                       # (B,T) signed projection
            mo=mo + (alpha-1.0)*(pr - pmean.unsqueeze(0)).unsqueeze(-1)*dvec
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    return logits, proj_out

# ---- per-position mean projection (held) for BOTH directions (natural, unsteered) ----
print("collect held per-position mean projections (target + placebo)...", flush=True)
PROBE = torch.stack([TARGET_DIR, PLACEBO_DIR], 0)               # (2,D)
proj_sum = torch.zeros(SEQL, 2, device=DEV)
for i in range(0, NHELD, BATCH):
    _, pj = forward(HELD[i:i+BATCH], collect_proj=(TARGET_LI, PROBE))
    proj_sum += pj
PROJMEAN_TARGET  = proj_sum[:,0] / NHELD                        # (SEQL,)
PROJMEAN_PLACEBO = proj_sum[:,1] / NHELD

# ============================ position-type grids (verbatim from arc_caps) ============================
posmat=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)
held_special=np.isin(held_np,SPECIAL)
cur_class=VOCAB_CLASS[held_np]
nxt=np.zeros_like(held_np); nxt[:,:-1]=held_np[:,1:]
next_special=np.zeros_like(held_special); next_special[:,:-1]=held_special[:,1:]
valid=(posmat>0)&(posmat<SEQL-1)&~held_special&~next_special
cur_sentend=SENTEND_BOOL[held_np]
cur_newline=NEWLINE_BOOL[held_np]
cur_midpunct=PUNCT_BOOL[held_np] & ~cur_sentend & ~cur_newline
cur_word=WORD_BOOL[held_np]
next_cap=CAP_BOOL[nxt]

M_sentend  = valid & cur_sentend
M_newline  = valid & cur_newline
M_boundary = (M_sentend | M_newline)                           # boundary = capital is contextually DUE
M_midword  = valid & cur_word & (HE_DSN>=8)                    # deep mid-sentence content (capital NOT due)
# behavioural due / not-due (next token actually is / is not capital)
M_due      = valid & next_cap                                  # capital DUE (next actually capital)
M_notdue   = valid & ~next_cap                                 # capital NOT due (next not capital)
# boundary intersections for the "where due" amplification test
M_bnd_due  = M_boundary & next_cap                             # boundary AND capital due
M_mid_notdue = M_midword & ~next_cap                           # mid-sentence, capital NOT due (override target)
print(f"counts: valid={valid.sum()} boundary={M_boundary.sum()} midword={M_midword.sum()} "
      f"due={M_due.sum()} notdue={M_notdue.sum()} bnd_due={M_bnd_due.sum()} mid_notdue={M_mid_notdue.sum()}",
      flush=True)

# ============================ grids: P(capital next) and CE under a steer ============================
@torch.no_grad()
def grids(steer):
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    for i in range(0,NHELD,BATCH):
        idx=HELD[i:i+BATCH]; b=idx.shape[0]
        lg,_=forward(idx, steer=steer); lg=lg.float()
        sm=torch.softmax(lg,-1)
        pc=sm[...,CAP_MASK].sum(-1)
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
def paired(a_grid, b_grid, mask):   # a - b at mask, paired per position
    d=(a_grid[mask]-b_grid[mask]); return stat(d)

# ============================ SWEEP ============================
ALPHAS = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

def run_sweep(dirvec, pmean, label):
    print(f"\n===== SWEEP [{label}] alpha in {ALPHAS} =====", flush=True)
    gcache={}
    for al in ALPHAS:
        steer = None if al==1.0 else (TARGET_LI, dirvec, pmean, al)
        # even alpha==1 goes through steer=None which is EXACTLY the model; keep explicit for clarity
        t0=time.time()
        Pg,Cg = grids(None if al==1.0 else (TARGET_LI, dirvec, pmean, al))
        gcache[al]=(Pg,Cg)
        print(f"  alpha={al:>5}: P(cap) valid {stat(Pg[valid])['mean']:.4f} due {stat(Pg[M_due])['mean']:.4f} "
              f"notdue {stat(Pg[M_notdue])['mean']:.4f} boundary {stat(Pg[M_boundary])['mean']:.4f} "
              f"midword {stat(Pg[M_midword])['mean']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return gcache

target_cache  = run_sweep(TARGET_DIR,  PROJMEAN_TARGET,  'TARGET mlp.L17.d1')
placebo_cache = run_sweep(PLACEBO_DIR, PROJMEAN_PLACEBO, 'PLACEBO random matched-norm')

# natural reference grids (alpha=1)
Pnat, Cnat = target_cache[1.0]

def sweep_report(cache):
    """for each alpha: P(capital) at position sets + collateral delta-CE (vs natural) + specificity."""
    rep={}
    for al in ALPHAS:
        Pg,Cg = cache[al]
        rep[str(al)] = {
            'P_capital': {
                'all_valid':   stat(Pg[valid]),
                'due':         stat(Pg[M_due]),
                'notdue':      stat(Pg[M_notdue]),
                'boundary':    stat(Pg[M_boundary]),
                'midword':     stat(Pg[M_midword]),
                'bnd_due':     stat(Pg[M_bnd_due]),
                'mid_notdue':  stat(Pg[M_mid_notdue]),
            },
            # collateral: delta cross-entropy relative to natural (alpha=1), paired per position
            'dCE_vs_natural': {
                'all_valid':   paired(Cg, Cnat, valid),
                'notdue':      paired(Cg, Cnat, M_notdue),      # off-target: next token NOT capital
                'due':         paired(Cg, Cnat, M_due),
                'midword':     paired(Cg, Cnat, M_midword),
            },
            # dP(capital) relative to natural (steering effect on capital mass), paired
            'dP_capital_vs_natural': {
                'due':         paired(Pg, Pnat, M_due),
                'notdue':      paired(Pg, Pnat, M_notdue),
                'boundary':    paired(Pg, Pnat, M_boundary),
                'midword':     paired(Pg, Pnat, M_midword),
                'mid_notdue':  paired(Pg, Pnat, M_mid_notdue),
            },
        }
    return rep

target_report  = sweep_report(target_cache)
placebo_report = sweep_report(placebo_cache)

# ---- dose-response monotonicity (Spearman of alpha vs mean P(capital)) ----
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    if ra.std()==0 or rb.std()==0: return 0.0
    return float(np.corrcoef(ra,rb)[0,1])
def dose_curve(cache, mask):
    return [stat(cache[al][0][mask])['mean'] for al in ALPHAS]

dose = {
    'alphas': ALPHAS,
    'target_P_capital_due':      dose_curve(target_cache, M_due),
    'target_P_capital_notdue':   dose_curve(target_cache, M_notdue),
    'target_P_capital_boundary': dose_curve(target_cache, M_boundary),
    'target_P_capital_midword':  dose_curve(target_cache, M_midword),
    'target_P_capital_valid':    dose_curve(target_cache, valid),
    'placebo_P_capital_due':     dose_curve(placebo_cache, M_due),
    'placebo_P_capital_notdue':  dose_curve(placebo_cache, M_notdue),
    'placebo_P_capital_valid':   dose_curve(placebo_cache, valid),
}
dose['spearman_alpha_vs_Pcap'] = {
    'target_due':      round(spearman(ALPHAS, dose['target_P_capital_due']),4),
    'target_notdue':   round(spearman(ALPHAS, dose['target_P_capital_notdue']),4),
    'target_boundary': round(spearman(ALPHAS, dose['target_P_capital_boundary']),4),
    'target_midword':  round(spearman(ALPHAS, dose['target_P_capital_midword']),4),
    'target_valid':    round(spearman(ALPHAS, dose['target_P_capital_valid']),4),
    'placebo_due':     round(spearman(ALPHAS, dose['placebo_P_capital_due']),4),
    'placebo_valid':   round(spearman(ALPHAS, dose['placebo_P_capital_valid']),4),
}

# ---- REACH: usable range of P(capital) at DUE and at NOTDUE across the sweep ----
def reach(cache):
    due=[stat(cache[al][0][M_due])['mean'] for al in ALPHAS]
    notdue=[stat(cache[al][0][M_notdue])['mean'] for al in ALPHAS]
    return {
        'due_min': round(min(due),5), 'due_max': round(max(due),5),
        'due_at_alpha_min': ALPHAS[int(np.argmin(due))], 'due_at_alpha_max': ALPHAS[int(np.argmax(due))],
        'due_swing': round(max(due)-min(due),5),
        'notdue_min': round(min(notdue),5), 'notdue_max': round(max(notdue),5),
        'notdue_swing': round(max(notdue)-min(notdue),5),
        'natural_due': round(stat(cache[1.0][0][M_due])['mean'],5),
        'natural_notdue': round(stat(cache[1.0][0][M_notdue])['mean'],5),
    }
REACH_TARGET = reach(target_cache)
REACH_PLACEBO = reach(placebo_cache)

# ---- CONTEXT-CONDITIONING RED-TEAM verdict ----
# Under up-steering, does the capital push at NOT-DUE (mid-sentence, lowercase due) grow like at DUE
# (override) or stay flat (conditioning holds)? Compare dP(capital) ratios at the strongest up-alpha.
UP = 8.0
dP_due_up    = target_report[str(UP)]['dP_capital_vs_natural']['boundary']['mean']
dP_notdue_up = target_report[str(UP)]['dP_capital_vs_natural']['mid_notdue']['mean']
override_ratio = round(dP_notdue_up / dP_due_up, 3) if dP_due_up and abs(dP_due_up) > 1e-9 else None
redteam = {
    'up_alpha': UP,
    'dP_capital_at_boundary(due)':    target_report[str(UP)]['dP_capital_vs_natural']['boundary'],
    'dP_capital_at_mid_notdue':       target_report[str(UP)]['dP_capital_vs_natural']['mid_notdue'],
    'dP_capital_at_midword':          target_report[str(UP)]['dP_capital_vs_natural']['midword'],
    'notdue_over_due_dP_ratio': override_ratio,
    'note': ('If notdue/due dP ratio << 1 the context-conditioning HOLDS under up-steering (only amplifies '
             'at genuine boundaries; selection logic upstream). If ~1 or growing, up-steering OVERRIDES the '
             'conditioning (controllable Title-Case override; selection logic IN the direction).'),
}

# ---- SPECIFICITY: capital effect at DUE vs off-target CE cost at NOTDUE, per alpha ----
specificity = {}
for al in ALPHAS:
    dP = target_report[str(al)]['dP_capital_vs_natural']['due']['mean']
    dce_off = target_report[str(al)]['dCE_vs_natural']['notdue']['mean']
    specificity[str(al)] = {
        'dP_capital_due': dP,
        'dCE_offtarget_notdue': dce_off,
        'dCE_all': target_report[str(al)]['dCE_vs_natural']['all_valid']['mean'],
        'specificity_ratio_capitalgain_per_offtarget_CE':
            round(abs(dP)/abs(dce_off),3) if dce_off and abs(dce_off)>1e-6 else None,
    }

OUT = {
 'meta':{'model':'bilin18','tokenizer':'gpt2 (model exposes none) -- FLAGGED',
         'held_slice':'FW[448:600,:128]','train_slice_for_mlp_dirs':'FW[0:256,:128]',
         'target':'mlp.L17.d1 (verified context-conditioned CAPITAL SELECTOR, §68/§69)',
         'target_dir_index':[TARGET_LI,TARGET_KK],
         'steer_formula':'new_mo = mo + (alpha-1)*(proj-projmean)*dir ; alpha=1 natural, alpha=0 mean-ablated',
         'placebo':'random matched (unit) norm direction at layer 17, seed 1234',
         'P_capital':'softmax mass on VOCAB_CLASS==capital (arc_caps metric)',
         'dCE_vs_natural':'cross-entropy(alpha) - cross-entropy(natural alpha=1), paired per position (>0 worse)',
         'dP_capital_vs_natural':'P_capital(alpha) - P_capital(natural), paired per position',
         'batch':BATCH,'alphas':ALPHAS,
         'position_sets':{'due':'valid & next-token-is-capital','notdue':'valid & next-token-not-capital',
                          'boundary':'valid & (sentence-end punct | newline) = capital contextually DUE',
                          'midword':'valid & content word & dsn>=8 = capital NOT due',
                          'bnd_due':'boundary & next capital','mid_notdue':'midword & next not capital'}},
 'counts':{'valid':int(valid.sum()),'boundary':int(M_boundary.sum()),'midword':int(M_midword.sum()),
           'due':int(M_due.sum()),'notdue':int(M_notdue.sum()),'bnd_due':int(M_bnd_due.sum()),
           'mid_notdue':int(M_mid_notdue.sum())},
 'dose_response':dose,
 'reach':{'target':REACH_TARGET,'placebo':REACH_PLACEBO},
 'specificity':specificity,
 'context_conditioning_redteam':redteam,
 'target_sweep_full':target_report,
 'placebo_sweep_full':placebo_report,
}
json.dump(OUT, open(f'{QK}/qk_edit_capselector.json','w'), indent=2)
print("\n===== DOSE-RESPONSE (P capital vs alpha) =====", flush=True)
print("  alpha:      ", [f"{a:>6}" for a in ALPHAS], flush=True)
print("  due:        ", [f"{v:6.3f}" for v in dose['target_P_capital_due']], flush=True)
print("  notdue:     ", [f"{v:6.3f}" for v in dose['target_P_capital_notdue']], flush=True)
print("  boundary:   ", [f"{v:6.3f}" for v in dose['target_P_capital_boundary']], flush=True)
print("  midword:    ", [f"{v:6.3f}" for v in dose['target_P_capital_midword']], flush=True)
print("  placebo due:", [f"{v:6.3f}" for v in dose['placebo_P_capital_due']], flush=True)
print("  Spearman:", dose['spearman_alpha_vs_Pcap'], flush=True)
print("\n===== REACH target:", REACH_TARGET, flush=True)
print("===== RED-TEAM (up alpha=%.0f): dP boundary(due)=%s  dP mid_notdue=%s  ratio=%s" % (
      UP, redteam['dP_capital_at_boundary(due)']['mean'], redteam['dP_capital_at_mid_notdue']['mean'],
      override_ratio), flush=True)
print("\nSaved qk_edit_capselector.json", flush=True)
print("QK EDIT CAPSELECTOR DONE", flush=True)
