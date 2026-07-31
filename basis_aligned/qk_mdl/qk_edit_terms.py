"""TERM-TARGETED EDITING on the section-91 DIFFERENTIAL PAIR -- the fold-audit's editing
upgrade and the direct rematch of section 75.

Background (RESULTS section 91): bilin18's layer-17 feed-forward is a differential pair --
the mlp-recent^2 term (MrxMr) writes a broad generic capital/word PRIOR and the
attention-earlier x mlp-recent term (AexMr) writes its context-GATED near-negation; the
layer's function is their difference. Section 75 steered the layer's top SVD direction
(mlp.L17.d1): a calibrated capitalization dial, but the conditioning "lives upstream" and
could not be surgically overridden. The term decomposition makes the GATED arm (AexMr)
directly addressable. QUESTION: does term-level steering give cleaner, more surgical dials
than direction-level steering?

INTERVENTION: scale a term's deviation from its held per-position mean,
    term' = TMEAN_k + alpha * (term_k - TMEAN_k),
and reassemble the layer output from the modified term + all other terms exactly:
    mo17 -> MEANF + sum_k c_k * (term_k - TMEAN_k),   c_k = alpha for targeted terms, 1 else.
alpha=1 -> the reassembled model (recon-gated ~7e-7); alpha=0 on a term -> the section-91
drop-term ablation (known-answer gates below); alpha>1 amplify; alpha<0 reverse.

DIALS (each swept over the SAME alpha grid as section 75 for side-by-side):
  1. gated_AexMr    -- the context-conditioned arm (the section-75 rematch dial)
  2. prior_MrxMr    -- prior-strength dial
  3. pair_coherent  -- BOTH arms' deviations scaled together (preserving their ratio).
     NOTE (exact algebra): because the layer output is LINEAR in the 15 terms, scaling both
     deviations by alpha changes mo17 by (alpha-1)*(devA+devB) -- i.e. it scales exactly the
     pair's NET functional write (the "difference"/contrast in section-91 language), while any
     edit of the cancelling common mode with the net held fixed is provably a NO-OP at the
     output. The pair has exactly ONE output-level degree of freedom: pair-coherent scaling
     IS the sharpening knob. (Numerically verified below: explicit net-write addition equals
     coherent scaling to float precision.)
  4. placebo_ExAr   -- embedding x attention-recent, centered energy share 0.0001: no
     capitalization dial expected.

MACHINERY VERBATIM: 5-group pair_terms polarization + coarse-group-accumulator forward +
per-position term means + reassembly harness from qk_L17_mixer.py / qk_allterm_census.py;
capital-class metric, lexical classes, position splits (capital-DUE / NOT-DUE / boundary /
midword), paired grids, dose-response/reach/specificity/red-team template from
qk_arc_caps.py / qk_edit_capselector.py.
GATES: recon < 1e-4; TMEAN consistent with qk_L17_mixer_means.pt; reassembly-at-alpha=1
census-convention delta-CE ~ 0; gated alpha=0 reproduces drop_AexMr +0.1479; prior alpha=0
reproduces drop_MrxMr +0.0697; pair alpha=0 reproduces drop_both +0.0587 (qk_L17_mixer.json).
Held-back FW[448:600,:128], paired standard errors, batch 6, <4GB, GPU guard.
Output: qk_edit_terms.json (section-75 reference numbers embedded for side-by-side).
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
OUT = f'{QK}/qk_edit_terms.json'

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
tok = AutoTokenizer.from_pretrained('gpt2')   # model exposes none -- FLAGGED (matches prior sections)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
HELD = FW[448:600, :SEQL].to(DEV); B0 = 6
S_, T_ = HELD.shape
NHELD = S_
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)

# ---- the five coarse groups and the 15 group-pair terms (VERBATIM qk_allterm_census.py) ----
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)   # 15
LI = 17
IA = PNAMES.index('AexMr')   # GATED arm (context-conditioned), share 0.349
IB = PNAMES.index('MrxMr')   # PRIOR arm (mlp-recent^2), share 0.224
IP = PNAMES.index('ExAr')    # placebo term, centered energy share 0.0001

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
W = mlp_wts(LI)

def pair_terms(groups, xpre, Lw, Rw, Dw):
    """15 interaction terms (list of (B,T,D)), sharing the common 1/rho^2 gauge; sum+bias == mo_L.
    VERBATIM construction from qk_hub_streampairs.pair_terms, generalized to 5 groups."""
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, mode=None, coeffs=None, TMEAN=None, MEANF=None, stats=None, extra=None):
    """Forward verbatim from qk_allterm_census.py / qk_L17_mixer.py (coarse-group stream
    accumulators to LI=17); RETURNS LOGITS (qk_L17_mixer_2.py style).
    mode: None (full model) | 'collect' (term means + recon gate)
        | 'steer' (mo17 -> MEANF + sum_k coeffs.get(kk,1)*(term_kk - TMEAN[kk])).
    extra=(kidxs, scale): additionally add scale * sum_{kk in kidxs}(term_kk - TMEAN[kk])
    (the explicit 'difference knob' -- used only for the pair identity verification)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LI:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if track and li == LI:
            groups = [cE*x0, SA, aout, SM, MR]     # E, Ae, Ar, Me, Mr; x is x_pre here
            terms = pair_terms(groups, x, W[0], W[1], W[2])
            if mode == 'collect':
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'steer':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in range(NT):
                    new = new + coeffs.get(kk, 1.0) * (terms[kk] - TMEAN[kk])
                if extra is not None:
                    for kk in extra[0]:
                        new = new + extra[1] * (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            del terms, groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# ---------------- PASS 1: term means + decomposition gate (VERBATIM) ----------------
print("PASS 1: term means + recon gate ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MEANF = TMEAN.sum(0) + W[3]
gate_fro = (st['fro_num']/st['fro_den'])**0.5
print(f"L17 GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e}", flush=True)
assert gate_fro < 1e-4, "decomposition gate FAILED"
# consistency with the section-91 saved means
saved = torch.load(f'{QK}/qk_L17_mixer_means.pt', map_location='cpu')
assert saved['PNAMES'] == PNAMES and saved['IA'] == IA and saved['IB'] == IB
tm_diff = float((saved['TMEAN'] - TMEAN.cpu()).abs().max())
print(f"TMEAN consistency vs qk_L17_mixer_means.pt: max abs diff {tm_diff:.2e}", flush=True)
assert tm_diff < 1e-3, "TMEAN consistency gate FAILED"

# --- lexical classes copied VERBATIM from qk_arc_caps.py / qk_edit_capselector.py ---
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
CAP_MASK = torch.from_numpy((VOCAB_CLASS=='capital')).to(DEV)
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

# ---------------- position-type grids (VERBATIM qk_edit_capselector.py) ----------------
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
next_class=VOCAB_CLASS[nxt]

M_sentend  = valid & cur_sentend
M_newline  = valid & cur_newline
M_boundary = (M_sentend | M_newline)                           # boundary = capital contextually DUE
M_midword  = valid & cur_word & (HE_DSN>=8)                    # deep mid-sentence content (capital NOT due)
M_due      = valid & next_cap                                  # capital DUE (next actually capital)
M_notdue   = valid & ~next_cap                                 # capital NOT due
M_bnd_due  = M_boundary & next_cap
M_mid_notdue = M_midword & ~next_cap
# structural next-token position sets (section 91: the pair's damage lands here)
STRUCT = {'next_bracket_open': valid & (next_class=='bracket_open'),
          'next_newline':      valid & (next_class=='newline'),
          'next_subword':      valid & (next_class=='subword'),
          'next_coordinator':  valid & (next_class=='coordinator')}
print(f"counts: valid={valid.sum()} boundary={M_boundary.sum()} midword={M_midword.sum()} "
      f"due={M_due.sum()} notdue={M_notdue.sum()} bnd_due={M_bnd_due.sum()} mid_notdue={M_mid_notdue.sum()} "
      + ' '.join(f"{k}={int(v.sum())}" for k, v in STRUCT.items()), flush=True)

# ---------------- grids: P(capital next) and CE under a term steer ----------------
@torch.no_grad()
def grids(coeffs):
    """coeffs: None (exact full model) | dict term_idx -> coefficient (steer via reassembly)."""
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    for i in range(0,NHELD,B0):
        idx=HELD[i:i+B0]; b=idx.shape[0]
        if coeffs is None:
            lg = fwd(idx).float()
        else:
            lg = fwd(idx, mode='steer', coeffs=coeffs, TMEAN=TMEAN, MEANF=MEANF).float()
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
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    if ra.std()==0 or rb.std()==0: return 0.0
    return float(np.corrcoef(ra,rb)[0,1])

# ---------------- natural reference + known-answer gates ----------------
print("natural (exact full model) grids ...", flush=True)
Pnat, Cnat = grids(None)
base_ce = float(np.nanmean(Cnat))
print(f"base CE mean {base_ce:.4f}", flush=True)

def census_dce(Cg):
    """census/section-91 convention: mean over ALL predict positions (no valid mask), vs base."""
    return float(np.mean(Cg[:, :T_-1] - Cnat[:, :T_-1]))

# gate: alpha=1 through the reassembly path must be the model (up to recon error)
Pre, Cre = grids({})
g_re = census_dce(Cre)
print(f"GATE reassembly alpha=1: census delta-CE {g_re:+.5f} (expect ~0)", flush=True)
assert abs(g_re) < 2e-3, "reassembly identity gate FAILED"

# numeric check: pair-coherent scaling == the explicit "difference knob" (scale the pair's
# NET functional write (devA+devB) on top of the full reassembly). Algebraic identity because
# the layer output is linear in the 15 terms; verified here to float precision on one batch.
with torch.no_grad():
    idx0 = HELD[:B0]
    lg_coh  = fwd(idx0, mode='steer', coeffs={IA: 2.0, IB: 2.0}, TMEAN=TMEAN, MEANF=MEANF).float()
    lg_diff = fwd(idx0, mode='steer', coeffs={}, extra=([IA, IB], 1.0),
                  TMEAN=TMEAN, MEANF=MEANF).float()
pair_ident = float((lg_coh - lg_diff).abs().max())
print(f"GATE pair identity: |coherent(alpha=2) - difference-knob(+1x net)| max {pair_ident:.2e}", flush=True)
assert pair_ident < 1e-3, "pair coherent==difference identity FAILED"
del lg_coh, lg_diff
torch.cuda.empty_cache()

# ---------------- SWEEP (same alpha grid as section 75) ----------------
ALPHAS = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DIALS = {'gated_AexMr':  [IA],
         'prior_MrxMr':  [IB],
         'pair_coherent':[IA, IB],
         'placebo_ExAr': [IP]}
GATE_AT_ZERO = {'gated_AexMr': 0.1479, 'prior_MrxMr': 0.0697, 'pair_coherent': 0.0587}

caches = {}
for dname, tlist in DIALS.items():
    print(f"\n===== SWEEP [{dname}] terms {[PNAMES[k] for k in tlist]} alpha in {ALPHAS} =====", flush=True)
    gc = {}
    for al in ALPHAS:
        if al == 1.0:
            gc[al] = (Pnat, Cnat); continue
        t0 = time.time()
        gc[al] = grids({k: al for k in tlist})
        Pg, Cg = gc[al]
        print(f"  alpha={al:>5}: P(cap) valid {stat(Pg[valid])['mean']:.4f} due {stat(Pg[M_due])['mean']:.4f} "
              f"notdue {stat(Pg[M_notdue])['mean']:.4f} boundary {stat(Pg[M_boundary])['mean']:.4f} "
              f"midword {stat(Pg[M_midword])['mean']:.4f} censusdCE {census_dce(Cg):+.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    caches[dname] = gc
    if dname in GATE_AT_ZERO:
        g0 = census_dce(gc[0.0][1])
        print(f"GATE {dname} alpha=0 census delta-CE {g0:+.4f} (expect {GATE_AT_ZERO[dname]:+.4f})", flush=True)
        assert abs(g0 - GATE_AT_ZERO[dname]) < 0.02, f"{dname} alpha=0 known-answer gate FAILED"

# ---------------- reports (VERBATIM sweep_report structure + structural splits) ----------------
def sweep_report(cache):
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
            'dCE_vs_natural': {
                'all_valid':   paired(Cg, Cnat, valid),
                'notdue':      paired(Cg, Cnat, M_notdue),
                'due':         paired(Cg, Cnat, M_due),
                'midword':     paired(Cg, Cnat, M_midword),
                'boundary':    paired(Cg, Cnat, M_boundary),
                **{k: paired(Cg, Cnat, mk) for k, mk in STRUCT.items()},
            },
            'dP_capital_vs_natural': {
                'due':         paired(Pg, Pnat, M_due),
                'notdue':      paired(Pg, Pnat, M_notdue),
                'boundary':    paired(Pg, Pnat, M_boundary),
                'midword':     paired(Pg, Pnat, M_midword),
                'mid_notdue':  paired(Pg, Pnat, M_mid_notdue),
                'bnd_due':     paired(Pg, Pnat, M_bnd_due),
            },
            'census_dCE': round(census_dce(Cg), 5),
        }
    return rep
reports = {d: sweep_report(caches[d]) for d in DIALS}

def dose_curve(cache, mask):
    return [stat(cache[al][0][mask])['mean'] for al in ALPHAS]
dose = {'alphas': ALPHAS}
for d in DIALS:
    dose[d] = {'due': dose_curve(caches[d], M_due), 'notdue': dose_curve(caches[d], M_notdue),
               'boundary': dose_curve(caches[d], M_boundary), 'midword': dose_curve(caches[d], M_midword),
               'valid': dose_curve(caches[d], valid)}
    dose[d]['spearman_due'] = round(spearman(ALPHAS, dose[d]['due']), 4)
    dose[d]['spearman_boundary'] = round(spearman(ALPHAS, dose[d]['boundary']), 4)

def reach(cache):
    due=[stat(cache[al][0][M_due])['mean'] for al in ALPHAS]
    notdue=[stat(cache[al][0][M_notdue])['mean'] for al in ALPHAS]
    return {'due_min': round(min(due),5), 'due_max': round(max(due),5),
            'due_at_alpha_min': ALPHAS[int(np.argmin(due))], 'due_at_alpha_max': ALPHAS[int(np.argmax(due))],
            'due_swing': round(max(due)-min(due),5),
            'notdue_min': round(min(notdue),5), 'notdue_max': round(max(notdue),5),
            'notdue_swing': round(max(notdue)-min(notdue),5),
            'natural_due': round(stat(cache[1.0][0][M_due])['mean'],5),
            'natural_notdue': round(stat(cache[1.0][0][M_notdue])['mean'],5)}
REACH = {d: reach(caches[d]) for d in DIALS}

# specificity (section-75 template): |capital change at due| per off-target CE cost at notdue
specificity = {}
for d in DIALS:
    specificity[d] = {}
    for al in ALPHAS:
        dP = reports[d][str(al)]['dP_capital_vs_natural']['due']['mean']
        dce_off = reports[d][str(al)]['dCE_vs_natural']['notdue']['mean']
        specificity[d][str(al)] = {
            'dP_capital_due': dP,
            'dCE_offtarget_notdue': dce_off,
            'dCE_all': reports[d][str(al)]['dCE_vs_natural']['all_valid']['mean'],
            'specificity_ratio_capitalgain_per_offtarget_CE':
                round(abs(dP)/abs(dce_off),3) if dce_off and abs(dce_off)>1e-6 else None}

# ---------------- SURGICAL TEST (the section-75 rematch) ----------------
def redteam_at(d, al):
    r = reports[d][str(al)]['dP_capital_vs_natural']
    bd = r['boundary']['mean']; mn = r['mid_notdue']['mean']
    return {'alpha': al, 'dP_boundary(due)': r['boundary'], 'dP_mid_notdue': r['mid_notdue'],
            'dP_midword': r['midword'],
            'notdue_over_due_dP_ratio': round(mn/bd, 3) if bd and abs(bd) > 1e-9 else None,
            'dCE_all_valid': reports[d][str(al)]['dCE_vs_natural']['all_valid']['mean']}
surgical = {
    'up_steer_gated': {str(al): redteam_at('gated_AexMr', al) for al in (2.0, 4.0, 8.0, 16.0)},
    'up_steer_prior_down': {str(al): redteam_at('prior_MrxMr', al) for al in (0.0, -1.0, -2.0, -4.0)},
    'down_steer_gated_suppression': {},
    'note': ('Up-steer gated arm: if the capital increase stays concentrated at boundaries '
             '(ratio << 1) the conditioning is preserved INSIDE the dial (the dial IS the '
             'conditioned path). Down-steer gated arm: surgical suppression = boundary capital '
             'drop >> not-due drop at small collateral delta cross-entropy.')}
for al in (0.0, -1.0, -2.0):
    r = reports['gated_AexMr'][str(al)]['dP_capital_vs_natural']
    bd = r['bnd_due']['mean']; nd = r['notdue']['mean']
    surgical['down_steer_gated_suppression'][str(al)] = {
        'dP_bnd_due': r['bnd_due'], 'dP_boundary': r['boundary'], 'dP_notdue': r['notdue'],
        'dP_mid_notdue': r['mid_notdue'],
        'boundary_drop_over_notdue_drop': round(bd/nd, 3) if nd and abs(nd) > 1e-9 else None,
        'dCE_all_valid': reports['gated_AexMr'][str(al)]['dCE_vs_natural']['all_valid']['mean'],
        'dCE_offtarget_notdue': reports['gated_AexMr'][str(al)]['dCE_vs_natural']['notdue']['mean']}

# ---------------- section-75 reference (embedded for side-by-side) ----------------
ref75 = json.load(open(f'{QK}/qk_edit_capselector.json'))
section75 = {'dose_response_target_P_capital_due': ref75['dose_response']['target_P_capital_due'],
             'dose_response_alphas': ref75['dose_response']['alphas'],
             'reach_target': ref75['reach']['target'],
             'specificity': ref75['specificity'],
             'context_conditioning_redteam': ref75['context_conditioning_redteam']}

OUTJ = {
 'meta': {'model': 'bilin18', 'layer': LI, 'held_slice': 'FW[448:600,:128]', 'batch': B0,
          'base_ce': round(base_ce, 4),
          'dials': {d: [PNAMES[k] for k in t] for d, t in DIALS.items()},
          'steer_formula': 'mo17 -> MEANF + sum_k c_k*(term_k - TMEAN_k); c=alpha on targeted '
                           'terms, 1 elsewhere; alpha=1 reassembled model, alpha=0 drop-term',
          'pair_note': 'the layer output is linear in the 15 terms, so pair_coherent (alpha on '
                       'both arms) adds exactly (alpha-1)*(devA+devB) = (alpha-1)*(pair net '
                       'functional write); scaling the pair difference/contrast and scaling both '
                       'arms coherently are THE SAME edit; the cancelling common mode has no '
                       'output-level degree of freedom (editing it with the net fixed is a no-op)',
          'placebo_energy_share': 0.0001,
          'P_capital': 'softmax mass on VOCAB_CLASS==capital (arc_caps metric)',
          'dCE_vs_natural': 'cross-entropy(alpha) - cross-entropy(natural), paired per position',
          'census_dCE': 'mean over ALL predict positions vs natural (gate currency, matches '
                        'qk_allterm_census / qk_L17_mixer convention)',
          'gates': {'recon_rel_err_global': gate_fro, 'recon_max_pos': st['maxrel'],
                    'tmean_max_abs_diff_vs_saved': tm_diff,
                    'reassembly_alpha1_census_dCE': round(g_re, 5),
                    'pair_coherent_equals_difference_knob_max_abs_logit_diff': pair_ident,
                    'gated_alpha0_census_dCE': round(census_dce(caches['gated_AexMr'][0.0][1]), 4),
                    'prior_alpha0_census_dCE': round(census_dce(caches['prior_MrxMr'][0.0][1]), 4),
                    'pair_alpha0_census_dCE': round(census_dce(caches['pair_coherent'][0.0][1]), 4),
                    'expected_at_zero': GATE_AT_ZERO},
          'machinery': 'VERBATIM qk_L17_mixer.py/qk_allterm_census.py (pair_terms, forward, '
                       'reassembly harness) + qk_arc_caps.py/qk_edit_capselector.py (capital '
                       'metric, position splits, sweep/reach/specificity/red-team template)',
          'alphas': ALPHAS},
 'counts': {'valid': int(valid.sum()), 'boundary': int(M_boundary.sum()), 'midword': int(M_midword.sum()),
            'due': int(M_due.sum()), 'notdue': int(M_notdue.sum()), 'bnd_due': int(M_bnd_due.sum()),
            'mid_notdue': int(M_mid_notdue.sum()),
            **{k: int(v.sum()) for k, v in STRUCT.items()}},
 'dose_response': dose,
 'reach': REACH,
 'specificity': specificity,
 'surgical_test': surgical,
 'section75_reference': section75,
 'sweeps_full': reports,
}
json.dump(OUTJ, open(OUT, 'w'), indent=1)

print("\n===== DOSE-RESPONSE P(capital at due) vs alpha =====", flush=True)
print("  alpha:        ", [f"{a:>6}" for a in ALPHAS], flush=True)
for d in DIALS:
    print(f"  {d:14s}", [f"{v:6.3f}" for v in dose[d]['due']], flush=True)
print("  section75 dir:", [f"{v:6.3f}" for v in section75['dose_response_target_P_capital_due']], flush=True)
print("\n===== REACH =====", flush=True)
for d in DIALS: print(f"  {d}: {REACH[d]}", flush=True)
print("\nSaved qk_edit_terms.json", flush=True)
print("QK EDIT TERMS DONE", flush=True)
