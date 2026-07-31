"""CONTEXT TRANSPLANTS at the input of layer 17's feed-forward -- the test of section
93b's concrete prediction, part 1 of 2 (FORCE + SUPPRESS).

Background (RESULTS sections 89/91/93/93b): bilin18's layer-17 input decomposes exactly
into five groups (embedding E, attention-earlier Ae, attention-recent Ar, mlp-earlier Me,
mlp-recent Mr); the feed-forward is a differential pair whose gated arm (AexMr) ADDS
capital mass where context licenses it and whose prior arm (MrxMr) SUBTRACTS broad
generic mass; the conditioning signal is the ATTENTION-EARLIER group (accumulated
context). Section 93 established that AMPLITUDE edits on the terms cannot re-aim the
conditioning (forcing capitals where not due costs ~2.9-5.1 nats of GLOBAL damage);
section 93b's pointer: surgical unconditioned overrides would require editing the term's
INPUTS. THIS SCRIPT: transplant the attention-earlier group vector at layer 17's MLP
input, per position. Layer 17 is the LAST layer, so a per-position edit to its MLP input
affects ONLY that position's prediction -- collateral is localized by construction.

EDIT: at chosen target positions p, x_mlp_in(p) = x_pre(p) + t * (Ae_donor - Ae_own(p));
all other groups intact; blk.mlp recomputes on the edited input; the residual carrier x
and everything downstream of other positions untouched. t=1 = full transplant.

PART 1 EXPERIMENTS:
  FORCE:    ~500 mid-sentence capital-NOT-due targets (next token lowercase word,
            distance-since-newline >= 8), donors = genuine sentence-boundary positions
            (same sequence preferred). Does P(capital) rise AT the targets?
  SUPPRESS: ~500 genuine boundary capital-DUE targets, donors = mid-sentence not-due
            positions. Does P(capital) fall AT the targets?
  GATE per config: logits at all NON-edited positions bit-identical to the natural model.

MACHINERY VERBATIM: 5-group coarse-group-accumulator forward from qk_allterm_census.py /
qk_L17_mixer.py / qk_edit_terms.py (groups = [cE*x0, SA, aout, SM, MR] at LI=17, x is
x_pre); capital-class metric, lexical classes, position splits from qk_arc_caps.py /
qk_edit_terms.py. GATES: group-sum identity sum(groups) == x_pre (< 1e-4 rel);
base cross-entropy reproduces 3.4946; base P(capital) reproduces arc_caps step 1
(sentence-punct 0.34902, newline 0.55269).
Held-back FW[448:600,:128], paired standard errors, batch 6, <4GB, GPU guard.
Output: qk_stream_transplant.json (partial; part 2 adds controls + dose) and
qk_stream_transplant_state.pt (Ae/Me grids, masks, assignments, grids for part 2).
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
OUT = f'{QK}/qk_stream_transplant.json'
STATE = f'{QK}/qk_stream_transplant_state.pt'

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
LI = 17
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)

# group naming (VERBATIM qk_allterm_census.py): groups = [E, Ae, Ar, Me, Mr]
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
GI_AE = 1   # attention-earlier accumulator (SA)
GI_ME = 3   # mlp-earlier accumulator (SM)

@torch.no_grad()
def fwd(idx, mode=None, EM=None, DON=None, gi=None, dose=1.0, stats=None):
    """Forward verbatim from qk_allterm_census.py / qk_L17_mixer.py / qk_edit_terms.py
    (coarse-group stream accumulators to LI=17); RETURNS LOGITS.
    mode: None (full model) | 'collect' (store Ae/Me group grids + group-sum gate)
        | 'transplant' (x_mlp_in = x_pre + EM * dose * (DON - groups[gi]); mlp recomputes;
          residual carrier x untouched -- the edit exists only at the MLP input)."""
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
            if mode == 'collect':
                gs = groups[0] + groups[1] + groups[2] + groups[3] + groups[4]
                num = (gs - x).norm(dim=-1); den = x.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((gs - x).pow(2).sum()); stats['fro_den'] += float(x.pow(2).sum())
                stats['AE'].append(groups[GI_AE].cpu())
                stats['ME'].append(groups[GI_ME].cpu())
            elif mode == 'transplant':
                x_in = x + EM.unsqueeze(-1) * (dose * (DON - groups[gi]))
                mo = blk.mlp(F.rms_norm(x_in, (D,)))
            del groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# --- lexical classes copied VERBATIM from qk_arc_caps.py / qk_edit_terms.py ---
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

# ---------------- position-type grids (VERBATIM qk_edit_terms.py) ----------------
posmat=np.tile(np.arange(SEQL),NHELD).reshape(NHELD,SEQL)
held_special=np.isin(held_np,SPECIAL)
cur_class=VOCAB_CLASS[held_np]
nxt=np.zeros_like(held_np); nxt[:,:-1]=held_np[:,1:]
next_special=np.zeros_like(held_special); next_special[:,:-1]=held_special[:,1:]
valid=(posmat>0)&(posmat<SEQL-1)&~held_special&~next_special
cur_sentend=SENTEND_BOOL[held_np]
cur_newline=NEWLINE_BOOL[held_np]
cur_word=WORD_BOOL[held_np]
next_cap=CAP_BOOL[nxt]
next_word=WORD_BOOL[nxt]

M_sentend  = valid & cur_sentend
M_newline  = valid & cur_newline
M_boundary = (M_sentend | M_newline)                           # boundary = capital contextually DUE
M_bnd_due  = M_boundary & next_cap
# FORCE targets per spec: mid-sentence NOT-due (next token lowercase word, dsn >= 8)
M_target_notdue = valid & next_word & (HE_DSN>=8) & ~cur_sentend & ~cur_newline
M_nonboundary   = valid & ~M_boundary                          # random-donor pool (part 2 control)
print(f"counts: valid={valid.sum()} boundary={M_boundary.sum()} bnd_due={M_bnd_due.sum()} "
      f"target_notdue={M_target_notdue.sum()} nonboundary={M_nonboundary.sum()}", flush=True)

# ---------------- PASS 1: collect Ae/Me group grids + group-sum identity gate ----------------
print("PASS 1: collect attention-earlier / mlp-earlier group grids at L17 ...", flush=True)
st = {'AE': [], 'ME': [], 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
AE_ALL = torch.cat(st['AE'], 0)    # (S,T,D) attention-earlier group vectors, cpu float32
ME_ALL = torch.cat(st['ME'], 0)    # (S,T,D) mlp-earlier group vectors
gate_group = (st['fro_num']/st['fro_den'])**0.5
print(f"GATE group-sum identity: global rel {gate_group:.2e} maxpos {st['maxrel']:.2e}", flush=True)
assert gate_group < 1e-4, "five-group sum identity gate FAILED"

# ---------------- grids: P(capital next), CE, and the zero-collateral gate ----------------
def se(x):
    x=np.asarray(x,float); n=len(x)
    return float(x.std(ddof=1)/math.sqrt(n)) if n>1 else float('nan')
def stat(vals):
    vals=np.asarray(vals,float)
    return {'mean':round(float(vals.mean()),5) if len(vals) else None,
            'SE':round(se(vals),5) if len(vals)>1 else None, 'n':int(len(vals))}
def paired(a_grid, b_grid, mask):   # a - b at mask, paired per position
    d=(a_grid[mask]-b_grid[mask]); return stat(d)

@torch.no_grad()
def grids_tr(spec):
    """spec: None (natural model) | dict(gi, EMg (S,T) bool np, DONg (S,T,D) cpu tensor, dose).
    Returns Pcap, CE grids; for spec != None also the zero-collateral gate numbers
    (max abs logit difference over NON-edited positions vs a same-loop natural forward,
    and the fraction of non-edited positions with exactly zero logit change)."""
    Pcap=np.full((NHELD,SEQL),np.nan,np.float32); CE=np.full((NHELD,SEQL),np.nan,np.float32)
    max_ne = 0.0; ne_zero = 0; ne_tot = 0; max_ed = 0.0
    for i in range(0,NHELD,B0):
        idx=HELD[i:i+B0]; b=idx.shape[0]
        if spec is None:
            lg = fwd(idx).float()
        else:
            EM = torch.from_numpy(spec['EMg'][i:i+b]).to(DEV)
            DON = spec['DONg'][i:i+b].to(DEV)
            lg = fwd(idx, mode='transplant', EM=EM.float(), DON=DON,
                     gi=spec['gi'], dose=spec['dose']).float()
            lgb = fwd(idx).float()
            dmax = (lg-lgb).abs().amax(-1)                      # (b,T)
            ne = ~EM
            if ne.any():
                max_ne = max(max_ne, float(dmax[ne].max()))
                ne_zero += int((dmax[ne]==0).sum()); ne_tot += int(ne.sum())
            if EM.any(): max_ed = max(max_ed, float(dmax[EM].max()))
            del lgb, dmax, EM, DON
        sm=torch.softmax(lg,-1)
        pc=sm[...,CAP_MASK].sum(-1)
        lp=F.log_softmax(lg[:,:-1],-1); tgt=idx[:,1:]
        nll=-lp.gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        Pcap[i:i+b]=pc.cpu().numpy(); CE[i:i+b,:-1]=nll.cpu().numpy()
        del lg,sm,pc,lp,nll
    if spec is None: return Pcap, CE
    return Pcap, CE, {'max_abs_logit_diff_nonedited': max_ne,
                      'frac_nonedited_exactly_zero': round(ne_zero/max(ne_tot,1), 6),
                      'max_abs_logit_diff_edited': max_ed}

# ---------------- natural reference + known-answer gates ----------------
print("natural (exact full model) grids ...", flush=True)
Pnat, Cnat = grids_tr(None)
base_ce = float(np.nanmean(Cnat))
print(f"base CE mean {base_ce:.4f} (expect 3.4946)", flush=True)
assert abs(base_ce - 3.4946) < 0.002, "base cross-entropy gate FAILED"
p_se = float(Pnat[M_sentend].mean()); p_nl = float(Pnat[M_newline].mean())
print(f"GATE base P(capital): sentence-punct {p_se:.5f} (expect 0.34902) "
      f"newline {p_nl:.5f} (expect 0.55269)", flush=True)
assert abs(p_se-0.34902) < 0.005 and abs(p_nl-0.55269) < 0.005, "arc_caps step-1 gate FAILED"

# ---------------- target + donor assignment (seeded, same-sequence preferred) ----------------
rng = np.random.default_rng(0)
def sample_targets(mask, n):
    pool = np.argwhere(mask)
    sel = pool[rng.permutation(len(pool))[:min(n, len(pool))]]
    return sel   # (n,2) rows (s,t)

def assign_donors(targets, donor_mask):
    """for each target (s,t): random donor position from the SAME sequence if the donor
    pool has one (excluding the target position itself), else from the global pool."""
    by_seq = [np.flatnonzero(donor_mask[s]) for s in range(S_)]
    glob = np.argwhere(donor_mask)
    out = np.zeros_like(targets); same = 0
    for r, (s, t) in enumerate(targets):
        pool = by_seq[s]; pool = pool[pool != t]
        if len(pool) > 0:
            out[r] = (s, int(rng.choice(pool))); same += 1
        else:
            out[r] = glob[int(rng.integers(len(glob)))]
    return out, same/len(targets)

def build_spec(targets, donors, src_all, gi, dose):
    EMg = np.zeros((S_, T_), bool)
    DONg = torch.zeros(S_, T_, D)
    for (s, t), (ds, dt) in zip(targets, donors):
        EMg[s, t] = True
        DONg[s, t] = src_all[ds, dt]
    return {'gi': gi, 'EMg': EMg, 'DONg': DONg, 'dose': dose}

N_TGT = 500
TGT_FORCE = sample_targets(M_target_notdue, N_TGT)
TGT_SUPP  = sample_targets(M_bnd_due, N_TGT)
DON_FORCE, frac_same_force = assign_donors(TGT_FORCE, M_boundary)         # boundary donors
DON_SUPP,  frac_same_supp  = assign_donors(TGT_SUPP,  M_target_notdue)    # mid-sentence donors
print(f"targets: force {len(TGT_FORCE)} (same-seq donors {frac_same_force:.2f}) "
      f"suppress {len(TGT_SUPP)} (same-seq donors {frac_same_supp:.2f})", flush=True)

def mask_of(targets):
    g = np.zeros((S_, T_), bool)
    g[targets[:,0], targets[:,1]] = True
    return g
EM_FORCE = mask_of(TGT_FORCE); EM_SUPP = mask_of(TGT_SUPP)

# donor-vs-own attention-earlier geometry at the force targets (for interpretation)
own_ae = AE_ALL[TGT_FORCE[:,0], TGT_FORCE[:,1]]
don_ae = AE_ALL[DON_FORCE[:,0], DON_FORCE[:,1]]
cosod = F.cosine_similarity(own_ae, don_ae, dim=-1)
geom = {'own_Ae_norm': stat(own_ae.norm(dim=-1).numpy()),
        'donor_Ae_norm': stat(don_ae.norm(dim=-1).numpy()),
        'cos_own_donor': stat(cosod.numpy()),
        'P_capital_at_force_donors_natural': stat(Pnat[DON_FORCE[:,0], DON_FORCE[:,1]]),
        'frac_same_sequence_donor_force': round(frac_same_force, 4),
        'frac_same_sequence_donor_suppress': round(frac_same_supp, 4)}
print(f"geometry: own Ae norm {geom['own_Ae_norm']['mean']:.2f} donor {geom['donor_Ae_norm']['mean']:.2f} "
      f"cos(own,donor) {geom['cos_own_donor']['mean']:.3f} "
      f"donor natural P(capital) {geom['P_capital_at_force_donors_natural']['mean']:.3f}", flush=True)

def config_report(name, Pg, Cg, gate, EMg):
    tgt = EMg
    rest = valid & ~tgt
    rep = {'n_edited': int(tgt.sum()),
           'P_capital_before': stat(Pnat[tgt]), 'P_capital_after': stat(Pg[tgt]),
           'dP_capital_edited': paired(Pg, Pnat, tgt),
           'dCE_edited': paired(Cg, Cnat, tgt),
           'dCE_nonedited_valid': paired(Cg, Cnat, rest),
           'global_dCE_all_predict_positions':
               round(float(np.mean(Cg[:, :T_-1] - Cnat[:, :T_-1])), 6),
           'zero_collateral_gate': gate}
    print(f"[{name}] P(cap) {rep['P_capital_before']['mean']:.4f} -> {rep['P_capital_after']['mean']:.4f} "
          f"(dP {rep['dP_capital_edited']['mean']:+.4f} +- {rep['dP_capital_edited']['SE']:.4f}); "
          f"dCE@edited {rep['dCE_edited']['mean']:+.4f} +- {rep['dCE_edited']['SE']:.4f}; "
          f"dCE@nonedited {rep['dCE_nonedited_valid']['mean']:+.6f}; "
          f"global dCE {rep['global_dCE_all_predict_positions']:+.6f}; "
          f"gate max|dlogit|nonedited {gate['max_abs_logit_diff_nonedited']:.2e} "
          f"(frac exactly zero {gate['frac_nonedited_exactly_zero']:.4f})", flush=True)
    return rep

# ---------------- EXPERIMENT 1: FORCE (the section-93 impossible edit) ----------------
print("\n===== FORCE: boundary attention-earlier into mid-sentence not-due targets =====", flush=True)
spec_force = build_spec(TGT_FORCE, DON_FORCE, AE_ALL, GI_AE, 1.0)
Pf, Cf, gate_f = grids_tr(spec_force)
rep_force = config_report('FORCE t=1', Pf, Cf, gate_f, EM_FORCE)
assert gate_f['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (force)"

# ---------------- EXPERIMENT 2: SUPPRESS (reverse transplant) ----------------
print("\n===== SUPPRESS: mid-sentence attention-earlier into boundary due targets =====", flush=True)
spec_supp = build_spec(TGT_SUPP, DON_SUPP, AE_ALL, GI_AE, 1.0)
Ps, Cs, gate_s = grids_tr(spec_supp)
rep_supp = config_report('SUPPRESS t=1', Ps, Cs, gate_s, EM_SUPP)
assert gate_s['max_abs_logit_diff_nonedited'] < 1e-4, "zero-collateral gate FAILED (suppress)"

# ---------------- section-93 amplitude-edit reference (embedded for side-by-side) ----------------
ref93 = json.load(open(f'{QK}/qk_edit_terms.json'))
sec93 = {'note': ('amplitude route (scale the gated arm AexMr): not-due capital gain vs GLOBAL '
                  'delta cross-entropy over all valid positions -- the section-93 numbers the '
                  'transplant must beat on locality'),
         'up_steer_gated': {al: {'dP_mid_notdue': ref93['surgical_test']['up_steer_gated'][al]['dP_mid_notdue']['mean'],
                                 'dCE_all_valid_global': ref93['surgical_test']['up_steer_gated'][al]['dCE_all_valid']}
                            for al in ('2.0', '4.0', '8.0', '16.0')}}

# ---------------- save partial JSON + state for part 2 ----------------
OUTJ = {
 'meta': {'model': 'bilin18', 'layer': LI, 'held_slice': 'FW[448:600,:128]', 'batch': B0,
          'base_ce': round(base_ce, 4),
          'edit': 'x_mlp_in(p) = x_pre(p) + t*(Ae_donor - Ae_own(p)) at target positions only; '
                  'blk.mlp recomputes; residual carrier untouched; layer 17 is last, so only '
                  'the edited positions\' logits can change (gated below)',
          'targets': {'force': 'mid-sentence capital-not-due (next token lowercase word/subword, '
                               'distance-since-newline >= 8), n=500, donors = genuine boundary '
                               'positions (sentence-end punct or newline), same sequence preferred',
                      'suppress': 'boundary capital-due (boundary and next token actually capital), '
                                  'n=500, donors = mid-sentence not-due positions'},
          'P_capital': 'softmax mass on VOCAB_CLASS==capital (arc_caps metric)',
          'machinery': 'VERBATIM qk_allterm_census.py/qk_L17_mixer.py/qk_edit_terms.py forward '
                       '(five-group accumulators at L17) + qk_arc_caps.py lexical/position splits',
          'gates': {'group_sum_identity_rel_global': gate_group,
                    'group_sum_identity_rel_maxpos': st['maxrel'],
                    'base_ce': round(base_ce, 5),
                    'base_P_capital_sentend': round(p_se, 5),
                    'base_P_capital_newline': round(p_nl, 5)}},
 'counts': {'valid': int(valid.sum()), 'boundary': int(M_boundary.sum()),
            'bnd_due': int(M_bnd_due.sum()), 'target_notdue_pool': int(M_target_notdue.sum())},
 'donor_geometry': geom,
 'force_test': rep_force,
 'suppress_test': rep_supp,
 'section93_amplitude_reference': sec93,
}
json.dump(OUTJ, open(OUT, 'w'), indent=1)
torch.save({'AE_ALL': AE_ALL, 'ME_ALL': ME_ALL,
            'TGT_FORCE': TGT_FORCE, 'DON_FORCE': DON_FORCE,
            'TGT_SUPP': TGT_SUPP, 'DON_SUPP': DON_SUPP,
            'EM_FORCE': EM_FORCE, 'EM_SUPP': EM_SUPP,
            'Pnat': Pnat, 'Cnat': Cnat, 'Pf': Pf, 'Cf': Cf,
            'M_target_notdue': M_target_notdue, 'M_boundary': M_boundary,
            'M_bnd_due': M_bnd_due, 'M_nonboundary': M_nonboundary, 'valid': valid},
           STATE)
print("\nSaved qk_stream_transplant.json (partial) + state. QK STREAM TRANSPLANT PART 1 DONE", flush=True)
