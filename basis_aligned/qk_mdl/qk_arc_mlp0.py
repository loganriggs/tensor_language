"""DEPTH-FIRST ALGORITHM ARC #2: WHAT ALGORITHM IS FEED-FORWARD BLOCK 0 (mlp.L0)?
mlp.L0 carries the second-largest block floor (1.2341 nats, section-89 census) but its only
characterization is "category-engine member". Section-89 anatomy: at layer 0 only two input groups
exist (E = embedding x0, coefficient-tracked; Ar = the layer's own attention-0 output); terms-to-95%
= 2 (ExE + ExAr); layer 0 is the one DIAGONAL-dominated layer.

HYPOTHESES tested here (part 1):
 (H1) TOKEN LOOKUP TABLE: the ExE term is a per-current-token-identity transform (learned unigram
      feature expansion). Test: section-86 method verbatim -- fraction of the raw ExE term's
      variance explained by current-token identity (within-token vs total variance, tokens with
      >=5 held occurrences). PLUS the analytic sharpening available only at layer 0: E is exactly
      cE*x0 = f(current token), so the ExE NUMERATOR (term * rho^2 gauge scalar) must be exactly
      token-determined -- verified numerically as a positive control.
 (H2) BIGRAM FEATURE GENERATOR: the ExAr term multiplies the current token's embedding with
      attended context (heavily previous-token content at layer 0, h.L0.3) -- a learned
      (current, previous)-token interaction table. Test: fraction of the raw ExAr term's variance
      explained by the (previous, current) PAIR identity vs current-alone vs previous-alone, on the
      pair-covered subset (pairs with >=5 held occurrences; coverage reported honestly; shuffled-
      label control for small-group R^2 inflation).
 (b)  CLASS SIGNATURES + CONCRETE TEXT EXAMPLES: drop-one causal removal of each live term
      (ExE / ExAr / ArxAr), class-summed delta-logit (base - dropped) at top-200 firing positions
      and at all valid positions; per-token and per-position concrete examples (what the term
      pushes after specific tokens / pairs) on held-back text.

MACHINERY VERBATIM: layer-0 term construction, polarization through the actual mlp weights,
per-position means, keep/drop harness, reconstruction gate from qk_allterm_census.py (5 coarse
groups, 15 terms -- at layer 0 the groups Ae/Me/Mr are identically zero so 12 terms vanish;
gate tightened to 1e-5 per spec). Class library (lex1/VOCAB_CLASS) verbatim from
qk_unsup_classpush.py. tier2_model.load_elriggs('bilin18'). Held-back FW[448:600,:128] for all
causal/variance numbers, paired standard errors, batch 6, footprint <4GB, GPU guard.
Output: qk_arc_mlp0.json (part 2 = qk_arc_mlp0_2.py adds the H3 downstream-flow / category-probe
block to the same json)."""
import json, os, sys, time, subprocess, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_arc_mlp0.json'

# ---------------- GPU GUARD (verbatim from census) ----------------
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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 6
S_, T_ = HELD.shape
tok = AutoTokenizer.from_pretrained('gpt2')
def dec(t): return repr(tok.decode([int(t)]))
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)

# ---- the five coarse groups and the 15 group-pair terms (VERBATIM from qk_allterm_census.py) ----
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)   # 15
LI = 0
IEXE = PNAMES.index('ExE'); IEXAR = PNAMES.index('ExAr'); IARAR = PNAMES.index('ArxAr')
LIVE = {'ExE': IEXE, 'ExAr': IEXAR, 'ArxAr': IARAR}

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
W = mlp_wts(LI)

def pair_terms(groups, xpre, Lw, Rw, Dw):
    """15 interaction terms (list of (B,T,D)), sharing the common 1/rho^2 gauge; sum+bias == mo_L.
    VERBATIM construction from qk_allterm_census.py pair_terms."""
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None):
    """Forward verbatim from qk_allterm_census.py specialized to LI=0: at layer 0 the tracked
    accumulators reduce to cE = lambdas0+lambdas1 (coefficient of x0), SA=SM=MR=0.
    mode: None (full) | 'collect' (term sums + recon gate + raw live terms + rho2)
          | 'subset' (mo -> MEANF + sum_{k in subset} (term_k - TMEAN[k])).  Returns logits."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI and mode is not None:
            cE = blk.lambdas[0] + blk.lambdas[1]        # census cE at layer 0
            z = torch.zeros_like(x)
            groups = [cE*x0, z, aout, z, z]             # E, Ae, Ar, Me, Mr; x is x_pre here
            terms = pair_terms(groups, x, W[0], W[1], W[2])
            if mode == 'collect':
                gs = sum(groups)
                stats['grp_err'] = max(stats['grp_err'],
                                       float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
                stats['raw'].append(torch.stack([terms[IEXE], terms[IEXAR], terms[IARAR]]).half().cpu())
                stats['rho2'].append((x.pow(2).sum(-1)/D).cpu())
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            del terms, groups
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

def ce_of(logits, idx):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(idx.shape[0], T_-1)

# ---------------- PASS 1: term means + DECOMPOSITION GATE + raw live terms ----------------
print("PASS 1: term means + reconstruction gate + raw live terms ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0,
      'grp_err': 0.0, 'raw': [], 'rho2': []}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MO_MEAN = st['mosum']/S_
MEANF = TMEAN.sum(0) + W[3]
gate_fro = (st['fro_num']/st['fro_den'])**0.5
mean_consist = float((MEANF - MO_MEAN).norm()/MO_MEAN.norm())
print(f"GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e} groupsum {st['grp_err']:.2e} "
      f"meancons {mean_consist:.2e}", flush=True)
assert gate_fro < 1e-5, "decomposition gate FAILED at 1e-5"
RAW = torch.cat(st['raw'], 1)                    # (3, S, T, D) fp16 cpu: ExE, ExAr, ArxAr
RHO2 = torch.cat(st['rho2'], 0)                  # (S, T) cpu
assert bool(torch.isfinite(RAW.float()).all()), "fp16 overflow in stored raw terms"
print(f"raw term storage finite; max |term| = {float(RAW.float().abs().max()):.1f}", flush=True)
del st
torch.save({'TMEAN': TMEAN.cpu(), 'MEANF': MEANF.cpu(), 'PNAMES': PNAMES},
           f'{QK}/qk_arc_mlp0_means.pt')
gate_rec = {'recon_rel_err_global': gate_fro, 'recon_rel_err_max_pos': None,
            'group_sum_rel_err_max': None, 'mean_consistency': mean_consist, 'pass': True}

# ---------------- base CE ----------------
print("BASE: full-model cross-entropy ...", flush=True)
base_ce = torch.cat([ce_of(fwd(HELD[i:i+B0]), HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE mean {float(base_ce.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base_ce).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# =====================================================================================
# LEXICAL CLASS LIBRARY -- VERBATIM from qk_unsup_classpush.py (lex1 / VOCAB_CLASS).
# =====================================================================================
BRACKETS_OPEN  = set("([{<")
BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN     = set("“‘`")
QUOTE_CLOSE    = set("”’")
QUOTE_STRAIGHT = set("\"'")
PUNCT  = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS  = {"the","a","an","this","that","these","those","some","any","each",
                "every","no","another","such"}
PRONOUNS     = {"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}

def lex1(s):
    if s == "": return 'other'
    if ('�' in s) or (s == tok.eos_token or '<|endoftext|>' in s): return 'special'
    if '\n' in s: return 'newline'
    body = s.strip(); low = body.lower()
    if body == "": return 'other'
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
    lead_space = s.startswith(' ')
    if lead_space and body.isalpha() and len(body) > 1: return 'word'
    if (not lead_space) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'

VOCAB_CLASS = np.array([lex1(tok.decode([t])) for t in range(V)], dtype=object)
CLASS_LIST = sorted(set(VOCAB_CLASS.tolist()))
CIDX = {c: i for i, c in enumerate(CLASS_LIST)}
CMAT = torch.zeros(len(CLASS_LIST), V, device=DEV)
for t in range(V):
    CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
PUSH_EXCLUDE = {'special'}
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
print(f"classes ({len(CLASS_LIST)}); {len(SPECIAL)} special ids", flush=True)

held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(T_), S_).reshape(S_, T_)
bad = (pos_t == 0) | np.isin(held_np, SPECIAL) | (pos_t >= T_-1)   # streampairs_2 trigger exclusion
valid = ~((pos_t >= T_-1))                                          # positions with a next token

# =====================================================================================
# H1: TOKEN-LOOKUP TEST on raw ExE (section-86 method verbatim) + analytic numerator control.
# =====================================================================================
def group_r2(X64, gids, elig, min_count):
    """Variance-explained by group identity on positions elig & group-count>=min_count.
    X64 (N,D) float64 cpu torch; gids (N,) int64 numpy; returns r2, n_groups, covered mask."""
    g = gids[elig]
    uniq, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    keep = counts >= min_count
    cov = np.zeros(len(gids), bool)
    idx_el = np.where(elig)[0]
    cov[idx_el[keep[inv]]] = True
    Xc = X64[torch.from_numpy(cov)]
    gcov = gids[cov]
    u2, inv2 = np.unique(gcov, return_inverse=True)
    G = len(u2)
    s = torch.zeros(G, X64.shape[1], dtype=torch.float64)
    s.index_add_(0, torch.from_numpy(inv2), Xc)
    n = torch.from_numpy(np.bincount(inv2).astype(np.float64))
    sq = float(Xc.pow(2).sum())
    within = sq - float((s.pow(2).sum(1)/n).sum())
    grand = Xc.sum(0)
    tot = sq - float(grand.pow(2).sum())/Xc.shape[0]
    r2 = 1.0 - within/max(tot, 1e-12)
    return r2, G, cov

def shuffle_control(X64, gids, cov, seed=0):
    rng = np.random.default_rng(seed)
    g = gids.copy()
    idxc = np.where(cov)[0]
    g[idxc] = g[rng.permutation(idxc)]
    r2, _, _ = group_r2(X64, g, cov, 1)
    return r2

flat_ids = held_np.reshape(-1)
N = S_ * T_
EXE64 = RAW[0].reshape(N, D).double()
all_elig = np.ones(N, bool)                                # section-86 used all positions
r2_exe, G_exe, cov_exe = group_r2(EXE64, flat_ids, all_elig, 5)
r2_exe_shuf = shuffle_control(EXE64, flat_ids, cov_exe)
print(f"H1 ExE: current-token R^2 = {r2_exe:.4f} ({G_exe} tokens >=5 occ, "
      f"{int(cov_exe.sum())}/{N} positions; shuffled control {r2_exe_shuf:.4f})", flush=True)
# analytic numerator control: ExE * rho2 must be EXACTLY a function of the current token
NUM64 = EXE64 * RHO2.reshape(N, 1).double()
r2_num, _, _ = group_r2(NUM64, flat_ids, all_elig, 5)
print(f"H1 control: ExE NUMERATOR (gauge removed) current-token R^2 = {r2_num:.6f} "
      f"(should be ~1: E = cE*x0 is exactly token-determined at layer 0)", flush=True)
del NUM64

# =====================================================================================
# H2: BIGRAM TEST on raw ExAr -- pair vs current-alone vs previous-alone.
# =====================================================================================
EXAR64 = RAW[1].reshape(N, D).double()
prev_ids = np.full(N, -1, np.int64)
prev_ids[1:] = flat_ids[:-1]
prev_ids[pos_t.reshape(-1) == 0] = -1                       # no previous within sequence
elig2 = ((pos_t.reshape(-1) >= 1) & ~np.isin(flat_ids, SPECIAL) & ~np.isin(prev_ids, SPECIAL))
pair_ids = prev_ids * V + flat_ids
H2 = {}
for mc in (5, 3):
    r2_pair, G_pair, cov_pair = group_r2(EXAR64, pair_ids, elig2, mc)
    r2_pair_shuf = shuffle_control(EXAR64, pair_ids, cov_pair)
    # same-subset singles (groups are supersets of pairs, min_count 1 on covered subset)
    r2_cur_sub, G_cs, _ = group_r2(EXAR64, flat_ids, cov_pair, 1)
    r2_prev_sub, G_ps, _ = group_r2(EXAR64, prev_ids, cov_pair, 1)
    H2[f'min_count_{mc}'] = {
        'pair_R2': round(r2_pair, 4), 'pair_R2_shuffled_control': round(r2_pair_shuf, 4),
        'current_alone_R2_same_subset': round(r2_cur_sub, 4),
        'previous_alone_R2_same_subset': round(r2_prev_sub, 4),
        'pair_minus_current_incremental_R2': round(r2_pair - r2_cur_sub, 4),
        'n_pairs': G_pair, 'n_covered_positions': int(cov_pair.sum()),
        'n_eligible_positions': int(elig2.sum()),
        'coverage_of_eligible': round(float(cov_pair.sum())/float(elig2.sum()), 4)}
    print(f"H2 ExAr (pairs >={mc} occ): pair R^2 {r2_pair:.4f} (shuf {r2_pair_shuf:.4f}) | "
          f"current-alone {r2_cur_sub:.4f} | previous-alone {r2_prev_sub:.4f} | "
          f"coverage {H2[f'min_count_{mc}']['coverage_of_eligible']:.3f}", flush=True)
# broad singles on own subsets (comparable to H1 convention)
r2_cur_own, G_co, _ = group_r2(EXAR64, flat_ids, elig2, 5)
r2_prev_own, G_po, _ = group_r2(EXAR64, prev_ids, elig2, 5)
H2['current_alone_R2_own_subset_ge5'] = round(r2_cur_own, 4)
H2['previous_alone_R2_own_subset_ge5'] = round(r2_prev_own, 4)
print(f"H2 singles on own >=5 subsets: current {r2_cur_own:.4f} previous {r2_prev_own:.4f}", flush=True)
# ArxAr for completeness: pure-context term, current-token R^2 should be LOW
ARAR64 = RAW[2].reshape(N, D).double()
r2_arar_cur, _, _ = group_r2(ARAR64, flat_ids, all_elig, 5)
print(f"contrast: ArxAr current-token R^2 = {r2_arar_cur:.4f} (pure-context expectation: low)", flush=True)
del EXE64, EXAR64, ARAR64

# =====================================================================================
# firing masks (top-200 by deviation norm, streampairs_2 convention) + example selection
# =====================================================================================
DEVN = {}
for nme, k3 in [('ExE', 0), ('ExAr', 1), ('ArxAr', 2)]:
    kk = LIVE[nme]
    DEVN[nme] = (RAW[k3].float() - TMEAN[kk].cpu().unsqueeze(0)).norm(dim=-1).numpy()   # (S,T)
KF = 200
fire_mask = {}
for nme in LIVE:
    a = DEVN[nme].copy().reshape(-1); a[bad.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KF)[-KF:]
    mk = np.zeros(N, bool); mk[tk] = True
    fire_mask[nme] = mk.reshape(S_, T_)

# H1 examples: 3 tokens (>=5 occ, non-special) with the most distinctive per-token mean ExE
EXEf = RAW[0].reshape(N, D).float()
grand = EXEf.mean(0)
uniqt, cnts = np.unique(flat_ids, return_counts=True)
cand_toks = [int(t) for t, c in zip(uniqt, cnts) if c >= 8 and t not in set(SPECIAL.tolist())]
tok_score = []
for t in cand_toks:
    sel = np.where(flat_ids == t)[0]
    tok_score.append((float((EXEf[sel].mean(0) - grand).norm()), t, len(sel)))
tok_score.sort(reverse=True)
TOK_EX = [(t, n) for _, t, n in tok_score[:3]]
print(f"H1 example tokens: {[(dec(t), n) for t, n in TOK_EX]}", flush=True)
tok_masks = {}
for t, _n in TOK_EX:
    mkk = (held_np == t) & ~bad
    tok_masks[t] = mkk
del EXEf

# H2 examples: 4 distinct (previous, current) pairs among top ExAr firing positions
fire_flat = np.where(fire_mask['ExAr'].reshape(-1))[0]
order_f = fire_flat[np.argsort(-DEVN['ExAr'].reshape(-1)[fire_flat])]
PAIR_EX = []; seen_cur = set()
for fi in order_f:
    s, t = fi // T_, fi % T_
    if t < 1: continue
    pr, cu = int(held_np[s, t-1]), int(held_np[s, t])
    if pr in set(SPECIAL.tolist()) or cu in seen_cur: continue
    PAIR_EX.append((int(s), int(t), pr, cu)); seen_cur.add(cu)
    if len(PAIR_EX) >= 4: break
print(f"H2 example pairs: {[(dec(p), dec(c)) for _, _, p, c in PAIR_EX]}", flush=True)

# =====================================================================================
# PASS B: drop-one causal removal per live term -- delta cross-entropy, class signatures,
# and per-example delta-logit collection.
# =====================================================================================
res_t = {nme: {'cs_fire': torch.zeros(len(CLASS_LIST), device=DEV), 'nf': 0,
               'cs_all': torch.zeros(len(CLASS_LIST), device=DEV), 'na': 0,
               'ce': []} for nme in LIVE}
tokex_sum = {t: torch.zeros(V, device=DEV) for t, _ in TOK_EX}   # drop-ExE dlogit sums per token
tokex_n = {t: 0 for t, _ in TOK_EX}
pairex_rows = {}                                                  # drop-ExAr dlogit at example pos
print(f"PASS B: 3 drop-one configurations x {math.ceil(S_/B0)} batches ...", flush=True)
t0 = time.time()
for bi, i in enumerate(range(0, S_, B0)):
    sb = slice(i, min(i+B0, S_))
    idx = HELD[sb]; b = idx.shape[0]
    base = fwd(idx).float()
    for nme, kk in LIVE.items():
        subs = [j for j in range(NT) if j != kk]
        abl = fwd(idx, mode='subset', subset=subs, TMEAN=TMEAN, MEANF=MEANF).float()
        res_t[nme]['ce'].append(ce_of(abl, idx).cpu())
        dl = (base[:, :T_-1] - abl[:, :T_-1])                    # positive => term boosts
        fm = torch.from_numpy(fire_mask[nme][sb, :T_-1]).to(DEV)
        if fm.any():
            res_t[nme]['cs_fire'] += CMAT @ dl[fm].sum(0); res_t[nme]['nf'] += int(fm.sum())
        vm = torch.from_numpy(valid[sb, :T_-1]).to(DEV)
        res_t[nme]['cs_all'] += CMAT @ dl[vm].sum(0); res_t[nme]['na'] += int(vm.sum())
        if nme == 'ExE':
            for t, _n in TOK_EX:
                tm = torch.from_numpy(tok_masks[t][sb, :T_-1]).to(DEV)
                if tm.any():
                    tokex_sum[t] += dl[tm].sum(0); tokex_n[t] += int(tm.sum())
        if nme == 'ExAr':
            for (s, tt, pr, cu) in PAIR_EX:
                if i <= s < i+b:
                    pairex_rows[(s, tt)] = dl[s-i, tt].cpu()
        del abl, dl
    if bi % 5 == 0: print(f"  batch {bi+1}/{math.ceil(S_/B0)}  {time.time()-t0:.0f}s", flush=True)
    del base
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

CENSUS = json.load(open(f'{QK}/qk_allterm_census.json'))['layers']['0']['configs']
def top8(cs):
    order = np.argsort(-np.abs(cs))
    pushed = next(j for j in order if CLASS_LIST[j] not in PUSH_EXCLUDE)
    conc = float(abs(cs[pushed])/max(1e-9, float(np.abs(cs).sum())))
    return ({CLASS_LIST[j]: round(float(cs[j]), 4) for j in order[:8]},
            CLASS_LIST[pushed], round(float(cs[pushed]), 4), round(conc, 4))

class_sig = {}
for nme in LIVE:
    r = res_t[nme]
    mn, se = dstat(torch.cat(r['ce'], 0))
    csf = (r['cs_fire']/max(1, r['nf'])).cpu().numpy()
    csa = (r['cs_all']/max(1, r['na'])).cpu().numpy()
    t8f, pf, vf, cf = top8(csf)
    t8a, pa, va, ca = top8(csa)
    cen = CENSUS.get(f'allbut_{nme}', {})
    class_sig[nme] = {
        'drop_one_dCE': round(mn, 4), 'drop_one_SE': round(se, 5),
        'census_allbut_dCE': cen.get('dCE'), 'census_match': bool(abs(mn - cen.get('dCE', mn)) < 0.01),
        'fire': {'pushed_class': pf, 'pushed_val': vf, 'concentration': cf, 'top8': t8f, 'n': r['nf']},
        'all_valid': {'pushed_class': pa, 'pushed_val': va, 'concentration': ca, 'top8': t8a, 'n': r['na']}}
    print(f"{nme}: drop-one dCE {mn:+.4f}+-{se:.5f} (census {cen.get('dCE')}) | fire push {pf} "
          f"conc {cf} | all push {pa} conc {ca}", flush=True)

# examples: decode
tok_examples = []
for t, n in TOK_EX:
    dlm = (tokex_sum[t]/max(1, tokex_n[t])).cpu()
    tv, ti = torch.topk(dlm, 8)
    bv, bi_ = torch.topk(-dlm, 4)
    cls = (CMAT.cpu() @ dlm).numpy()
    corder = np.argsort(-np.abs(cls))[:5]
    tok_examples.append({
        'token': tok.decode([t]), 'token_id': t, 'n_positions': tokex_n[t],
        'top_boosted_tokens': [[tok.decode([int(j)]), round(float(v), 3)] for v, j in zip(tv, ti)],
        'top_suppressed_tokens': [[tok.decode([int(j)]), round(float(-v), 3)] for v, j in zip(bv, bi_)],
        'top_class_movement': {CLASS_LIST[j]: round(float(cls[j]), 3) for j in corder}})
    print(f"H1 example {dec(t)} (n={tokex_n[t]}): boosts "
          f"{[x[0] for x in tok_examples[-1]['top_boosted_tokens'][:5]]}", flush=True)

pair_examples = []
for (s, tt, pr, cu) in PAIR_EX:
    dlr = pairex_rows[(s, tt)]
    tv, ti = torch.topk(dlr, 8)
    bv, bi_ = torch.topk(-dlr, 4)
    cls = (CMAT.cpu() @ dlr).numpy()
    corder = np.argsort(-np.abs(cls))[:5]
    ctx = tok.decode(held_np[s, max(0, tt-10):tt+1].tolist())
    pair_examples.append({
        'seq': s, 'pos': tt, 'previous_token': tok.decode([pr]), 'current_token': tok.decode([cu]),
        'context_snippet': ctx, 'actual_next_token': tok.decode([int(held_np[s, tt+1])]),
        'exar_deviation_norm': round(float(DEVN['ExAr'][s, tt]), 2),
        'top_boosted_tokens': [[tok.decode([int(j)]), round(float(v), 3)] for v, j in zip(tv, ti)],
        'top_suppressed_tokens': [[tok.decode([int(j)]), round(float(-v), 3)] for v, j in zip(bv, bi_)],
        'top_class_movement': {CLASS_LIST[j]: round(float(cls[j]), 3) for j in corder}})
    print(f"H2 example ({dec(pr)},{dec(cu)}) ctx ...{ctx[-40:]!r}: boosts "
          f"{[x[0] for x in pair_examples[-1]['top_boosted_tokens'][:5]]}", flush=True)

out = {
 'meta': {
  'model': 'bilin18', 'held': 'FW[448:600,:128]', 'batch': B0, 'layer': 0,
  'machinery': 'term construction / polarization / per-position means / keep-drop harness VERBATIM '
               'qk_allterm_census.py (5 groups, 15 terms; Ae/Me/Mr identically zero at layer 0); '
               'class library VERBATIM qk_unsup_classpush.py (lex1/VOCAB_CLASS); '
               'variance method = section-86 within-group vs total (qk_hub_streampairs_2.py)',
  'gate': 'reconstruction gated at 1e-5 (measured below) before all causal numbers',
  'currency': 'delta cross-entropy per valid held position (nats), paired standard error; '
              'class-summed delta-logit (base - term-dropped)',
  'base_ce': round(float(base_ce.mean()), 4)},
 'gate': {'recon_rel_err_global': gate_fro, 'mean_consistency': mean_consist, 'pass': True},
 'H1_token_lookup': {
  'exe_current_token_R2': round(r2_exe, 4), 'n_tokens_ge5': G_exe,
  'n_covered_positions': int(cov_exe.sum()), 'n_total_positions': N,
  'shuffled_label_control_R2': round(r2_exe_shuf, 4),
  'numerator_gauge_removed_R2': round(r2_num, 6),
  'analytic_note': 'at layer 0, E = (lambda0+lambda1)*x0 is EXACTLY a function of the current '
                   'token; the ExE numerator is token-determined by construction, so all '
                   'residual variance in the ExE term comes from the shared 1/rho^2 gauge scalar '
                   '(context-dependent normalization), verified by the numerator R^2',
  'contrast_arar_current_token_R2': round(r2_arar_cur, 4)},
 'H2_bigram': H2,
 'class_signatures': class_sig,
 'examples': {'token_lookup_ExE': tok_examples, 'bigram_ExAr': pair_examples},
}
json.dump(out, open(OUT, 'w'), indent=1)
print("Saved qk_arc_mlp0.json", flush=True)
print("QK ARC MLP0 PART1 DONE", flush=True)
