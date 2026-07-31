"""DEPTH-FIRST ALGORITHM ARC #3, part 1: WHAT ALGORITHM IS FEED-FORWARD BLOCK 3 (mlp.L3)?
mlp.L3 carries the fourth-largest block floor (0.6163 nats, section-89 census) but is characterized
only as "category-engine member". Section-89 anatomy: terms-to-95% = 4 (MrxMr + ArxMr + MexMr +
ArxAr); embedding causally dead. Context: section 97 -- block 0 is an exact current-token
feature-table + bigram correction consumed by blocks 1-3 (98.3%); section 96 -- block 2 is a dense
quadratic expansion consumed 84% by block 3's own square; section 86 -- block 1 is the five-term
hub. Block 3 is the category engine's FINAL STAGE before the code is used model-wide.

THIS SCRIPT (part 1):
 (gate) reconstruction at 1e-5 + reproduce the section-89 census numbers for layer 3 (floor
        0.6163; allbut_MrxMr 0.0062; allbut_ArxMr 0.0077; allbut_MexMr 0.0048; only_MrxMr 0.2483;
        only_ArxMr 0.2564; top2 0.0806; top4 0.0178) before any claims.
 (H3)  MIXER TEST on the ArxMr term (attention-recent x mlp-recent, the #2 term): section-97-style
       variance test -- fraction of the raw term's variance explained by current-token identity and
       by (previous, current) pair identity, with shuffled-label controls; contrast with MrxMr /
       MexMr / ArxAr. At layer 0 the analogous numbers were 0.95 (token table) and 0.86 (bigram);
       genuinely contextual terms should be LOW here.
 (b)   CLASS SIGNATURES + CONCRETE TEXT EXAMPLES: drop-one causal removal of each of the four live
       terms, class-summed delta-logit (base - dropped) at top-200 firing positions and at all
       valid positions; concrete held-back snippets at the top ArxMr and MrxMr firing sites with
       top boosted/suppressed tokens.

MACHINERY VERBATIM: five-group stream accumulators + pair_terms polarization + per-position means
+ keep/drop subset harness from qk_allterm_census.py via qk_arc_square.py (general-LI version);
class library (lex1/VOCAB_CLASS) verbatim qk_unsup_classpush.py via qk_arc_mlp0.py; variance
method group_r2 + shuffle_control verbatim qk_arc_mlp0.py (section-86 within-group vs total).
tier2_model.load_elriggs('bilin18'). Held-back FW[448:600,:128], paired standard errors, batch 6,
footprint <4GB, GPU guard. Output: qk_arc_mlp3.json (+ qk_arc_mlp3_means.pt for part 2;
part 2 = qk_arc_mlp3_2.py adds mediation profile / freeze-patch routing / category probe)."""
import json, os, sys, time, subprocess, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_arc_mlp3.json'

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
LI = 3
LIVE_NAMES = ['MrxMr', 'ArxMr', 'MexMr', 'ArxAr']       # section-89 terms-to-95% at layer 3
LIVE = {nme: PNAMES.index(nme) for nme in LIVE_NAMES}

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
    return terms, rho2

@torch.no_grad()
def fwd(idx, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None):
    """Forward with coarse-group accumulators, VERBATIM from qk_allterm_census.py via
    qk_arc_square.py (general LI). mode: None (full) | 'collect' (term sums + recon gate + raw
    live terms + rho2 + per-layer mo means for blocks 0..LI) | 'subset' (mo_LI -> MEANF +
    sum_{k in subset} (term_k - TMEAN[k])). Returns logits."""
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
            groups = [cE*x0, SA, aout, SM, MR]          # E, Ae, Ar, Me, Mr; x is x_pre here
            terms, rho2 = pair_terms(groups, x, W[0], W[1], W[2])
            if mode == 'collect':
                gs = sum(groups)
                stats['grp_err'] = max(stats['grp_err'],
                                       float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'][LI] += mo.sum(0)
                recon = sum(terms) + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
                stats['raw'].append(torch.stack([terms[LIVE[n]] for n in LIVE_NAMES]).float().cpu())
                stats['rho2'].append(rho2[..., 0].cpu())
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            del terms, groups
        elif track and li < LI and mode == 'collect':
            stats['mosum'][li] += mo.sum(0)
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

def ce_of(logits, idx):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(idx.shape[0], T_-1)

# ---------------- PASS 1: term means + DECOMPOSITION GATE + raw live terms ----------------
print("PASS 1: term means + reconstruction gate + raw live terms ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'mosum': {li: torch.zeros(T_, D, device=DEV) for li in range(LI+1)},
      'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0, 'grp_err': 0.0, 'raw': [], 'rho2': []}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MO_MEAN = {li: st['mosum'][li]/S_ for li in range(LI+1)}
MEANF = TMEAN.sum(0) + W[3]
gate_fro = (st['fro_num']/st['fro_den'])**0.5
mean_consist = float((MEANF - MO_MEAN[LI]).norm()/MO_MEAN[LI].norm())
print(f"GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e} groupsum {st['grp_err']:.2e} "
      f"meancons {mean_consist:.2e}", flush=True)
assert gate_fro < 1e-5, "decomposition gate FAILED at 1e-5"
RAW = torch.cat(st['raw'], 1)                    # (4, S, T, D) fp32 cpu: MrxMr, ArxMr, MexMr, ArxAr
RHO2 = torch.cat(st['rho2'], 0)                  # (S, T) cpu
assert bool(torch.isfinite(RAW).all()), "non-finite stored raw terms"
print(f"raw term storage finite; max |term| = {float(RAW.abs().max()):.1f}", flush=True)
del st
torch.save({'TMEAN': TMEAN.cpu(), 'MEANF': MEANF.cpu(), 'PNAMES': PNAMES,
            'MO_MEAN': {li: MO_MEAN[li].cpu() for li in MO_MEAN}},
           f'{QK}/qk_arc_mlp3_means.pt')

# ---------------- base CE ----------------
print("BASE: full-model cross-entropy ...", flush=True)
base_ce = torch.cat([ce_of(fwd(HELD[i:i+B0]), HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE mean {float(base_ce.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base_ce).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run_subset(subset):
    out = []
    for i in range(0, S_, B0):
        idx = HELD[i:i+B0]
        out.append(ce_of(fwd(idx, mode='subset', subset=subset, TMEAN=TMEAN, MEANF=MEANF), idx).cpu())
    return torch.cat(out, 0)

# ---------------- CENSUS REPRODUCTION GATE (section-89 layer-3 numbers) ----------------
CENSUS = json.load(open(f'{QK}/qk_allterm_census.json'))['layers']['3']
ALLK = list(range(NT))
repro_cfgs = {
    'mean_only': [],
    'only_MrxMr': [LIVE['MrxMr']],
    'only_ArxMr': [LIVE['ArxMr']],
    'top2_energy': [LIVE['MrxMr'], LIVE['ArxMr']],
    'top4_energy': [LIVE[n] for n in LIVE_NAMES],
}
census_repro = {}
print("CENSUS REPRODUCTION:", flush=True)
for name, subs in repro_cfgs.items():
    mn, se = dstat(run_subset(subs))
    ref = CENSUS['floor_dCE'] if name == 'mean_only' else CENSUS['configs'][name]['dCE']
    census_repro[name] = {'dCE': round(mn, 4), 'SE': round(se, 5), 'census_ref': ref,
                          'match': bool(abs(mn - ref) < 0.01)}
    print(f"  {name:12s} dCE {mn:+.4f} +- {se:.5f}  (census {ref})  match={census_repro[name]['match']}",
          flush=True)
assert all(v['match'] for v in census_repro.values()), "census reproduction gate FAILED"
print("census reproduction gate PASSED", flush=True)

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
# H3 VARIANCE TESTS -- section-86/97 method VERBATIM (group_r2 + shuffle_control from
# qk_arc_mlp0.py): how much of each raw live term is explained by current-token identity
# and by (previous, current) pair identity, vs genuinely contextual.
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
all_elig = np.ones(N, bool)
prev_ids = np.full(N, -1, np.int64)
prev_ids[1:] = flat_ids[:-1]
prev_ids[pos_t.reshape(-1) == 0] = -1
elig2 = ((pos_t.reshape(-1) >= 1) & ~np.isin(flat_ids, SPECIAL) & ~np.isin(prev_ids, SPECIAL))
pair_ids = prev_ids * V + flat_ids

H3_var = {}
for ki, nme in enumerate(LIVE_NAMES):
    X64 = RAW[ki].reshape(N, D).double()
    r2_cur, G_cur, cov_cur = group_r2(X64, flat_ids, all_elig, 5)
    r2_cur_shuf = shuffle_control(X64, flat_ids, cov_cur)
    rec = {'current_token_R2_ge5': round(r2_cur, 4), 'n_tokens_ge5': G_cur,
           'current_token_shuffled_control': round(r2_cur_shuf, 4)}
    if nme in ('ArxMr', 'MrxMr'):        # pair test on the mixer + the square for contrast
        for mc in (5, 3):
            r2_pair, G_pair, cov_pair = group_r2(X64, pair_ids, elig2, mc)
            r2_pair_shuf = shuffle_control(X64, pair_ids, cov_pair)
            r2_cur_sub, _, _ = group_r2(X64, flat_ids, cov_pair, 1)
            rec[f'pair_min_count_{mc}'] = {
                'pair_R2': round(r2_pair, 4), 'pair_R2_shuffled_control': round(r2_pair_shuf, 4),
                'current_alone_R2_same_subset': round(r2_cur_sub, 4),
                'pair_minus_current_incremental_R2': round(r2_pair - r2_cur_sub, 4),
                'n_pairs': G_pair, 'n_covered_positions': int(cov_pair.sum()),
                'coverage_of_eligible': round(float(cov_pair.sum())/float(elig2.sum()), 4)}
    H3_var[nme] = rec
    print(f"H3 {nme}: current-token R^2 {r2_cur:.4f} (shuf {r2_cur_shuf:.4f})"
          + (f" | pair>=5 R^2 {rec['pair_min_count_5']['pair_R2']:.4f} "
             f"(shuf {rec['pair_min_count_5']['pair_R2_shuffled_control']:.4f}, "
             f"cov {rec['pair_min_count_5']['coverage_of_eligible']:.3f})"
             if 'pair_min_count_5' in rec else ''), flush=True)
    del X64
H3_var['layer0_reference'] = {'ExE_current_token_R2': 0.953, 'ExAr_pair_R2_ge5': 0.861,
                              'note': 'section-97 layer-0 values for comparison'}

# =====================================================================================
# firing masks (top-200 by deviation norm, streampairs_2 convention) + example selection
# =====================================================================================
DEVN = {}
for ki, nme in enumerate(LIVE_NAMES):
    kk = LIVE[nme]
    DEVN[nme] = (RAW[ki] - TMEAN[kk].cpu().unsqueeze(0)).norm(dim=-1).numpy()   # (S,T)
KF = 200
fire_mask = {}
for nme in LIVE:
    a = DEVN[nme].copy().reshape(-1); a[bad.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KF)[-KF:]
    mk = np.zeros(N, bool); mk[tk] = True
    fire_mask[nme] = mk.reshape(S_, T_)

# examples: top firing positions with distinct current tokens (ArxMr mixer: 4; MrxMr square: 3)
def pick_examples(nme, n):
    fire_flat = np.where(fire_mask[nme].reshape(-1))[0]
    order_f = fire_flat[np.argsort(-DEVN[nme].reshape(-1)[fire_flat])]
    picks = []; seen_cur = set()
    for fi in order_f:
        s, t = fi // T_, fi % T_
        if t < 1: continue
        cu = int(held_np[s, t])
        if cu in seen_cur: continue
        picks.append((int(s), int(t))); seen_cur.add(cu)
        if len(picks) >= n: break
    return picks
EX_POS = {'ArxMr': pick_examples('ArxMr', 4), 'MrxMr': pick_examples('MrxMr', 3)}
for nme, pl in EX_POS.items():
    print(f"{nme} example positions: "
          f"{[(s, t, dec(held_np[s, t])) for s, t in pl]}", flush=True)

# =====================================================================================
# PASS B: drop-one causal removal per live term -- delta cross-entropy, class signatures,
# and per-example delta-logit collection.
# =====================================================================================
res_t = {nme: {'cs_fire': torch.zeros(len(CLASS_LIST), device=DEV), 'nf': 0,
               'cs_all': torch.zeros(len(CLASS_LIST), device=DEV), 'na': 0,
               'ce': []} for nme in LIVE}
ex_rows = {}                                            # (nme, s, t) -> delta-logit row
print(f"PASS B: 4 drop-one configurations x {math.ceil(S_/B0)} batches ...", flush=True)
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
        if nme in EX_POS:
            for (s, tt) in EX_POS[nme]:
                if i <= s < i+b:
                    ex_rows[(nme, s, tt)] = dl[s-i, tt].cpu()
        del abl, dl
    if bi % 5 == 0: print(f"  batch {bi+1}/{math.ceil(S_/B0)}  {time.time()-t0:.0f}s", flush=True)
    del base
print(f"PASS B done in {time.time()-t0:.0f}s", flush=True)

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
    cen = CENSUS['configs'].get(f'allbut_{nme}', {})
    class_sig[nme] = {
        'drop_one_dCE': round(mn, 4), 'drop_one_SE': round(se, 5),
        'census_allbut_dCE': cen.get('dCE'),
        'census_match': bool(abs(mn - cen.get('dCE', mn)) < 0.01) if cen else None,
        'fire': {'pushed_class': pf, 'pushed_val': vf, 'concentration': cf, 'top8': t8f, 'n': r['nf']},
        'all_valid': {'pushed_class': pa, 'pushed_val': va, 'concentration': ca, 'top8': t8a, 'n': r['na']}}
    print(f"{nme}: drop-one dCE {mn:+.4f}+-{se:.5f} (census {cen.get('dCE')}) | fire push {pf} "
          f"conc {cf} | all push {pa} conc {ca}", flush=True)

# examples: decode
examples = {}
for nme in EX_POS:
    exl = []
    for (s, tt) in EX_POS[nme]:
        dlr = ex_rows[(nme, s, tt)]
        tv, ti = torch.topk(dlr, 8)
        bv, bi_ = torch.topk(-dlr, 4)
        cls = (CMAT.cpu() @ dlr).numpy()
        corder = np.argsort(-np.abs(cls))[:5]
        ctx = tok.decode(held_np[s, max(0, tt-14):tt+1].tolist())
        exl.append({
            'seq': s, 'pos': tt,
            'previous_token': tok.decode([int(held_np[s, tt-1])]),
            'current_token': tok.decode([int(held_np[s, tt])]),
            'context_snippet_ending_at_pos': ctx,
            'actual_next_token': tok.decode([int(held_np[s, tt+1])]),
            'deviation_norm': round(float(DEVN[nme][s, tt]), 2),
            'top_boosted_tokens': [[tok.decode([int(j)]), round(float(v), 3)] for v, j in zip(tv, ti)],
            'top_suppressed_tokens': [[tok.decode([int(j)]), round(float(-v), 3)] for v, j in zip(bv, bi_)],
            'top_class_movement': {CLASS_LIST[j]: round(float(cls[j]), 3) for j in corder}})
        print(f"{nme} example ctx ...{ctx[-45:]!r}: boosts "
              f"{[x[0] for x in exl[-1]['top_boosted_tokens'][:5]]}", flush=True)
    examples[nme] = exl

out = {
 'meta': {
  'model': 'bilin18', 'held': 'FW[448:600,:128]', 'batch': B0, 'layer': LI,
  'machinery': 'term construction / group accumulators / polarization / per-position means / '
               'keep-drop harness VERBATIM qk_allterm_census.py via qk_arc_square.py (general LI); '
               'class library VERBATIM qk_unsup_classpush.py (lex1/VOCAB_CLASS); '
               'variance method = section-86/97 within-group vs total (qk_arc_mlp0.py group_r2)',
  'gate': 'reconstruction gated at 1e-5 AND census layer-3 numbers reproduced before all claims',
  'currency': 'delta cross-entropy per valid held position (nats), paired standard error; '
              'class-summed delta-logit (base - term-dropped)',
  'base_ce': round(float(base_ce.mean()), 4)},
 'gate': {'recon_rel_err_global': gate_fro, 'mean_consistency': mean_consist, 'pass': True},
 'census_reproduction': census_repro,
 'H3_variance_tests': H3_var,
 'class_signatures': class_sig,
 'examples': examples,
}
json.dump(out, open(OUT, 'w'), indent=1)
print("Saved qk_arc_mlp3.json", flush=True)
print("QK ARC MLP3 PART1 DONE", flush=True)
