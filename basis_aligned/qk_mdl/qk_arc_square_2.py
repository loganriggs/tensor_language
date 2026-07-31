"""DEPTH-FIRST ALGORITHM ARC #1, script 2: (A) K=64 keep-side split configs (completes script 1's
frame); (B) TOP-PAIR characterization -- parent-direction trigger classes (the section-68 way),
held-back text snippets where both parents are active, and the pair product's causal class-summed
delta-logit push at its firing positions; (C) DOWNSTREAM READ (H3): freeze-patching -- ablate
layer 2's MrxMr term and route the damage: direct-residual-carry-only vs mediated through each
later layer's MLP (unfreeze one mediator at a time) vs later attention; plus the direct-readout
class signature of the damage.

Machinery: forward + group accumulators + term construction VERBATIM qk_allterm_census.py /
qk_arc_square.py; class library (lex1/VOCAB_CLASS) + class-summed delta-logit + trigger masks
VERBATIM qk_unsup_classpush.py / qk_mlp1_tail.py; paired standard errors as qk_mlp1_tail.py.
Held FW[448:600,:128], batch 6. Appends results into qk_arc_square.json."""
import json, sys, math, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

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
def dec(t): return repr(tok.decode([int(t)]))
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 6
S_, T_ = HELD.shape
LI = 2; KMAX = 64; KCAUSAL = 200
print(f"bilin18 NL={NL} D={D} held {S_}x{T_}", flush=True)

ST = torch.load(f'{QK}/qk_arc_square_state.pt', weights_only=False)
U = ST['U'].to(DEV); PMEAN = ST['PMEAN'].to(DEV)
TMEAN = ST['TMEAN'].to(DEV); MEANF = ST['MEANF'].to(DEV)
A_HELD = ST['A_HELD'].numpy(); RHO2_HELD = ST['RHO2_HELD'].numpy()
base_ce_s1 = ST['base_ce']
PE = ST['pair_energy_share_topcross']

GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']; NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS); MRMR = PNAMES.index('MrxMr')

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
W = mlp_wts(LI)

def pair_terms(groups, xpre, Lw, Rw, Dw):
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms, rho2

# fixed pair vectors (recomputed, cheap)
PLu = U @ W[0].T; PRu = U @ W[1].T
S_hid = 0.5*(PLu[:, None, :]*PRu[None, :, :] + PLu[None, :, :]*PRu[:, None, :])
Bp = torch.einsum('ijh,dh->ijd', S_hid, W[2])
del S_hid, PLu, PRu
MEAN_PROJ = torch.einsum('tij,ijd->td', PMEAN, Bp)              # per-position mean of proj64
ar64 = torch.arange(KMAX, device=DEV)
MEAN_DIAG = torch.einsum('ti,id->td', PMEAN[:, ar64, ar64], Bp[ar64, ar64])
MEAN_CROSS = MEAN_PROJ - MEAN_DIAG

# ---------------- lexical class library (VERBATIM lex1/VOCAB_CLASS) ----------------
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
for t in range(V): CMAT[CIDX[VOCAB_CLASS[t]], t] = 1.0
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(T_), S_).reshape(S_, T_)
bad_trigger = (pos_t == 0) | np.isin(held_np, SPECIAL) | (pos_t >= T_-1)

def topmask(act):
    a = act.copy().reshape(-1); a[bad_trigger.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KCAUSAL)[-KCAUSAL:]
    mk = np.zeros(S_*T_, bool); mk[tk] = True
    return mk.reshape(S_, T_)

# =====================================================================================
# forward with (i) MrxMr-part interventions at LI=2 and (ii) later-layer freeze-patching.
# config: dict(ablate=bool, part in {None,'proj64','diag64','cross64'}, drop_pair=None|(i,j),
#              freeze_attn=set, freeze_mlp=set)
# cache: dict to store/read clean component outputs per batch (li -> (aout, mo)).
# =====================================================================================
@torch.no_grad()
def fwd(idx, config=None, cache=None, fill_cache=False, want_logits=False, term_norm_out=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cfg_ = config or {}
    fa = cfg_.get('freeze_attn', ()); fm = cfg_.get('freeze_mlp', ())
    track = bool(cfg_.get('ablate') or cfg_.get('part') or cfg_.get('drop_pair') or fill_cache or term_norm_out is not None)
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
        aout = a.c_proj(yh.reshape(B, T, -1))
        if li in fa: aout = cache[li][0]
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in fm: mo = cache[li][1]
        if track and li == LI:
            groups = [cE*x0, SA, aout, SM, MR]
            terms, rho2 = pair_terms(groups, x, W[0], W[1], W[2])
            term_dev = terms[MRMR] - TMEAN[MRMR][None]
            if term_norm_out is not None:
                term_norm_out.append(term_dev.norm(dim=-1).cpu())
            if cfg_.get('ablate'):
                mo = mo - term_dev
                part = cfg_.get('part')
                if part is not None:
                    av = MR @ U.T
                    P = av[:, :, :, None] * av[:, :, None, :] / rho2[..., None]
                    if part == 'proj64':
                        mo = mo + torch.einsum('btij,ijd->btd', P, Bp) - MEAN_PROJ[None]
                    elif part == 'diag64':
                        mo = mo + torch.einsum('bti,id->btd', P[:, :, ar64, ar64], Bp[ar64, ar64]) - MEAN_DIAG[None]
                    elif part == 'cross64':
                        pj = torch.einsum('btij,ijd->btd', P, Bp)
                        dg = torch.einsum('bti,id->btd', P[:, :, ar64, ar64], Bp[ar64, ar64])
                        mo = mo + (pj - dg) - MEAN_CROSS[None]
                    del av, P
            dp = cfg_.get('drop_pair')
            if dp is not None:
                i_, j_ = dp
                av = MR @ U.T
                pdev = (av[:, :, i_]*av[:, :, j_]/rho2[..., 0] - PMEAN[None, :, i_, j_])
                mo = mo - 2.0*pdev[..., None]*Bp[i_, j_][None, None]
                del av
            del terms, groups
        if fill_cache: cache[li] = (aout, mo)
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return (ce, logits) if want_logits else (ce, None)

# =====================================================================================
# PRE-PASS: base CE + per-position MrxMr-deviation norm (firing mask for the term)
# =====================================================================================
print("PRE-PASS: base CE + term firing magnitudes ...", flush=True)
tno = []
ces = []
for i in range(0, S_, B0):
    ce, _ = fwd(HELD[i:i+B0], term_norm_out=tno)
    ces.append(ce.cpu())
base = torch.cat(ces, 0)
term_act = torch.cat(tno, 0).numpy()                    # (S,T)
print(f"base CE {float(base.mean()):.4f} (script-1 base {float(base_ce_s1.mean()):.4f})", flush=True)
term_mask = topmask(term_act)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

RES = json.load(open(f'{QK}/qk_arc_square.json'))

# =====================================================================================
# PART A: K=64 keep-side configs (full context: ablate the term, add back one part)
# =====================================================================================
print("PART A: K=64 keep-side configs ...", flush=True)
for name, part in [('keep_proj64_droprem', 'proj64'), ('keep_diag64_droprem', 'diag64'),
                   ('keep_cross64_droprem', 'cross64')]:
    out = []
    for i in range(0, S_, B0):
        ce, _ = fwd(HELD[i:i+B0], config={'ablate': True, 'part': part})
        out.append(ce.cpu())
    mn, se = dstat(torch.cat(out, 0))
    RES['configs'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:28s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
json.dump(RES, open(f'{QK}/qk_arc_square.json', 'w'), indent=1)

# =====================================================================================
# PART B: top-pair characterization (parents, snippets, causal class push of the pair)
# =====================================================================================
TOP_PAIRS = [(pe[0], pe[1]) for pe in PE[:3]]
PARENT_DIRS = sorted(set([d for p in TOP_PAIRS for d in p]))
print(f"PART B: pairs {TOP_PAIRS}, parent dirs {PARENT_DIRS}", flush=True)

def trigger_class_sig(act):
    mk = topmask(np.abs(act))
    fs, fp = np.where(mk)
    cur = held_np[fs, fp]
    classes = VOCAB_CLASS[cur]
    cnt = Counter(classes.tolist())
    toktop = Counter([dec(t) for t in cur]).most_common(8)
    return [(c, int(n)) for c, n in cnt.most_common(6)], toktop

parents = {}
for d_ in PARENT_DIRS:
    csig, ttop = trigger_class_sig(A_HELD[:, :, d_])
    parents[d_] = {'trigger_classes_top6': csig, 'trigger_tokens_top8': ttop}
    print(f"  dir{d_}: classes {csig} tokens {ttop[:5]}", flush=True)

# pair firing = |deviation of P_ij from its per-position mean| (the causal quantity's magnitude)
PMEAN_np = PMEAN.cpu().numpy()
pair_report = []
for (i_, j_) in TOP_PAIRS:
    pact = np.abs(A_HELD[:, :, i_]*A_HELD[:, :, j_]/RHO2_HELD - PMEAN_np[None, :, i_, j_])
    pmask = topmask(pact)
    # snippets: top-4 firing positions
    a1 = A_HELD[:, :, i_]; a2 = A_HELD[:, :, j_]
    flat = pact.copy(); flat[bad_trigger] = -1e30
    order = np.argsort(-flat.reshape(-1))[:4]
    snips = []
    for o in order:
        s, p = divmod(int(o), T_)
        lo = max(0, p-11)
        ctx = tok.decode(held_np[s, lo:p+1])
        snips.append({'seq': s, 'pos': p, 'context_ending_at_pos': ctx,
                      'current_token': dec(held_np[s, p]),
                      'a_parent1': round(float(a1[s, p]), 2), 'a_parent2': round(float(a2[s, p]), 2)})
    # causal class push: drop the pair, class-summed delta-logit at firing positions
    seqs = np.where(pmask.any(axis=1))[0]
    classsum = torch.zeros(len(CLASS_LIST), device=DEV)
    dlog_sum = torch.zeros(V, device=DEV); nf = 0
    dvals = []
    for bi in range(0, len(seqs), B0):
        sb = seqs[bi:bi+B0]; idx = HELD[sb]
        bce, blog = fwd(idx, want_logits=True)
        ace, alog = fwd(idx, config={'drop_pair': (i_, j_)}, want_logits=True)
        mk = torch.from_numpy(pmask[sb]).to(DEV)
        dl = (blog.float() - alog.float())[mk]           # base - dropped: what the pair pushes
        classsum += CMAT @ dl.sum(0); dlog_sum += dl.sum(0); nf += int(dl.shape[0])
        mkce = torch.from_numpy(pmask[sb, :T_-1]).to(DEV)
        dvals.append((ace - bce)[mkce].cpu().numpy())
        del blog, alog, dl
    cs = (classsum/max(nf, 1)).cpu().numpy()
    dm = (dlog_sum/max(nf, 1)).cpu().numpy()
    co = np.argsort(-np.abs(cs))
    boost = np.argsort(-dm)[:6]; supp = np.argsort(dm)[:6]
    dv = np.concatenate(dvals)
    tr_mn = float(dv.mean()); tr_se = float(dv.std(ddof=1)/math.sqrt(len(dv)))
    rec = {'pair': [int(i_), int(j_)], 'energy_share': next(e for a, b, e in PE if (a, b) == (i_, j_)),
           'parents': {str(i_): parents[i_], str(j_): parents[j_]},
           'trigger_dCE_of_pair_drop_at_firing': round(tr_mn, 5), 'trigger_dCE_SE': round(tr_se, 5),
           'class_summed_push_top6': {CLASS_LIST[c]: round(float(cs[c]), 4) for c in co[:6]},
           'top_boosted_tokens': [(dec(t), round(float(dm[t]), 4)) for t in boost],
           'top_suppressed_tokens': [(dec(t), round(float(dm[t]), 4)) for t in supp],
           'snippets': snips}
    pair_report.append(rec)
    print(f"  pair ({i_},{j_}): firing dCE {tr_mn:+.5f}+-{tr_se:.5f} push "
          f"{ {CLASS_LIST[c]: round(float(cs[c]),3) for c in co[:4]} }", flush=True)
RES['pair_characterization'] = pair_report
json.dump(RES, open(f'{QK}/qk_arc_square.json', 'w'), indent=1)

# =====================================================================================
# PART C: DOWNSTREAM ROUTING -- freeze-patching. Ablate MrxMr; freeze later components
# to clean cached values; unfreeze one mediator at a time.
# =====================================================================================
LATER = list(range(LI+1, NL))
ALL_FA = set(LATER); ALL_FM = set(LATER)
route_cfgs = [('sanity_freeze_all_no_ablate', {'ablate': False, 'freeze_attn': ALL_FA, 'freeze_mlp': ALL_FM}),
              ('total_damage', {'ablate': True}),
              ('direct_only', {'ablate': True, 'freeze_attn': ALL_FA, 'freeze_mlp': ALL_FM}),
              ('direct_plus_all_attn', {'ablate': True, 'freeze_attn': set(), 'freeze_mlp': ALL_FM}),
              ('freeze_only_mlp3', {'ablate': True, 'freeze_attn': set(), 'freeze_mlp': {3}})]
for li in LATER:
    route_cfgs.append((f'direct_plus_mlp{li}', {'ablate': True, 'freeze_attn': ALL_FA,
                                                'freeze_mlp': ALL_FM - {li}}))
print(f"PART C: {len(route_cfgs)} routing configs ...", flush=True)
acc = {name: [] for name, _ in route_cfgs}
classsum_direct = torch.zeros(len(CLASS_LIST), device=DEV)
classsum_total = torch.zeros(len(CLASS_LIST), device=DEV)
dlog_direct = torch.zeros(V, device=DEV); dlog_total = torch.zeros(V, device=DEV); nfire_t = 0
t0 = time.time()
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    cache = {}
    bce, blog = fwd(idx, fill_cache=True, cache=cache, want_logits=True)
    mk = torch.from_numpy(term_mask[i:i+B0]).to(DEV)
    for name, c in route_cfgs:
        want = name in ('direct_only', 'total_damage') and bool(mk.any())
        ce, lg = fwd(idx, config=dict(c, **{}), cache=cache, want_logits=want)
        acc[name].append(ce.cpu())
        if want:
            dl = (blog.float() - lg.float())[mk]
            if name == 'direct_only':
                classsum_direct += CMAT @ dl.sum(0); dlog_direct += dl.sum(0)
            else:
                classsum_total += CMAT @ dl.sum(0); dlog_total += dl.sum(0); nfire_t += int(dl.shape[0])
            del dl
        del lg
    del cache, blog
print(f"PART C forwards done ({time.time()-t0:.0f}s)", flush=True)
routing = {}
for name, _ in route_cfgs:
    mn, se = dstat(torch.cat(acc[name], 0))
    routing[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:26s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
direct = routing['direct_only']['dCE']; total = routing['total_damage']['dCE']
med = {}
for li in LATER:
    med[li] = round(routing[f'direct_plus_mlp{li}']['dCE'] - direct, 4)
routing['mediation_via_single_mlp_minus_direct'] = med
routing['direct_frac_of_total'] = round(direct/total, 4) if total else None
cs_d = (classsum_direct/max(nfire_t, 1)).cpu().numpy()
cs_t = (classsum_total/max(nfire_t, 1)).cpu().numpy()
dm_d = (dlog_direct/max(nfire_t, 1)).cpu().numpy(); dm_t = (dlog_total/max(nfire_t, 1)).cpu().numpy()
def sig(cs, dm):
    co = np.argsort(-np.abs(cs))
    return {'class_summed_top6': {CLASS_LIST[c]: round(float(cs[c]), 4) for c in co[:6]},
            'top_boosted_tokens': [(dec(t), round(float(dm[t]), 4)) for t in np.argsort(-dm)[:6]],
            'top_suppressed_tokens': [(dec(t), round(float(dm[t]), 4)) for t in np.argsort(dm)[:6]]}
routing['direct_readout_signature_at_term_firing'] = sig(cs_d, dm_d)
routing['total_signature_at_term_firing'] = sig(cs_t, dm_t)
RES['downstream_routing'] = routing
json.dump(RES, open(f'{QK}/qk_arc_square.json', 'w'), indent=1)
print("QK ARC SQUARE (script 2) DONE", flush=True)
