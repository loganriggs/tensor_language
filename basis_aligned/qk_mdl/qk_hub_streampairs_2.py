"""Stream-pair hub decomposition, part 2: CLASS SIGNATURES of the top interaction terms.
For the top-energy terms (plus ExE for the bigram question): causally remove ONE term's deviation
(mo1 -> MEANF + sum_{j != k} (term_j - TMEAN_j), everything else intact) and read the class-summed
delta-logit (base - ablated) -- the paragraph-68 currency -- (a) at the term's top-200 firing
positions (largest deviation norm), (b) over all valid positions. Pushed class, concentration,
top-8 class movements. Extra for ExE: how much of the raw ExE term is a function of the CURRENT
token alone (bigram-table test): within-token variance ratio over repeated tokens.
Class library (lex1/VOCAB_CLASS) VERBATIM from qk_unsup_classpush.py; forward + stream machinery
from qk_hub_streampairs.py; harness lineage qk_hub_threshold.py. Held FW[448:600,:128], batch 6."""
import json, sys, time, subprocess, math
import numpy as np
import torch
import torch.nn.functional as F
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
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 6; LI = 1
S_, T_ = HELD.shape
b0, b1 = m.transformer.h[0], m.transformer.h[1]
L1 = b1.mlp.Left.weight.detach().float(); R1 = b1.mlp.Right.weight.detach().float()
D1w = b1.mlp.Down.weight.detach().float(); bias1 = b1.mlp.Down_bias.detach().float()
saved = torch.load(f'{QK}/qk_hub_streampairs_means.pt', map_location=DEV)
TMEAN = saved['TMEAN'].to(DEV); MEANF = saved['MEANF'].to(DEV)
PNAMES = saved['PNAMES']; PAIRS = saved['PAIRS']; lamE = saved['lamE']; lam0 = saved['lam0']
NT = len(PAIRS)
part1 = json.load(open(f'{QK}/qk_hub_streampairs.json'))
energy_rank = part1['energy_rank']
CANDS = []
for n in energy_rank[:5] + ['ExE']:
    if n not in CANDS: CANDS.append(n)
CIDXT = {n: k for k, n in enumerate(PNAMES)}
print(f"candidates (top-5 energy + ExE): {CANDS}", flush=True)

# ---------------- lexical class library VERBATIM from qk_unsup_classpush.py ----------------
tok = AutoTokenizer.from_pretrained('gpt2')
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
print(f"classes: {len(CLASS_LIST)}; special tokens masked: {len(SPECIAL)}", flush=True)

def pair_terms(E_, A0_, M0_, A1_, xpre):
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    Ss = [E_, A0_, M0_, A1_]
    PL = [s @ L1.T for s in Ss]; PR = [s @ R1.T for s in Ss]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ D1w.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, mode=None, drop=None, collect=None):
    """mode None: full model. mode 'drop': mo1 -> MEANF + sum_{j != drop}(term_j - TMEAN_j).
    collect dict: store per-term deviation norms ('devn') and raw ExE term ('exe'). Returns logits."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    a0c = m0c = None
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
        if li == 0: a0c, m0c = aout, mo
        if li == LI and (mode is not None or collect is not None):
            E_ = lamE * x0; A0_ = lam0 * a0c; M0_ = lam0 * m0c; A1_ = aout
            terms = pair_terms(E_, A0_, M0_, A1_, x)
            if collect is not None:
                dn = torch.stack([(terms[k] - TMEAN[k]).norm(dim=-1) for k in range(NT)])  # (10,B,T)
                collect['devn'].append(dn.cpu().numpy())
                collect['exe'].append(terms[CIDXT['ExE']].half().cpu())
            if mode == 'drop':
                new = MEANF.unsqueeze(0).expand(B, -1, -1).clone()
                for kk in range(NT):
                    if kk != drop: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            del terms
        x = x + mo
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)

# ---------------- PASS A: deviation norms per term + raw ExE term ----------------
print("PASS A: term deviation norms + raw ExE ...", flush=True)
col = {'devn': [], 'exe': []}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], collect=col)
DEVN = np.concatenate(col['devn'], axis=1)          # (10,S,T)
EXE = torch.cat(col['exe'], 0).reshape(-1, D)       # (S*T,D) raw ExE term, fp16 cpu
del col
held_np = HELD.cpu().numpy()
pos_t = np.tile(np.arange(T_), S_).reshape(S_, T_)
bad = (pos_t == 0) | np.isin(held_np, SPECIAL) | (pos_t >= T_-1)
KF = 200
fire_mask = {}
for n in CANDS:
    a = DEVN[CIDXT[n]].copy().reshape(-1); a[bad.reshape(-1)] = -1e30
    tk = np.argpartition(a, -KF)[-KF:]
    mk = np.zeros(S_*T_, bool); mk[tk] = True
    fire_mask[n] = mk.reshape(S_, T_)

# ---------------- ExE bigram-table test: within-token variance of the raw ExE term ----------------
tok_ids = held_np.reshape(-1)
exe = EXE.float()
gmean = exe.mean(0)
tot_var = float((exe - gmean).pow(2).sum())
uniq, counts = np.unique(tok_ids, return_counts=True)
within = 0.0; n_used = 0; covered = 0
for t in uniq[counts >= 5]:
    sel = np.where(tok_ids == t)[0]
    sub = exe[sel]; within += float((sub - sub.mean(0)).pow(2).sum()); n_used += 1; covered += len(sel)
# restrict total to the same positions for a fair ratio
sel_all = np.isin(tok_ids, uniq[counts >= 5])
sub_all = exe[np.where(sel_all)[0]]
tot_r = float((sub_all - sub_all.mean(0)).pow(2).sum())
exe_token_R2 = 1.0 - within/tot_r
print(f"ExE bigram test: current-token identity explains R^2={exe_token_R2:.4f} of raw ExE variance "
      f"({n_used} tokens with >=5 occurrences, {covered} positions)", flush=True)
del EXE, exe, sub_all

# ---------------- PASS B: class-summed delta-logit per candidate (drop-one, base - ablated) ----------------
tgt_all = HELD
res = {n: {'cs_fire': torch.zeros(len(CLASS_LIST), device=DEV), 'nf': 0,
           'cs_all': torch.zeros(len(CLASS_LIST), device=DEV), 'na': 0,
           'dce_s': 0.0, 'dce_sq': 0.0, 'dce_n': 0} for n in CANDS}
print(f"PASS B: {len(CANDS)} drop-one candidates x {math.ceil(S_/B0)} batches ...", flush=True)
t0 = time.time()
valid = ~bad
for bi, i in enumerate(range(0, S_, B0)):
    sb = slice(i, min(i+B0, S_))
    idx = HELD[sb]; b = idx.shape[0]
    base = fwd(idx).float()
    blp = F.log_softmax(base[:, :T_-1], -1)
    bce = -blp.gather(-1, tgt_all[sb][:, 1:].unsqueeze(-1)).squeeze(-1); del blp
    vmask = torch.from_numpy(valid[sb, :T_-1]).to(DEV)
    for n in CANDS:
        abl = fwd(idx, mode='drop', drop=CIDXT[n]).float()
        alp = F.log_softmax(abl[:, :T_-1], -1)
        ace = -alp.gather(-1, tgt_all[sb][:, 1:].unsqueeze(-1)).squeeze(-1); del alp
        dce = (ace - bce)[vmask]
        res[n]['dce_s'] += float(dce.sum()); res[n]['dce_sq'] += float((dce*dce).sum()); res[n]['dce_n'] += int(dce.numel())
        dl = (base[:, :T_-1] - abl[:, :T_-1])
        fm = torch.from_numpy(fire_mask[n][sb, :T_-1]).to(DEV)
        if fm.any():
            res[n]['cs_fire'] += CMAT @ dl[fm].sum(0); res[n]['nf'] += int(fm.sum())
        res[n]['cs_all'] += CMAT @ dl[vmask].sum(0); res[n]['na'] += int(vmask.sum())
        del abl, dl, dce
    if bi % 5 == 0: print(f"  batch {bi+1}/{math.ceil(S_/B0)}  {time.time()-t0:.0f}s", flush=True)
    del base, bce

out = {'meta': {'candidates': CANDS, 'K_fire': KF, 'held': 'FW[448:600,:128]',
                'currency': 'class-summed delta-logit (base - term-dropped), paragraph-68 style; '
                            'firing = top-200 positions by that term deviation norm',
                'exe_bigram_test_R2_current_token': round(exe_token_R2, 4)},
       'terms': {}}
for n in CANDS:
    r = res[n]
    csf = (r['cs_fire']/max(1, r['nf'])).cpu().numpy()
    csa = (r['cs_all']/max(1, r['na'])).cpu().numpy()
    def top8(cs):
        order = np.argsort(-np.abs(cs))
        pushed = next(j for j in order if CLASS_LIST[j] not in PUSH_EXCLUDE)
        conc = float(abs(cs[pushed])/max(1e-9, float(np.abs(cs).sum())))
        return ({CLASS_LIST[j]: round(float(cs[j]), 4) for j in order[:8]},
                CLASS_LIST[pushed], round(float(cs[pushed]), 4), round(conc, 4))
    d8f, pf, vf, cf = top8(csf); d8a, pa, va, ca = top8(csa)
    mn = r['dce_s']/r['dce_n']; se = math.sqrt(max(r['dce_sq']/r['dce_n']-mn*mn, 0)/r['dce_n'])
    out['terms'][n] = {'drop_one_dCE': round(mn, 4), 'drop_one_dCE_SE': round(se, 5),
                       'firing': {'pushed_class': pf, 'pushed_val': vf, 'concentration': cf, 'top8': d8f,
                                  'n_positions': r['nf']},
                       'all_positions': {'pushed_class': pa, 'pushed_val': va, 'concentration': ca, 'top8': d8a}}
    print(f"  {n:7s} drop-one dCE {mn:+.4f}±{se:.5f} | firing push {pf} {vf:+.3f} (conc {cf:.2f}) "
          f"top8 {d8f} | all-pos push {pa} {va:+.3f}", flush=True)
json.dump(out, open(f'{QK}/qk_hub_streampairs2.json', 'w'), indent=1)
print("QK HUB STREAMPAIRS 2 DONE", flush=True)
