"""CAUSAL VERIFICATION of a SECOND set of unsupervised path-discovery candidates
(qk_unsup_discover.py / qk_unsup_paths.json) in bilin18.  Sibling agent verifies the
first set; THIS verifies: mlp.L1.d3, mlp.L0.d1, h.L3.3, h.L6.0 (novel candidates) +
h.L8.3 (known-recovered date successor, validation).

Held-out slice FW[448:600,:128] -- untouched by discovery.  MLP directions are the
SAME eigenvectors of the block MLP-output gram computed on the DISCOVERY TRAIN slice
FW[0:256,:128] (direction indices are defined by training), then applied out-of-sample.

Per path:
 (1) TRIGGER out-of-sample: top-K |activation| positions on held-out -> current-token
     (+ attended-source for heads) composition; selectivity ratio |act|(trigger)/|act|(rest).
 (2) CAUSAL: mean-ablate the path ON trigger-matching positions (set the path's coefficient
     to its in-distribution mean -- MLP: proj->mean along the unit dir; head: yh[:,h]->mean yh).
     Read next-token logits at those positions baseline vs ablated. Report the tokens whose
     logit DROPS most, dLogit on the discovered boosted set vs a random control set (paired SE),
     and CE increase on positions whose ACTUAL next token is in the boosted set.
 (3) verdict: genuine trigger-gated computation vs diffuse prior vs artifact.

Forward pass copied VERBATIM from qk_bracket_patch.py (tier2_model.reference_forward).
"""
import json, sys, math
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
tok = AutoTokenizer.from_pretrained('gpt2')
LMW = m.lm_head.weight
N_SVD = 4

FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQL = 128
TRAIN = FW[0:256, :SEQL].to(DEV)          # discovery slice -> defines MLP directions
TEST = FW[448:600, :SEQL].to(DEV)         # held-out
BATCH = 4
def dec(t): return repr(tok.decode([int(t)]))

# special/degenerate tokens (same rule as discovery)
_special = {tok.eos_token_id}
for _t in range(V):
    if '�' in tok.decode([_t]): _special.add(_t)
SPECIAL = np.array(sorted(_special))


@torch.no_grad()
def forward(idx, ablate=None, mean_yh=None, mlp_dir=None, mean_proj=None,
            pos_mask=None, capture_head=None, capture_mlpdir=None):
    """VERBATIM bilin18 forward.
    ablate: None | ('head',li,h) | ('mlpdir',li) with mlp_dir(unit,D)+mean_proj(scalar);
            applied ONLY where pos_mask[b,t] is True (mean-ablation).
    capture_head=(li,h): return per-pos head-output norm + argmax|attn| source.
    capture_mlpdir=(li,dir_unit): return per-pos projection of block-li MLP output onto dir.
    Returns logits (B,T,V) plus any captured arrays."""
    B, Tt = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
    cap = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn; hc = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hc).view(B, Tt, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)          # (B,Tt,NH,HD)
        if capture_head is not None and capture_head[0] == li:
            h = capture_head[1]
            Wr = a.c_proj.weight.view(D, NH, HD)
            comp = torch.einsum('btc,oc->bto', yh[:, :, h], Wr[:, h])  # (B,Tt,D)
            cap['hnorm'] = comp.norm(dim=-1)                  # (B,Tt)
            cap['src'] = pat[:, h].abs().argmax(-1)           # (B,Tt) key pos
        if ablate is not None and ablate[0] == 'head' and ablate[1] == li:
            h = ablate[2]; yh = yh.clone()
            repl = mean_yh.to(yh.dtype).view(1, 1, HD).expand(B, Tt, HD)
            yh[:, :, h] = torch.where(pos_mask.unsqueeze(-1), repl, yh[:, :, h])
        x = x + a.c_proj(yh.reshape(B, Tt, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if capture_mlpdir is not None and capture_mlpdir[0] == li:
            cap['proj'] = torch.einsum('btd,d->bt', mo, capture_mlpdir[1])   # (B,Tt)
        if ablate is not None and ablate[0] == 'mlpdir' and ablate[1] == li:
            proj = torch.einsum('btd,d->bt', mo, mlp_dir)     # (B,Tt)
            delta = torch.where(pos_mask, mean_proj - proj, torch.zeros_like(proj))
            mo = mo + delta.unsqueeze(-1) * mlp_dir.view(1, 1, D)
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return logits, cap


# ---- MLP directions from TRAIN gram (identical to discovery) ----
@torch.no_grad()
def train_mlp_dirs():
    gram = [torch.zeros(D, D, device=DEV) for _ in range(NL)]
    for i in range(0, TRAIN.shape[0], BATCH):
        idx = TRAIN[i:i+BATCH]; B, Tt = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            a = blk.attn; hc = F.rms_norm(x, (D,))
            def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hc).view(B, Tt, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, kk_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, kk_)/HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh.reshape(B, Tt, -1))
            mo = blk.mlp(F.rms_norm(x, (D,)))
            gram[li] += torch.einsum('btd,bte->de', mo, mo)
            x = x + mo
    dirs = torch.zeros(NL, N_SVD, D, device=DEV)
    for li in range(NL):
        _, evecs = torch.linalg.eigh(gram[li])
        dirs[li] = evecs[:, -N_SVD:].T.flip(0)
    return dirs

print("computing MLP directions from TRAIN gram...", flush=True)
MLP_DIRS = train_mlp_dirs()


def effect_topk(dir_unit, n=16):
    logits = (LMW @ dir_unit).float(); logits = logits - logits.mean()
    top = torch.topk(logits, n + len(SPECIAL) + 5)
    ids = [int(i) for i in top.indices.cpu().numpy() if int(i) not in set(SPECIAL.tolist())][:n]
    return ids


def conc_ids(ids):
    ids = np.asarray(ids)
    if len(ids) <= 1: return 0.0
    _, c = np.unique(ids, return_counts=True); p = c / c.sum()
    return float(1.0 - (-(p*np.log(p)).sum()) / math.log(len(ids)))

# ---- held-out token bookkeeping ----
test_cur = TEST.cpu().numpy().reshape(-1)                 # current token per flat pos
NT = TEST.shape[0]; Npos = NT * SEQL
pos_t = np.tile(np.arange(SEQL), NT)
is_special_test = np.isin(test_cur, SPECIAL)
# actual next token (shift within sequence); last col invalid
nxt = np.full(Npos, -1, dtype=np.int64)
tc2 = TEST.cpu().numpy()
nxt.reshape(NT, SEQL)[:, :-1] = tc2[:, 1:]

rng = np.random.RandomState(0)
CTRL = np.array([t for t in rng.choice(V, 400, replace=False) if t not in set(SPECIAL.tolist())][:200])

K_TRIG = 50           # top-K for out-of-sample trigger characterisation (matches discovery)
N_CAUSAL = 400        # fallback: top-|act| positions when no token trigger given
N_CAP = 3000          # cap on token-matched causal positions (highest |act| kept)


@torch.no_grad()
def capture_activation(path):
    """return per-flat-pos activation array + (heads) source token array + mean stats."""
    act = np.zeros(Npos, dtype=np.float32)
    src = np.zeros(Npos, dtype=np.int64)
    if path['kind'] == 'head':
        ch = (path['li'], path['h']); cm = None
        sum_yh = torch.zeros(HD, device=DEV); cnt = 0
    else:
        du = MLP_DIRS[path['li'], path['k']]; du = du / (du.norm()+1e-9)
        ch = None; cm = (path['li'], du)
        sum_proj = 0.0; cnt = 0
    for i in range(0, NT, BATCH):
        idx = TEST[i:i+BATCH]; B = idx.shape[0]
        f0 = i*SEQL; f1 = f0 + B*SEQL
        if path['kind'] == 'head':
            _, cap = forward(idx, capture_head=ch)
            act[f0:f1] = cap['hnorm'].reshape(-1).float().cpu().numpy()
            srcpos = cap['src'].cpu().numpy()                 # (B,Tt) key idx
            src[f0:f1] = np.take_along_axis(idx.cpu().numpy(), srcpos, axis=1).reshape(-1)
            # mean yh for the head over all positions
            # recompute yh mean cheaply from a dedicated capture (reuse forward once)
        else:
            _, cap = forward(idx, capture_mlpdir=cm)
            pr = cap['proj'].reshape(-1).float().cpu().numpy()
            act[f0:f1] = pr; sum_proj += pr.sum(); cnt += pr.size
    stat = {}
    if path['kind'] != 'head':
        stat['mean_proj'] = float(sum_proj / cnt)
        stat['dir_unit'] = du
    return act, src, stat


@torch.no_grad()
def mean_head_yh(li, h):
    s = torch.zeros(HD, device=DEV); n = 0
    for i in range(0, NT, BATCH):
        idx = TEST[i:i+BATCH]; B, Tt = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(Tt, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tt, Tt, device=DEV, dtype=torch.bool))
        for L in range(li+1):
            blk = m.transformer.h[L]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            a = blk.attn; hc = F.rms_norm(x, (D,))
            def qk(l): z = F.rms_norm(l(hc).view(B, Tt, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            vv = a.c_v(hc).view(B, Tt, NH, HD)
            if v1 is None: v1 = vv
            vv = (1-a.lamb)*vv + a.lamb*v1.view_as(vv)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh = torch.einsum('bhqk,bkhd->bqhd', pat, vv)
            if L == li:
                s += yh[:, :, h].reshape(-1, HD).sum(0); n += B*Tt
                break
            x = x + a.c_proj(yh.reshape(B, Tt, -1))
            x = x + blk.mlp(F.rms_norm(x, (D,)))
    return s / n


def verify(path, trig_tokset=None):
    r = {'comp': path['comp'], 'kind': path['kind'],
         'discovered_boost': path['boost'], 'discovered_trigger': path['trig']}
    act, src, stat = capture_activation(path)
    aa = np.abs(act).copy(); aa[(pos_t == 0) | is_special_test] = -1.0
    tk = np.argpartition(aa, -K_TRIG)[-K_TRIG:]
    cur = test_cur[tk]
    r['heldout_top50_cur'] = [(dec(t), int(c)) for t, c in
        sorted(zip(*np.unique(cur, return_counts=True)), key=lambda z: -z[1])[:6]]
    r['heldout_cur_purity'] = round(conc_ids(cur), 3)
    if path['kind'] == 'head':
        srct = src[tk]
        r['heldout_top50_src'] = [(dec(t), int(c)) for t, c in
            sorted(zip(*np.unique(srct, return_counts=True)), key=lambda z: -z[1])[:6]]
        r['heldout_src_purity'] = round(conc_ids(srct), 3)

    # ---- causal trigger positions ----
    valid = (pos_t != 0) & ~is_special_test
    glob_med = float(np.median(np.abs(act[valid])) + 1e-9)
    if path.get('trig_cur_ids') or path.get('trig_src_ids'):
        match = valid.copy()
        if path.get('trig_cur_ids'):
            match &= np.isin(test_cur, np.array(path['trig_cur_ids']))
        if path.get('trig_src_ids'):
            match &= np.isin(src, np.array(path['trig_src_ids']))
        cand = np.where(match)[0]
        r['n_token_matched'] = int(cand.size)
        # firing rate among token-matched: |act| above global median
        r['firing_rate_matched'] = round(float((np.abs(act[cand]) > glob_med).mean()), 3) if cand.size else 0.0
        if cand.size > N_CAP:                                  # keep highest-|act| matches
            cand = cand[np.argsort(-np.abs(act[cand]))[:N_CAP]]
        causal_idx = cand
    else:
        causal_idx = np.argpartition(aa, -N_CAUSAL)[-N_CAUSAL:]
        causal_idx = causal_idx[aa[causal_idx] > 0]
        r['n_token_matched'] = None; r['firing_rate_matched'] = None
    r['selectivity_trig/median'] = round(float(np.abs(act[causal_idx]).mean() / glob_med), 2)

    # boosted-token set: clean interpretable set from config (discovery linear-proxy words)
    if path['kind'] == 'head':
        myh = mean_head_yh(path['li'], path['h'])
    boost_ids = np.array(path['boost_ids'])
    r['boost_ids_tested'] = [dec(t) for t in boost_ids]
    alt_ids = np.array(path['alt_boost_ids']) if path.get('alt_boost_ids') else None
    if alt_ids is not None:
        r['alt_boost_name'] = path['alt_boost_name']
        r['alt_boost_ids_tested'] = [dec(t) for t in alt_ids]

    # build per-batch boolean pos_mask for causal positions
    causal_set = set(causal_idx.tolist())
    nca = len(causal_idx)

    # gather baseline + ablated logits at causal positions
    def run(ablate_kw):
        outB = []
        for i in range(0, NT, BATCH):
            idx = TEST[i:i+BATCH]; B = idx.shape[0]; f0 = i*SEQL
            flat = np.arange(f0, f0 + B*SEQL)
            sel = np.isin(flat, causal_idx)
            if not sel.any():
                continue
            pm = torch.from_numpy(sel.reshape(B, SEQL)).to(DEV)
            lg, _ = forward(idx, pos_mask=pm, **ablate_kw)
            lg = lg.reshape(B*SEQL, V)
            outB.append((flat[sel], lg[torch.from_numpy(sel).to(DEV)].float().cpu().numpy()))
        pos = np.concatenate([o[0] for o in outB]); L = np.concatenate([o[1] for o in outB])
        order = np.argsort(pos)
        return pos[order], L[order]

    if path['kind'] == 'head':
        abl = {'ablate': ('head', path['li'], path['h']), 'mean_yh': myh}
    else:
        abl = {'ablate': ('mlpdir', path['li']), 'mlp_dir': stat['dir_unit'],
               'mean_proj': torch.tensor(stat['mean_proj'], device=DEV)}
    posb, Lbase = run({})
    posa, Labl = run(abl)
    assert np.array_equal(posb, posa)
    dLog = Labl - Lbase                                    # (nca, V) change in logit
    # tokens whose logit drops most (averaged over causal positions)
    mean_dLog = dLog.mean(0)
    most_dropped = np.argsort(mean_dLog)[:12]
    r['most_dropped_tokens'] = [(dec(t), round(float(mean_dLog[t]), 3)) for t in most_dropped]
    r['boost_in_top12_dropped'] = int(len(set(most_dropped.tolist()) & set(boost_ids.tolist())))
    # paired dLogit on boost set vs control set
    dboost = dLog[:, boost_ids].mean(1)                   # per-pos mean over boost tokens
    dctrl = dLog[:, CTRL].mean(1)
    def pse(v): return round(float(v.std(ddof=1) / math.sqrt(len(v))), 4)
    r['dLogit_boost_mean'] = round(float(dboost.mean()), 4)
    r['dLogit_boost_se'] = pse(dboost)
    r['dLogit_control_mean'] = round(float(dctrl.mean()), 4)
    r['dLogit_control_se'] = pse(dctrl)
    r['dLogit_allvocab_mean'] = round(float(mean_dLog.mean()), 4)
    r['n_causal'] = int(nca)
    # specificity: boost drop vs control drop, in SE units
    diff = dboost - dctrl
    r['boost_minus_control_mean'] = round(float(diff.mean()), 4)
    r['boost_minus_control_se'] = pse(diff)
    r['boost_minus_control_z'] = round(float(diff.mean()/(diff.std(ddof=1)/math.sqrt(len(diff))+1e-12)), 2)

    # optional alternate boost set (e.g. capitalized sentence-openers for h.L6.0)
    if alt_ids is not None:
        dalt = dLog[:, alt_ids].mean(1); da = dalt - dctrl
        r['alt_dLogit_mean'] = round(float(dalt.mean()), 4)
        r['alt_minus_control_mean'] = round(float(da.mean()), 4)
        r['alt_minus_control_z'] = round(float(da.mean()/(da.std(ddof=1)/math.sqrt(len(da))+1e-12)), 2)

    # CE test on positions whose ACTUAL next token is in the boost set
    nxt_causal = nxt[posb]
    inboost = np.isin(nxt_causal, boost_ids)
    r['n_causal_next_in_boost'] = int(inboost.sum())
    if inboost.sum() >= 3:
        lp_base = torch.log_softmax(torch.from_numpy(Lbase[inboost]), -1).numpy()
        lp_abl = torch.log_softmax(torch.from_numpy(Labl[inboost]), -1).numpy()
        tgt = nxt_causal[inboost]
        ce_b = -lp_base[np.arange(len(tgt)), tgt]
        ce_a = -lp_abl[np.arange(len(tgt)), tgt]
        d = ce_a - ce_b
        r['CE_increase_on_actual_boost_next'] = round(float(d.mean()), 4)
        r['CE_increase_se'] = pse(d)
    else:
        r['CE_increase_on_actual_boost_next'] = None
    return r


# ---- path configs (discovered trigger + boost, from qk_unsup_paths.json) ----
DAYS = [352, 362, 513, 604, 642, 718, 767, 807, 860, 838, 1367, 1105, 1511, 1478, 1315, 1467, 1596, 1248, 678, 1160, 2310, 2534, 2242, 1987, 1679, 2608, 2681, 2579, 2808, 1542, 3261]
CONJ = [290, 393, 11, 5, 284, 12, 14, 355]                 # and,or,',','&',to,-,/,as
SENTEND = [13, 198, 26, 30, 0]                             # . \n ; ? !
DET = [262, 257, 281]                                      # the a an
PATHS = [
    dict(comp='mlp.L1.d3', kind='mlp', li=1, k=3,
         trig="' the'/' an' (determiner)", boost="whole/entire/aforementioned/usual/particular",
         trig_cur_ids=DET,
         boost_ids=[2187, 2104, 21818, 20794, 6678, 1948, 1459, 3297]),   # whole,entire,entirety,aforementioned,usual,particular,current,sort
    dict(comp='mlp.L0.d1', kind='mlp', li=0, k=1,
         trig="' the'/' a' (determiner)", boost="various/different/several/recent/latest",
         trig_cur_ids=DET,
         boost_ids=[2972, 1180, 1811, 2274, 3452, 15064, 584, 3294]),     # various,different,several,recent,latest,newer,other,multiple
    dict(comp='h.L3.3', kind='head', li=3, h=3,
         trig="digits/',' attending conjunction ' and'/' or'/','", boost="alike/respectively/etc",
         trig_src_ids=CONJ,
         boost_ids=[12936, 8148, 3503, 7927, 4232]),      # alike,respectively,etc,vice,whatever
    dict(comp='h.L6.0', kind='head', li=6, h=0,
         trig="sentence-end . \\n ; ? !", boost="therefore/also/but/or/so (lowercase connectives)",
         trig_cur_ids=SENTEND,
         boost_ids=[4361, 635, 475, 393, 523, 290, 2158, 4145],   # therefore,also,but,or,so,and,however,thus
         alt_boost_name="capitalized sentence-openers (Lastly/Finally/Similarly/Additionally...)",
         alt_boost_ids=[37511, 11158, 28039, 23216, 6610, 4864, 26583, 24606, 24951, 7583, 19093, 10294, 13193]),
    dict(comp='h.L8.3', kind='head', li=8, h=3,
         trig="attending a day-number source ' 1'..' 31'", boost="18/15/13/17 (nearby day-numbers)",
         trig_src_ids=DAYS,
         boost_ids=[1596, 1248, 1315, 1511, 1478, 1467, 1105, 678]),  # 17,18,15,13,14,16,12,19
]

results = []
for p in PATHS:
    print(f"\n=== verifying {p['comp']} ===", flush=True)
    r = verify(p)
    results.append(r)
    print(json.dumps(r, indent=1), flush=True)

json.dump({'meta': {'held_out': 'FW[448:600,:128]', 'model': 'bilin18',
                    'mlp_dirs_from': 'TRAIN gram FW[0:256,:128]'},
           'results': results},
          open(f'{QK}/qk_unsup_verify2.json', 'w'), indent=2)
print("\nSAVED qk_unsup_verify2.json", flush=True)
print("QK UNSUP VERIFY2 DONE", flush=True)
