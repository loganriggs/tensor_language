"""CLEAN DECOMPOSITION of the content term into LOCAL (per-token) + CROSS-TOKEN (topic/continuum), resolving
§936's caveat (the rank-64 token strip left per-token structure inside "content"). Causal held-out protocol at
L15: keep pos+class exact; decompose the rest (token+content) as:
  per_token_mean[tid] = held-out mean of (R - struct) for each token id (the FULL local per-token content), and
  remainder = (R - struct) - per_token_mean[tid]   (genuine CROSS-TOKEN content).
Stand-ins (substitute the (token+content) part, run 16-17, measure loss recovery = (ablate-standin)/(ablate-full)):
  A) per-token mean only;  B) per-token mean + topic-centroid(remainder, K=256);  C) topic-centroid of the whole
  (token+content) only (no per-token) as a reference.  NULL: shuffled per-token map (random token's mean).
This cleanly answers: how much of the content term is LOCAL per-token vs genuine CROSS-TOKEN (topic).

REGISTERED PREDICTIONS:
  (0) SANITY: ablate=recovery 0; shuffled per-token null ~0; per-token mean recovers well above the whole-topic
      reference C (local structure dominates, §936).
  (a) LOCAL DOMINATES: per-token-mean stand-in (A) recovers the bulk of the content term, and adding topic
      reconstruction of the remainder (B) adds only a SMALL gain (cross-token topic is a modest add-on) ->
      content term = mostly local per-token + a modest long-range topic slice;
  (b) report recovery A, B, C, the B-A gain (cross-token topic contribution), and the null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_pertoken_split_results.json'
CONTENT_L = 15; NEVAL = 400; SEQ = 256; RPOS = 32; RCLASS = 8; KREM = 256; RIDGE = 1e2
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
SUB = {'on': False, 'newout': None}


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(mo, i_, o_):
    if not SUB['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; ny = SUB['newout'].to(y.dtype)
    return (ny,) + tuple(o_[1:]) if isinstance(o_, tuple) else ny


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capL(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans_fit(X, k, iters=15, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return c


@torch.no_grad()
def loss_with(test_blocks, Ustruct, g, per_token_table=None, id2row=None, cen=None, cen_raw=None,
              use_pertoken=False, use_rem=False, use_whole_centroid=False, shuffle_pt=False, seed=0):
    tot = []; rng = np.random.RandomState(seed)
    for i in range(0, test_blocks.shape[0], 4):
        bb = test_blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        R = capL(idx); b, T, _ = R.shape; Rf = R.reshape(-1, D)
        struct = g + ((Rf-g)@Ustruct)@Ustruct.T; rest = Rf - struct
        if not (use_pertoken or use_whole_centroid):
            SUB['on'] = False
        else:
            toks_slice = bb[:, :-1].reshape(-1).cpu().numpy()
            rowi = torch.tensor([id2row.get(int(t), 0) for t in toks_slice], device=DEV)
            if shuffle_pt:
                rowi = torch.tensor(rng.randint(0, per_token_table.shape[0], size=rowi.shape[0]), device=DEV)
            new = torch.zeros_like(rest)
            if use_whole_centroid:
                cn = rest/(rest.norm(dim=1, keepdim=True)+1e-9); a = torch.cdist(cn, cen).argmin(1); new = cen_raw[a]
            else:
                pt = per_token_table[rowi]; new = pt.clone()
                if use_rem:
                    rem = rest - pt; cn = rem/(rem.norm(dim=1, keepdim=True)+1e-9)
                    a = torch.cdist(cn, cen).argmin(1); new = pt + cen_raw[a]
            SUB['newout'] = (struct + new).reshape(b, T, D); SUB['on'] = True
        lg = forward_logits(idx).float(); SUB['on'] = False
        lp = F.log_softmax(lg, -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        tot.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    return float(np.concatenate(tot).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]; ntr = int(0.6*nb); tr = blocks[:ntr]; te = blocks[ntr:]
    Str = tr.cpu().numpy()
    Rtr0 = []
    for i in range(0, ntr, 4): Rtr0.append(capL(tr[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    Rtr = torch.cat(Rtr0, 0)
    pos = np.broadcast_to(np.arange(SEQ-1), (ntr, SEQ-1)).reshape(-1)
    nxtc = np.full_like(Str[:, :-1], -1); nxtc[:, :-1] = Str[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Upos, g = mean_subspace(Rtr, pos.astype(np.int64), RPOS); Uclass, _ = mean_subspace(Rtr, nxtcls, RCLASS)
    Ustruct = torch.linalg.svd(torch.cat([Upos, Uclass], 1), full_matrices=False)[0][:, :RPOS+RCLASS].contiguous()
    rest_tr = Rtr - (g + ((Rtr-g)@Ustruct)@Ustruct.T)
    toks_tr = Str[:, :-1].reshape(-1)
    uids = np.unique(toks_tr); id2row = {int(t): i for i, t in enumerate(uids)}
    # per-token mean table (train)
    table = torch.zeros(len(uids), D, device=DEV)
    for ti, tid in enumerate(uids):
        mk = toks_tr == tid
        if mk.sum() > 0: table[ti] = rest_tr[torch.tensor(mk, device=DEV)].mean(0)
    pt_tr = table[torch.tensor([id2row[int(t)] for t in toks_tr], device=DEV)]
    rem_tr = rest_tr - pt_tr
    cn_rem = rem_tr/(rem_tr.norm(dim=1, keepdim=True)+1e-9); cen_rem = kmeans_fit(cn_rem, KREM, seed=0)
    a_rem = torch.cdist(cn_rem, cen_rem).argmin(1)
    cen_rem_raw = torch.stack([rem_tr[a_rem==j].mean(0) if (a_rem==j).any() else torch.zeros(D, device=DEV) for j in range(KREM)], 0)
    # whole-centroid reference (of rest, K=256)
    cn_rest = rest_tr/(rest_tr.norm(dim=1, keepdim=True)+1e-9); cen_w = kmeans_fit(cn_rest, KREM, seed=1)
    a_w = torch.cdist(cn_rest, cen_w).argmin(1)
    cen_w_raw = torch.stack([rest_tr[a_w==j].mean(0) if (a_w==j).any() else torch.zeros(D, device=DEV) for j in range(KREM)], 0)
    hh = m.transformer.h[CONTENT_L].register_forward_hook(sub_hook)
    loss_full = loss_with(te, Ustruct, g)
    loss_ablate = loss_with(te, Ustruct, g, per_token_table=table, id2row=id2row, cen=cen_rem, cen_raw=torch.zeros_like(cen_rem_raw), use_pertoken=True)  # per-token replaced by zeros? -> ablate = zeros
    # proper ablate: replace rest with zeros
    def loss_zero():
        tot = []
        for i in range(0, te.shape[0], 4):
            bb = te[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            R = capL(idx); b, T, _ = R.shape; Rf = R.reshape(-1, D); struct = g + ((Rf-g)@Ustruct)@Ustruct.T
            SUB['newout'] = struct.reshape(b, T, D); SUB['on'] = True
            lg = forward_logits(idx).float(); SUB['on'] = False
            lp = F.log_softmax(lg, -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
            tot.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
        return float(np.concatenate(tot).mean())
    loss_ablate = loss_zero()
    def rec(l): return round(float((loss_ablate - l)/(loss_ablate - loss_full + 1e-9)), 4)
    out = {'loss_full': round(loss_full, 4), 'loss_ablate': round(loss_ablate, 4), 'recovery': {}}
    out['recovery']['A_per_token_only'] = rec(loss_with(te, Ustruct, g, per_token_table=table, id2row=id2row, cen=cen_rem, cen_raw=cen_rem_raw, use_pertoken=True))
    out['recovery']['B_per_token_plus_topic_rem'] = rec(loss_with(te, Ustruct, g, per_token_table=table, id2row=id2row, cen=cen_rem, cen_raw=cen_rem_raw, use_pertoken=True, use_rem=True))
    out['recovery']['C_whole_centroid_K256'] = rec(loss_with(te, Ustruct, g, cen=cen_w, cen_raw=cen_w_raw, use_whole_centroid=True))
    out['recovery']['null_shuffled_pertoken'] = rec(loss_with(te, Ustruct, g, per_token_table=table, id2row=id2row, cen=cen_rem, cen_raw=cen_rem_raw, use_pertoken=True, shuffle_pt=True, seed=3))
    hh.remove()
    r = out['recovery']; out['cross_token_topic_gain_B_minus_A'] = round(r['B_per_token_plus_topic_rem'] - r['A_per_token_only'], 4)
    for k, v in r.items(): print(f"{k:>30}: {v:+.4f}", flush=True)
    print(f"cross-token topic gain (B-A): {out['cross_token_topic_gain_B_minus_A']:+.4f}", flush=True)
    out['pred_a_local_dominates'] = bool(r['A_per_token_only'] > r['C_whole_centroid_K256'] and out['cross_token_topic_gain_B_minus_A'] < r['A_per_token_only'] and r['null_shuffled_pertoken'] < 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) local per-token dominates, topic modest add-on: {out['pred_a_local_dominates']} | full {loss_full:.3f} ablate {loss_ablate:.3f}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
