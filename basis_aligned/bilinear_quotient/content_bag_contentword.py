"""DOES A CONTENT-WORD-WEIGHTED bag improve the benchmark's content term? §936 (content_bag_benchmark) tested an
ALL-TOKEN causal running-mean bag (recovery 0.052, WORSE than the K=1024 topic centroid 0.169). But §996
(content_bag_selectivity) showed the bag is predominantly a bag of CONTENT words (content-word context recovers 80%
of the far content vs 51% for function words). So a CONTENT-WORD-weighted bag should be a better content stand-in.
Same causal, held-out protocol as §936: keep the structured part (token+pos+class subspace) exact at L15, replace the
content part with a stand-in, run 16-17, measure loss recovery. Compare stand-ins:
  - CW-BAG-MAP: content_hat = (causal running mean over CONTENT-WORD positions only) @ M  (NEW, §996-motivated)
  - BAG-MAP: all-token causal running-mean bag @ M  (§936 reference, ~0.052)
  - CURRENT-TOKEN-MAP: current embedding @ M  (§936 ~0.222)
  - TOPIC-CENTROID K=1024  (§936 ~0.169)
  - SHUFFLED-CW-BAG null -> ~0.
All maps fit on TRAIN, evaluated on held-out rows.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-cw-bag null recovery ~0 (<0.03); all-token bag reproduces ~§936 (~0.05).
  (a) CW-BAG BEATS ALL-TOKEN BAG: the content-word-weighted bag recovers MORE content than the all-token bag
      (operationalizing §996) -> content-word weighting is the right bag;
  (b) report recovery for each; whether cw-bag also beats the topic centroid (0.169) and how close to current-token
      (0.222)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_bag_contentword_results.json'
CONTENT_L = 15; NEVAL = 400; SEQ = 256; RTOK = 64; RPOS = 32; RCLASS = 8; KTOPIC = 1024; RIDGE_MAP = 1e3
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
CONTENT_CLS = {'word', 'cap', 'number'}
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
def bagfeat(blocks_slice):
    """causal running mean of rms-normed embeddings, returned (b*T, D)."""
    idx = blocks_slice[:, :-1].contiguous().to(DEV)
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); T = E.shape[1]
    num = torch.cumsum(E, 1); den = torch.arange(1, T+1, device=DEV, dtype=E.dtype).view(1, T, 1)
    return (num/den).reshape(-1, D), E.reshape(-1, D)


@torch.no_grad()
def bagfeat_cw(blocks_slice, tok2content):
    """causal running mean of rms-normed embeddings over CONTENT-WORD positions only, (b*T, D)."""
    idx = blocks_slice[:, :-1].contiguous().to(DEV)
    E = F.rms_norm(m.transformer.wte(idx), (D,)).float(); T = E.shape[1]
    w = tok2content[idx].float().unsqueeze(-1)              # (b,T,1) 1 at content-word key positions
    num = torch.cumsum(w*E, 1); den = torch.cumsum(w, 1).clamp_min(1.0)
    return (num/den).reshape(-1, D)


@torch.no_grad()
def capture_content(blocks_set, Ustruct, g):
    R = []
    for i in range(0, blocks_set.shape[0], 4): R.append(capL(blocks_set[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(R, 0); struct = g + ((R-g)@Ustruct)@Ustruct.T
    return R, R - struct, struct


@torch.no_grad()
def loss_with(test_blocks, Ustruct, g, standin_fn):
    """standin_fn(bagfeat_te, curfeat_te, content_te, struct_te, i0) -> new_content (n_slice, D) or None(=clean)."""
    tot = []
    for i in range(0, test_blocks.shape[0], 4):
        bb = test_blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        R = capL(idx); b, T, _ = R.shape; Rf = R.reshape(-1, D)
        struct = g + ((Rf-g)@Ustruct)@Ustruct.T
        if standin_fn is None:
            SUB['on'] = False
        else:
            bf, cf = bagfeat(bb)
            new_content = standin_fn(bf, cf, Rf-struct, struct, i)
            SUB['newout'] = (struct + new_content).reshape(b, T, D); SUB['on'] = True
        lg = forward_logits(idx).float(); SUB['on'] = False
        lp = F.log_softmax(lg, -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        tot.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    return float(np.concatenate(tot).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]; ntr = int(0.6*nb); tr = blocks[:ntr]; te = blocks[ntr:]
    Str = tr.cpu().numpy()
    # structured subspace on train
    Rtr0 = []
    for i in range(0, ntr, 4): Rtr0.append(capL(tr[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    Rtr = torch.cat(Rtr0, 0)
    toks = Str[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (ntr, SEQ-1)).reshape(-1)
    nxtc = np.full_like(Str[:, :-1], -1); nxtc[:, :-1] = Str[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    V = int(m.lm_head.weight.shape[0]); t2c = np.zeros(V, bool)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): t2c[int(tid)] = classify(d(int(tid))) in CONTENT_CLS
    tok2content = torch.tensor(t2c, device=DEV)
    Utok, g = mean_subspace(Rtr, toks, RTOK); Upos, _ = mean_subspace(Rtr, pos.astype(np.int64), RPOS)
    Uclass, _ = mean_subspace(Rtr, nxtcls, RCLASS)
    Ustruct = torch.linalg.svd(torch.cat([Utok, Upos, Uclass], 1), full_matrices=False)[0][:, :RTOK+RPOS+RCLASS].contiguous()
    content_tr = Rtr - (g + ((Rtr-g)@Ustruct)@Ustruct.T)
    # train features
    bf_tr_list = []; cf_tr_list = []
    for i in range(0, ntr, 4):
        bf, cf = bagfeat(tr[i:i+4]); bf_tr_list.append(bf); cf_tr_list.append(cf)
    Bag_tr = torch.cat(bf_tr_list, 0); Cur_tr = torch.cat(cf_tr_list, 0)
    cwbf_tr = torch.cat([bagfeat_cw(tr[i:i+4], tok2content) for i in range(0, ntr, 4)], 0)
    def ridge_map(X, Ytar):
        A = X.T @ X + RIDGE_MAP*torch.eye(D, device=DEV); return torch.linalg.solve(A, X.T @ Ytar)
    Mbag = ridge_map(Bag_tr, content_tr); Mcur = ridge_map(Cur_tr, content_tr); Mcwbag = ridge_map(cwbf_tr, content_tr)
    # topic centroids (K=1024) on train content
    cntr = content_tr/(content_tr.norm(dim=1, keepdim=True)+1e-9); cen = kmeans_fit(cntr, KTOPIC, seed=0)
    assign_tr = torch.cdist(cntr, cen).argmin(1)
    raw_cent = torch.stack([content_tr[assign_tr==j].mean(0) if (assign_tr==j).any() else torch.zeros(D, device=DEV) for j in range(KTOPIC)], 0)
    hh = m.transformer.h[CONTENT_L].register_forward_hook(sub_hook)
    loss_full = loss_with(te, Ustruct, g, None)
    loss_ablate = loss_with(te, Ustruct, g, lambda bf, cf, ct, st, i: torch.zeros_like(ct))
    def rec(l): return round(float((loss_ablate - l)/(loss_ablate - loss_full + 1e-9)), 4)
    out = {'loss_full': round(loss_full, 4), 'loss_ablate': round(loss_ablate, 4), 'recovery': {}}
    # stand-ins
    out['recovery']['bag_map'] = rec(loss_with(te, Ustruct, g, lambda bf, cf, ct, st, i: bf @ Mbag))
    out['recovery']['current_token_map'] = rec(loss_with(te, Ustruct, g, lambda bf, cf, ct, st, i: cf @ Mcur))
    out['recovery']['cw_bag_map'] = rec(loss_with(te, Ustruct, g, lambda bf, cf, ct, st, i: bagfeat_cw(te[i:i+4], tok2content) @ Mcwbag))
    def shuffled_cwbag(bf, cf, ct, st, i):
        x = bagfeat_cw(te[i:i+4], tok2content); p = torch.randperm(x.shape[0], generator=torch.Generator(device=DEV).manual_seed(i+7), device=DEV); return x[p] @ Mcwbag
    out['recovery']['shuffled_cwbag_null'] = rec(loss_with(te, Ustruct, g, shuffled_cwbag))
    def centroid_standin(bf, cf, ct, st, i):
        cn = ct/(ct.norm(dim=1, keepdim=True)+1e-9); a = torch.cdist(cn, cen).argmin(1); return raw_cent[a]
    out['recovery']['topic_centroid_K1024'] = rec(loss_with(te, Ustruct, g, centroid_standin))
    def shuffled_bag(bf, cf, ct, st, i):
        p = torch.randperm(bf.shape[0], generator=torch.Generator(device=DEV).manual_seed(i+1), device=DEV); return bf[p] @ Mbag
    out['recovery']['shuffled_bag_null'] = rec(loss_with(te, Ustruct, g, shuffled_bag))
    hh.remove()
    for k, v in out['recovery'].items(): print(f"{k:>22}: recovery {v:+.4f}", flush=True)
    r = out['recovery']
    out['pred_0_null_ok'] = bool(r['shuffled_cwbag_null'] < 0.03)
    out['pred_a_cwbag_beats_alltoken_bag'] = bool(r['cw_bag_map'] > r['bag_map'])
    out['cwbag_beats_centroid'] = bool(r['cw_bag_map'] > r['topic_centroid_K1024'])
    out['cwbag_vs_current_token'] = round(r['cw_bag_map'] - r['current_token_map'], 4)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a cw-bag>all-token-bag {out['pred_a_cwbag_beats_alltoken_bag']} | cw-bag>centroid {out['cwbag_beats_centroid']} | cw-bag vs current-token {out['cwbag_vs_current_token']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
