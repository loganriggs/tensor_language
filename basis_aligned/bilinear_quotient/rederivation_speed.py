"""DIRECTLY test §963's claim that class re-derives FAST (per-block) and content SLOWLY (long-range). Ablate the
class subspace, and separately the content subspace, at a fixed early layer (L2), then decode the respective
variable (next-class / topic) at INCREASING downstream layers (L3,L4,L6,L9,L12,L15). The recovery-vs-depth curve
shows the re-derivation SPEED: class should recover within a few blocks; content should recover gradually.

REGISTERED PREDICTIONS:
  (0) SANITY: at the ablation layer L2 itself the ablated variable is at/below base; clean decode at each layer is
      well above base.
  (a) CLASS FAST, CONTENT SLOW: after L2 ablation, class next-class decode recovers to near its clean value within
      ~1-3 layers (steep early rise); content topic decode recovers GRADUALLY over many layers (shallow rise,
      still climbing late) -> re-derivation speed differs (fast/local grammar vs slow/long-range content),
      confirming §963;
  (b) report per-layer recovered-fraction curves for class and content."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'rederivation_speed_results.json'
NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RCONTENT = 24; RCLASS = 8; RIDGE = 1e2; L_ABL = 2
DECODE_LS = [2, 3, 4, 6, 9, 12, 15]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
ABL = {'on': False, 'U': None}


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


def abl_hook(mo, i_, o_):
    if not ABL['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; U = ABL['U']; v = y.reshape(-1, D)
    v2 = v - (v @ U) @ U.T
    return (v2.reshape(sh),) + tuple(o_[1:]) if isinstance(o_, tuple) else v2.reshape(sh)


def forward_capture(idx, layers):
    cap = {}
    hs = []
    for L in layers:
        def mk(L):
            def h(mo, i_, o_): cap[L] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    for h in hs: h.remove()
    return cap


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def content_of(R, toks, pos):
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    return (R-g) - ((R-g)@Ucp)@Ucp.T


def acc(F_, y, ncls, tr, te):
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = F_[tr].T @ F_[tr] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[tr].T @ Y)
    return float((F_[te] @ W).argmax(1).cpu().numpy().__eq__(y[te]).mean())


@torch.no_grad()
def capture_all(layers):
    caps = {L: [] for L in layers}
    for i in range(0, NB, 4):
        c = forward_capture(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), layers)
        for L in layers: caps[L].append(c[L])
    return {L: torch.cat(caps[L], 0) for L in layers}


@torch.no_grad()
def main():
    global blocks, NB
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); NB = S.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (NB, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    ycls = np.where(nxtcls >= 0, nxtcls, 0)
    hook = m.transformer.h[L_ABL].register_forward_hook(abl_hook)
    # clean captures
    ABL['on'] = False; clean = capture_all(DECODE_LS)
    # per-layer subspaces + probes from clean
    n = clean[15].shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    cls_base = float(np.bincount(ycls[nxtcls>=0], minlength=RCLASS).max()/ (nxtcls>=0).sum())
    topic_all = {}; Ucls_all = {}; Ucont_all = {}
    for L in DECODE_LS:
        Ucls_all[L], _ = mean_subspace(clean[L], nxtcls, RCLASS)
        cont = content_of(clean[L], toks, pos); cn = cont/(cont.norm(dim=1,keepdim=True)+1e-9)
        topic_all[L] = kmeans(cn, K).cpu().numpy(); Ucont_all[L], _ = mean_subspace(cont, topic_all[L], RCONTENT)
    top_base = float(np.bincount(topic_all[15], minlength=K).max()/len(topic_all[15]))
    clean_cls = {L: acc(clean[L], ycls, RCLASS, tr, te) for L in DECODE_LS}
    clean_top = {L: acc(content_of(clean[L], toks, pos), topic_all[L], K, tr, te) for L in DECODE_LS}
    # ablate CLASS at L2, decode class downstream
    ABL['on'] = True; ABL['U'] = Ucls_all[L_ABL]; abl_c = capture_all([L for L in DECODE_LS if L >= L_ABL]); ABL['on'] = False
    # ablate CONTENT at L2, decode topic downstream
    ABL['on'] = True; ABL['U'] = Ucont_all[L_ABL]; abl_t = capture_all([L for L in DECODE_LS if L >= L_ABL]); ABL['on'] = False
    hook.remove()
    out = {'cls_base': round(cls_base,4), 'top_base': round(top_base,4), 'L_ablate': L_ABL, 'class_recovery': {}, 'content_recovery': {}}
    for L in DECODE_LS:
        if L < L_ABL: continue
        ca = acc(abl_c[L], ycls, RCLASS, tr, te); rec_c = (ca - cls_base)/(clean_cls[L]-cls_base+1e-9)
        ta = acc(content_of(abl_t[L], toks, pos), topic_all[L], K, tr, te); rec_t = (ta - top_base)/(clean_top[L]-top_base+1e-9)
        out['class_recovery'][str(L)] = {'acc': round(ca,3), 'clean': round(clean_cls[L],3), 'recovered_frac': round(float(rec_c),3)}
        out['content_recovery'][str(L)] = {'acc': round(ta,3), 'clean': round(clean_top[L],3), 'recovered_frac': round(float(rec_t),3)}
        print(f"L{L:>2}: class recovered {rec_c:+.3f} (acc {ca:.3f}/clean {clean_cls[L]:.3f}) | content recovered {rec_t:+.3f} (acc {ta:.3f}/clean {clean_top[L]:.3f})", flush=True)
    # speed: layer at which recovery first exceeds 0.5
    def first_half(rec):
        for L in DECODE_LS:
            if L >= L_ABL and rec[str(L)]['recovered_frac'] >= 0.5: return L
        return None
    out['class_half_recover_layer'] = first_half(out['class_recovery'])
    out['content_half_recover_layer'] = first_half(out['content_recovery'])
    out['pred_a_class_faster'] = bool(out['class_half_recover_layer'] is not None and (out['content_half_recover_layer'] is None or out['class_half_recover_layer'] <= out['content_half_recover_layer']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"class reaches 50% recovery by L{out['class_half_recover_layer']}, content by L{out['content_half_recover_layer']}", flush=True)
    print(f"(a) class re-derives faster than content: {out['pred_a_class_faster']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
