"""IS MLP1'S WRITE TOPIC-SPECIFIC OR A GENERIC SHARED SUBSTRATE? §931 found mean-ablating MLP1 collapses L15
topic-decode to base (drop 0.667). Caveat: mean-ablating a whole submodule removes its entire residual
contribution, so a big front writer could drop topic for GENERIC reasons (it writes most of the residual mass).
Disambiguate: for a few front components (MLP0-3), a mid control (MLP9), and ATTN1, mean-ablate and measure the
drop in BOTH the topic-decode (content channel) AND the next-token CLASS-decode (grammar channel), plus the
overall next-token CE increase. If MLP1 drops topic and class by COMPARABLE fractions -> it is a GENERIC shared
substrate writer (writes both machines' inputs, like mlp0 for class §915). If topic-drop >> class-drop -> it is
TOPIC-SPECIFIC.

REGISTERED PREDICTIONS:
  (0) SANITY: clean topic-decode ~0.84 and clean class-decode well above base; MLP1 ablation raises overall CE
      the most among the tested components; MLP9 (mid) ablation is small on both (front-loaded substrate).
  (a) GENERIC SHARED SUBSTRATE: ablating MLP1 (and MLP0) drops CLASS-decode and TOPIC-decode by COMPARABLE
      fractions of their clean-above-base range (both large) and raises CE a lot -> the front MLPs write a shared
      grammar+content substrate, not a topic-dedicated code;
  (b) report per-component topic-drop, class-drop (fraction of clean-above-base) and CE increase."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_component_specificity_results.json'
CONTENT_L = 15; NEVAL = 160; SEQ = 256; RTOK = 64; RPOS = 32; K = 32; RIDGE = 1e2; RCLASS = 8
COMPONENTS = [('mlp', 0), ('mlp', 1), ('mlp', 2), ('mlp', 3), ('mlp', 9), ('attn', 1)]
ABL = {'L': -1, 'comp': None, 'mean': None}
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


def mk_hook(kind, L):
    def h(mo, i_, o_):
        if ABL['L'] != L or ABL['comp'] != kind: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; mv = ABL['mean'][(kind, L)].to(y.dtype)
        ny = mv.view(1, 1, D).expand_as(y).clone()
        return (ny,) + tuple(o_[1:]) if isinstance(o_, tuple) else ny
    return h


def forward_all(idx):
    """return (L15 residual, logits)"""
    cap = {}
    def ch(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(ch)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r'], readout(x).float()


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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    tgt = S[:, 1:].reshape(-1)
    # pass 1: capture submodule means + clean L15 + clean CE
    sums = {(k, L): torch.zeros(D, device=DEV) for (k, L) in COMPONENTS}; cnt = 0
    def cap_hook(kind, L):
        def h(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_; sums[(kind, L)] += y.reshape(-1, D).sum(0)
        return h
    hs = []
    for (kind, L) in COMPONENTS:
        sub = getattr(m.transformer.h[L], kind); hs.append(sub.register_forward_hook(cap_hook(kind, L)))
    Rc = []; ce_c = []
    for i in range(0, nb, 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); r, lg = forward_all(idx)
        Rc.append(r.reshape(-1, D)); lp = F.log_softmax(lg, -1).reshape(-1, lg.shape[-1])
        tf = torch.tensor(tgt[i*(SEQ-1)//1:0] if False else S[i:i+4, 1:].reshape(-1), device=DEV)
        ce_c.append((-lp[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy()); cnt += idx.shape[0]*idx.shape[1]
    for h in hs: h.remove()
    ABL['mean'] = {k: v/cnt for k, v in sums.items()}
    Rc = torch.cat(Rc, 0); clean_ce = float(np.concatenate(ce_c).mean())
    # topic labels + probes (fixed from clean)
    Utok, g = mean_subspace(Rc, toks, RTOK); Upos, _ = mean_subspace(Rc, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    contentc = (Rc-g) - ((Rc-g)@Ucp)@Ucp.T; cn = contentc/(contentc.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy()
    n = Rc.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    def probe(Feat, y, ncls):
        Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
        A = Feat[tr].T @ Feat[tr] + RIDGE*torch.eye(Feat.shape[1], device=DEV); W = torch.linalg.solve(A, Feat[tr].T @ Y)
        return W
    Wt = probe(contentc, topic, K)
    clsmask = nxtcls >= 0
    Wc = probe(Rc, np.where(clsmask, nxtcls, 0), RCLASS)  # class from FULL residual
    def topic_acc(content): return float((content[te] @ Wt).argmax(1).cpu().numpy().__eq__(topic[te]).mean())
    def class_acc(R):
        pred = (R[te] @ Wc).argmax(1).cpu().numpy(); m2 = clsmask[te]; return float((pred[m2] == nxtcls[te][m2]).mean())
    topic_base = float(np.bincount(topic, minlength=K).max()/len(topic))
    class_base = float(np.bincount(nxtcls[clsmask], minlength=RCLASS).max()/clsmask.sum())
    clean_topic = topic_acc(contentc); clean_class = class_acc(Rc)
    # pass 2: ablate each component
    ah = []
    for (kind, L) in COMPONENTS:
        sub = getattr(m.transformer.h[L], kind); ah.append(sub.register_forward_hook(mk_hook(kind, L)))
    def run_ablated(kind, L):
        ABL['comp'] = kind; ABL['L'] = L; RR = []; ce = []
        for i in range(0, nb, 4):
            idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); r, lg = forward_all(idx)
            RR.append(r.reshape(-1, D)); lp = F.log_softmax(lg, -1).reshape(-1, lg.shape[-1])
            tf = torch.tensor(S[i:i+4, 1:].reshape(-1), device=DEV)
            ce.append((-lp[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
        ABL['comp'] = None; ABL['L'] = -1
        R = torch.cat(RR, 0); content = (R-g) - ((R-g)@Ucp)@Ucp.T
        rvar = float(((R - Rc).pow(2).sum(1).mean() / (Rc - Rc.mean(0)).pow(2).sum(1).mean()))
        return topic_acc(content), class_acc(R), float(np.concatenate(ce).mean()), rvar
    def frac(clean, abl, base): return round((clean - abl) / (clean - base + 1e-9), 4)
    out = {'clean_topic': round(clean_topic, 4), 'clean_class': round(clean_class, 4),
           'topic_base': round(topic_base, 4), 'class_base': round(class_base, 4), 'clean_ce': round(clean_ce, 4),
           'components': {}}
    for (kind, L) in COMPONENTS:
        ta, ca, ce, rvar = run_ablated(kind, L); name = f"{kind}{L}"
        out['components'][name] = {'topic_acc': round(ta, 4), 'class_acc': round(ca, 4),
                                   'topic_drop_frac': frac(clean_topic, ta, topic_base),
                                   'class_drop_frac': frac(clean_class, ca, class_base),
                                   'ce_increase': round(ce - clean_ce, 4), 'resid_var_removed': round(rvar, 4)}
        c = out['components'][name]
        print(f"{name:>6}: topic-drop {c['topic_drop_frac']:+.3f} class-drop {c['class_drop_frac']:+.3f} | CE +{c['ce_increase']:.3f} | residΔvar {c['resid_var_removed']:.3f}", flush=True)
    for h in ah: h.remove()
    m1 = out['components']['mlp1']
    out['pred_a_generic_substrate'] = bool(abs(m1['topic_drop_frac'] - m1['class_drop_frac']) < 0.25 and m1['class_drop_frac'] > 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"clean topic {clean_topic:.3f}(base {topic_base:.3f}) class {clean_class:.3f}(base {class_base:.3f})", flush=True)
    print(f"(a) MLP1 is generic shared substrate (topic-drop ~= class-drop, both large): {out['pred_a_generic_substrate']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
