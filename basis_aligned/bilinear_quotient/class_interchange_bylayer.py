"""Is the GRAMMAR/CLASS variable causally addressable across DEPTH (front onward), symmetric to content_interchange
_bylayer? Class is WRITTEN at the front (mlp0, §915) and its subspace ROTATES write->read across depth (§897), so
build the per-layer class subspace and patch it base<-source at L2, L8, L15, measuring whether the predicted
next-token CLASS shifts toward the source's class (the clean class metric, no confound — worked in §957-960).

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace patch leaves class->source at ~baseline (chance) at every layer.
  (a) CLASS CAUSAL FROM THE FRONT ONWARD: class-subspace interchange shifts predicted class toward the source
      (class->source > random) at L2, L8, and L15 -> the class variable is causally addressable across depth (once
      written at the front), not only at L15;
  (b) report class->source for class-patch vs random-patch per layer."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_interchange_bylayer_results.json'
NEVAL = 200; SEQ = 256; RCLASS = 8; QP = 200
LAYERS = [2, 8, 15]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
PATCH = {'on': False, 'vec': None, 'U': None, 'L': -1}


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


def patch_hook_factory(L):
    def h(mo, i_, o_):
        if not PATCH['on'] or PATCH['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; y = y.clone(); U = PATCH['U']
        b = y[:, QP, :]; y[:, QP, :] = b - (b @ U) @ U.T + PATCH['vec']
        return (y,) + tuple(o_[1:]) if isinstance(o_, tuple) else y
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capL(idx, L):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    gd = torch.Generator(device=DEV).manual_seed(0); Urand = torch.linalg.qr(torch.randn(D, RCLASS, generator=gd, device=DEV))[0]
    hooks = {L: m.transformer.h[L].register_forward_hook(patch_hook_factory(L)) for L in LAYERS}
    ld = {}
    for L in LAYERS:
        Rs = []
        for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), L).reshape(-1, D))
        R = torch.cat(Rs, 0); Uclass, _ = mean_subspace(R, nxtcls, RCLASS)
        qv = []  # per-seq QP residual in chunks (avoid full-batch lm_head OOM)
        for i in range(0, nb, 4): qv.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), L)[:, QP, :])
        ld[L] = {'Uclass': Uclass, 'qvec': torch.cat(qv, 0)}
    pairs = [(i, (i+nb//2) % nb) for i in range(nb)]
    def predclass(lg): return classify(d(int(lg.argmax())))
    out = {'QP': QP, 'by_layer': {}}
    for L in LAYERS:
        agg = {'class': 0, 'random': 0, 'n': 0}
        for (bi, si) in pairs:
            src_cls = classify(d(int(S[si, QP+1]))) if QP+1 < SEQ else 'word'
            idx = blocks[bi:bi+1, :SEQ].to(DEV)[:, :-1].contiguous(); src = ld[L]['qvec'][si]
            for name, U in [('class', ld[L]['Uclass']), ('random', Urand)]:
                PATCH['U'] = U; PATCH['vec'] = (src @ U) @ U.T; PATCH['L'] = L; PATCH['on'] = True
                lg = forward_logits(idx).float()[0, QP]; PATCH['on'] = False
                agg[name] += int(predclass(lg) == src_cls)
            agg['n'] += 1
        out['by_layer'][str(L)] = {'class_patch_to_source': round(agg['class']/agg['n'], 3),
                                   'random_patch_to_source': round(agg['random']/agg['n'], 3), 'n': agg['n']}
        print(f"L{L}: class-patch->source {out['by_layer'][str(L)]['class_patch_to_source']} | random {out['by_layer'][str(L)]['random_patch_to_source']}", flush=True)
    for h in hooks.values(): h.remove()
    out['pred_a_class_causal_across_depth'] = bool(all(out['by_layer'][str(L)]['class_patch_to_source'] > out['by_layer'][str(L)]['random_patch_to_source'] + 0.05 for L in LAYERS))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) class causally addressable across depth (L2,L8,L15 > random): {out['pred_a_class_causal_across_depth']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
