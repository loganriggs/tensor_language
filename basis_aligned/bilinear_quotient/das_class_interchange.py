"""VARIABLE-LEVEL causal abstraction, step 1: is the grammatical-CLASS variable a real causal variable,
verified by INTERCHANGE INTERVENTION (Geiger-style), and can we identify its subspace from WEIGHTS? (user
redirect: label the VARIABLES a component reads/writes — not the head — using weight info, then intervene on
the variable and check predictable downstream change.)

Interchange intervention (≠ mean-steering, which was weak §888 because it pushed a lone read-direction):
run a BASE input and a SOURCE input; at layer L, replace ONLY the class-subspace coordinates of the base's
residual (at a fixed query position) with the SOURCE's — patching the real coordinated activation, respecting
read+write jointly. If the model's predicted next-token CLASS flips to the SOURCE's class, that subspace
REALIZES the class variable causally (interchange-intervention accuracy, IIA).

Three subspace candidates for the SAME variable:
  - ACTIVATION-derived: class-conditional-mean subspace of the layer-L residual (rank 7).
  - WEIGHT-derived: class-write subspace from lm_head (group vocab by class, mean unembedding row per class,
    SVD → the directions the readout uses to SEPARATE classes) — a weights-only guess at the variable.
  - RANDOM subspace of the same rank (null).

REGISTERED PREDICTIONS:
  (0) SANITY: no-patch baseline keeps the base class (flip rate ~ chance of coincidental class match);
  (a) CLASS IS A CAUSAL VARIABLE via interchange: patching the ACTIVATION class-subspace flips the predicted
      class to the SOURCE's far above the random-subspace null and above no-patch (IIA >> null) — even though
      mean-steering was weak, interchange works;
  (b) WEIGHTS IDENTIFY THE VARIABLE: the WEIGHT-derived (lm_head) class-subspace gives IIA comparable to the
      activation-derived one -> the variable's direction is readable from weights alone (user's point that
      weights and activations are the same type);
  (c) if activation-subspace IIA ~ random null, class is not a low-rank causal variable at this layer
      (report plainly)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'das_class_interchange_results.json'
NEVAL = 480; SEQ = 256; QP = 128; PATCH_L = 15; RANK = 7
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
PATCH = {'on': False, 'U': None, 'src': None}   # src: (B, D) source projections to inject at QP


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


def patch_hook(mo, i_, o_):
    if not PATCH['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_          # (B, T, D)
    U = PATCH['U']; b = y[:, QP, :]
    b_new = b - (b @ U) @ U.T + PATCH['src']            # swap the subspace coords for the source's
    y = y.clone(); y[:, QP, :] = b_new
    return (y,) + tuple(o_[1:]) if isinstance(o_, tuple) else y


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capture_L(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[PATCH_L].register_forward_hook(h)
    lg = forward_logits(idx); hh.remove()
    return cap['r'], lg


def class_subspace_from_means(R, lab, r):
    g = R.mean(0, keepdim=True); rows = []
    for c in range(len(CLASSES)):
        mk = lab == c
        if mk.sum() < 5: continue
        rows.append(R[torch.tensor(mk, device=DEV)].mean(0) - g[0])
    M = torch.stack(rows, 0)
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def class_subspace_from_weights(tok2cls, r):
    W = m.lm_head.weight.detach().float()              # (V, D)
    g = W.mean(0, keepdim=True); rows = []
    for c in range(len(CLASSES)):
        mk = tok2cls == c
        if mk.sum() < 5: continue
        rows.append(W[torch.tensor(mk, device=DEV)].mean(0) - g[0])
    M = torch.stack(rows, 0)
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :SEQ].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    nxt_cls = np.array([CLASSES.index(classify(d(int(S[r, QP+1])))) for r in range(nb)])   # class of token predicted at QP
    # capture clean L15 residual at QP + clean predicted class
    Rqp = torch.zeros(nb, D); pred_clean = np.zeros(nb, dtype=np.int64)
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV); r, lg = capture_L(idx)
        Rqp[i:i+idx.shape[0]] = r[:, QP, :].cpu()
        p = lg[:, QP, :].float().argmax(-1).cpu().numpy()
        pred_clean[i:i+idx.shape[0]] = np.array([tok2cls[int(t)] for t in p])
    Rqp = Rqp.to(DEV)
    lab = torch.tensor(nxt_cls, device=DEV)
    U_act = class_subspace_from_means(Rqp, lab, RANK)
    U_wt = class_subspace_from_weights(cidx, RANK)
    g = torch.Generator(device=DEV).manual_seed(0); U_rnd = torch.linalg.qr(torch.randn(D, RANK, generator=g, device=DEV))[0]
    # build base/source pairs: source has a DIFFERENT next-class than base
    rng = np.random.RandomState(0); idxs = np.arange(nb)
    src_of = np.zeros(nb, dtype=np.int64)
    for bi in range(nb):
        cand = idxs[nxt_cls != nxt_cls[bi]]; src_of[bi] = cand[rng.randint(len(cand))]
    hh = m.transformer.h[PATCH_L].register_forward_hook(patch_hook)
    def run_interchange(U):
        PATCH['U'] = U; flips = 0; n = 0
        for i in range(0, nb, 8):
            bidx = blocks[i:i+8].to(DEV); bb = np.arange(i, min(i+8, nb))
            src = Rqp[torch.tensor(src_of[bb], device=DEV)]                 # (b, D) source residuals at QP
            PATCH['src'] = (src @ U) @ U.T; PATCH['on'] = True
            lg = forward_logits(bidx); PATCH['on'] = False
            p = lg[:, QP, :].float().argmax(-1).cpu().numpy()
            pcls = np.array([tok2cls[int(t)] for t in p])
            flips += int((pcls == nxt_cls[src_of[bb]]).sum()); n += len(bb)
        return round(flips / n, 3)
    iia_act = run_interchange(U_act); iia_wt = run_interchange(U_wt); iia_rnd = run_interchange(U_rnd)
    hh.remove()
    # baselines: clean predicted class matches source class by coincidence
    base_coincide = round(float((pred_clean == nxt_cls[src_of]).mean()), 3)
    src_cls_rate = round(float(np.mean([np.mean(nxt_cls == nxt_cls[src_of[bi]]) for bi in range(nb)])), 3)
    out = {'patch_layer': PATCH_L, 'query_pos': QP, 'rank': RANK, 'n': nb,
           'IIA_activation_subspace': iia_act, 'IIA_weight_subspace': iia_wt, 'IIA_random_subspace': iia_rnd,
           'baseline_clean_coincidental_match': base_coincide, 'source_class_prior': src_cls_rate,
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_class_causal_variable'] = bool(iia_act > iia_rnd + 0.15 and iia_act > base_coincide + 0.15)
    out['pred_b_weights_identify_variable'] = bool(iia_wt > iia_rnd + 0.10 and iia_wt > 0.6 * iia_act)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"INTERCHANGE (patch class-subspace base<-source at L{PATCH_L}, pos {QP}); flip-to-source-class rate:", flush=True)
    print(f"  activation-subspace IIA {iia_act} | weight-subspace IIA {iia_wt} | random-subspace {iia_rnd}", flush=True)
    print(f"  no-patch coincidental match {base_coincide} | source-class prior {src_cls_rate}", flush=True)
    print(f"(a) class is a causal variable via interchange: {out['pred_a_class_causal_variable']}", flush=True)
    print(f"(b) weights identify the variable: {out['pred_b_weights_identify_variable']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
