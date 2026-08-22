"""WORKED EXAMPLES: trace individual predictions through the named mechanisms end to end, to make the whole
account concrete. For a handful of real contexts, at the final position report: (1) the current token and its
multi-axis CLASS attributes (§915); (2) the context's TOPIC (its distinctive words, §866); (3) the model's
top-5 predicted next tokens and the predicted CLASS; (4) the causal effect of ABLATING the class channel
(should change the predicted part-of-speech) vs the CONTENT channel (should change the topical specificity) —
the recombination (§921-923). No prediction to register (descriptive worked example); controls = the ablations
themselves (class vs content vs none).
"""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'worked_example_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RCLASS = 8; RCONTENT = 64
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
ABL = {'on': False, 'U': None, 'pos': None}
QPS = [80, 120, 160, 200]


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
    y = o_[0] if isinstance(o_, tuple) else o_; U = ABL['U']; p = ABL['pos']
    b = y[:, p, :]; y = y.clone(); y[:, p, :] = b - (b @ U) @ U.T
    return (y,) + tuple(o_[1:]) if isinstance(o_, tuple) else y


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capL(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


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
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Uclass, _ = mean_subspace(R, nxtcls, RCLASS)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy().reshape(nb, SEQ-1)
    Ucontent = torch.linalg.svd(content - content.mean(0, keepdim=True), full_matrices=False)[2][:RCONTENT].T.contiguous()
    # topic distinctive tokens
    from collections import Counter
    tgt = np.full_like(S, -1); tgt[:, :-1] = S[:, 1:]; tgtf = tgt.reshape(-1); base = Counter(tgtf[tgtf >= 0]); Nn = int((tgtf >= 0).sum())
    tflat = topic.reshape(-1); dname = {}
    for j in range(K):
        mk = tflat == j; nc = Counter(); tt = tgtf[:len(tflat)]
        for t in np.unique(tt[mk]):
            if t < 0: continue
            c = int((tt[mk] == t).sum())
            if c < 4: continue
            nc[t] = (c/max(mk.sum(),1))/((base.get(t,0)+1)/Nn)
        dname[j] = [repr(d(int(t))) for t, _ in nc.most_common(6)]
    hh = m.transformer.h[CONTENT_L].register_forward_hook(abl_hook)
    def top5(idx, U, on, p):
        ABL['U'] = U if U is not None else Uclass; ABL['on'] = on; ABL['pos'] = p
        lg = forward_logits(idx).float()[0, p]; ABL['on'] = False
        pr = torch.topk(lg, 5).indices.cpu().numpy(); return [repr(d(int(t))) for t in pr], classify(d(int(pr[0])))
    examples = []
    for si in range(0, min(nb, 6)):
        idx = blocks[si:si+1, :SEQ].to(DEV)[:, :-1].contiguous()
        for p in QPS:
            curtok = d(int(S[si, p])); ctx = "".join(d(int(t)) for t in S[si, max(0,p-8):p+1])
            tj = int(topic[si, p]); base_pred, base_cls = top5(idx, None, False, p)
            abl_cls_pred, abl_cls_cls = top5(idx, Uclass, True, p)
            abl_con_pred, abl_con_cls = top5(idx, Ucontent, True, p)
            examples.append({'context_tail': ctx, 'current_token': curtok,
                             'current_class': classify(curtok), 'topic_cluster': tj, 'topic_distinctive': dname.get(tj, []),
                             'baseline_top5': base_pred, 'baseline_pred_class': base_cls,
                             'ablate_class_top5': abl_cls_pred, 'ablate_class_pred_class': abl_cls_cls,
                             'ablate_content_top5': abl_con_pred, 'ablate_content_pred_class': abl_con_cls})
    hh.remove()
    out = {'n_examples': len(examples), 'examples': examples[:16], 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for e in examples[:8]:
        print(f"...{e['context_tail']!r} | cur={e['current_token']!r}({e['current_class']}) topic~{e['topic_distinctive'][:4]}", flush=True)
        print(f"   pred {e['baseline_top5'][:4]} (class {e['baseline_pred_class']}) | -class-> {e['ablate_class_top5'][:3]}(class {e['ablate_class_pred_class']}) | -content-> {e['ablate_content_top5'][:3]}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
