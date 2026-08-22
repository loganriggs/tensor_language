"""[CE-based, cleaner metric than sparse distinctive-tokens §921] Ablate each channel at L15 and measure the chain-rule CE split (class-CE = grammar loss, within-CE = content loss): class-ablation should raise CLASS-CE, topic-ablation should raise WITHIN-CE — a loss double-dissociation. HOW do the two orthogonal channels RECOMBINE at the readout to pick a word? (§920: grammar and topic are
separable/orthogonal — but the model must combine them to emit one token). Hypothesis: the word is their
INTERSECTION — the CLASS channel controls the part-of-speech of the prediction, the TOPIC channel controls its
subject. Test by ablating (projecting out) each channel at the readout input (L15) and measuring the effect on
the PREDICTED token's class-correctness (grammar) vs topic-correctness (subject):
  - ablate CLASS channel -> the predicted token's CLASS should degrade (grammar broken) but its topic survive;
  - ablate TOPIC channel -> the predicted token's TOPIC should degrade (subject broken) but its class survive.
If each ablation selectively damages its own attribute of the prediction, the word is the class∩topic
intersection. Controls: ablate a random matched-rank subspace (null); no-ablation baseline.

REGISTERED PREDICTIONS:
  (0) SANITY: no-ablation predicted token is mostly class-correct and topic-plausible;
  (a) INTERSECTION: ablating class hurts the predicted token's class-match >> its topic-match; ablating topic
      hurts topic-match >> class-match (double dissociation) -> class channel = grammar-of-word, topic channel
      = subject-of-word, word = intersection;
  (b) if both ablations hurt both, the channels are not cleanly the two attributes of the word (report)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'word_recombination_ce_results.json'
CONTENT_L = 15; NEVAL = 220; RTOK = 64; RPOS = 32; K = 12; RCLASS = 8
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
    y = o_[0] if isinstance(o_, tuple) else o_; U = ABL['U']; sh = y.shape; v = y.reshape(-1, D)
    v2 = v - (v @ U) @ U.T
    return (v2.reshape(sh),) + tuple(o_[1:]) if isinstance(o_, tuple) else v2.reshape(sh)


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
    blocks = rows[:, :257].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(256), (nb, 256)).reshape(-1)
    nxtc = np.full_like(S[:, :-1], -1); nxtc[:, :-1] = S[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Uclass, _ = mean_subspace(R, nxtcls, RCLASS)
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, topic, K-1)
    gd = torch.Generator(device=DEV).manual_seed(0); Urnd = torch.linalg.qr(torch.randn(D, RCLASS, generator=gd, device=DEV))[0]
    # token->class and token->topic maps for measuring the PREDICTED token's attributes
    V = int(m.lm_head.weight.shape[0]); tok2cls = np.full(V, 7)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    # predicted token's topic: assign predicted token to nearest topic centroid via its content-cluster... approx by
    # measuring whether predicted token is in the SAME topic-distinctive set as the true context topic. Simpler:
    # topic-match = does predicted token appear in the context's topic distinctive tokens.
    from collections import Counter
    tgt_all = np.full_like(S, -1); tgt_all[:, :-1] = S[:, 1:]; tgt_all = tgt_all.reshape(-1)
    base = Counter(tgt_all[tgt_all >= 0]); Nn = int((tgt_all >= 0).sum()); dtok = {}
    tp_full = topic  # per (seq,pos) topic label for positions 0..254
    for j in range(K):
        mk = tp_full == j; nj = int((tgt_all[:len(tp_full)][mk] >= 0).sum()) if False else int(mk.sum())
        nc = Counter();
        # distinctive next tokens of topic j
        tt = tgt_all[:len(tp_full)]
        for t in np.unique(tt[mk]):
            if t < 0: continue
            c = int((tt[mk] == t).sum())
            if c < 4: continue
            nc[t] = (c/max(mk.sum(),1)) / ((base.get(t,0)+1)/Nn)
        dtok[j] = set([t for t,_ in nc.most_common(50)])
    Cmat = F.one_hot(torch.tensor(tok2cls, device=DEV), len(CLASSES)).float()
    cidx = torch.tensor(tok2cls, device=DEV)
    hh = m.transformer.h[CONTENT_L].register_forward_hook(abl_hook)
    def measure(U, on):
        ABL['U'] = U if U is not None else Uclass; ABL['on'] = on
        tc = tw = 0.0; nn = 0
        for i in range(0, nb, 4):
            bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lp = F.log_softmax(forward_logits(idx).float(), -1); pcl = lp.exp() @ Cmat
            tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1]); tcli = cidx[tf]
            lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
            lp_cls = (pcl.reshape(-1, len(CLASSES))[torch.arange(tf.shape[0], device=DEV), tcli] + 1e-12).log()
            tc += float((-lp_cls).sum()); tw += float((-(lp_tok-lp_cls)).sum()); nn += tf.shape[0]
        ABL['on'] = False; return tc/nn, tw/nn
    base_cc, base_tc = measure(None, False)
    abl_cls_cc, abl_cls_tc = measure(Uclass, True)
    abl_top_cc, abl_top_tc = measure(Utopic, True)
    abl_rnd_cc, abl_rnd_tc = measure(Urnd, True)
    hh.remove()
    out = {'baseline': {'class_ce': round(base_cc,3), 'within_ce': round(base_tc,3)},
           'ablate_class': {'class_ce': round(abl_cls_cc,3), 'within_ce': round(abl_cls_tc,3)},
           'ablate_topic': {'class_ce': round(abl_top_cc,3), 'within_ce': round(abl_top_tc,3)},
           'ablate_random': {'class_ce': round(abl_rnd_cc,3), 'within_ce': round(abl_rnd_tc,3)},
           'class_abl_raises_classCE': round(abl_cls_cc-base_cc,3), 'class_abl_raises_withinCE': round(abl_cls_tc-base_tc,3),
           'topic_abl_raises_classCE': round(abl_top_cc-base_cc,3), 'topic_abl_raises_withinCE': round(abl_top_tc-base_tc,3),
           'runtime_s': round(time.time()-t0,1)}
    out['pred_a_intersection'] = bool(out['class_abl_raises_classCE'] > out['class_abl_raises_withinCE'] and out['topic_abl_raises_withinCE'] > out['topic_abl_raises_classCE'])
    json.dump(out, open(OUT,'w'), indent=1)
    print(f"baseline: class-CE {base_cc:.3f} within-CE {base_tc:.3f}", flush=True)
    print(f"ablate CLASS: class-CE {abl_cls_cc:.3f} (+{abl_cls_cc-base_cc:.3f}) within-CE {abl_cls_tc:.3f} (+{abl_cls_tc-base_tc:.3f})", flush=True)
    print(f"ablate TOPIC: class-CE {abl_top_cc:.3f} (+{abl_top_cc-base_cc:.3f}) within-CE {abl_top_tc:.3f} (+{abl_top_tc-base_tc:.3f})", flush=True)
    print(f"ablate RANDOM: class-CE {abl_rnd_cc:.3f} within-CE {abl_rnd_tc:.3f}", flush=True)
    print(f"(a) class->classCE, topic->withinCE double dissociation: {out['pred_a_intersection']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
