"""RECONCILE §894 (topic causal on topic-distinctive tokens) vs §922 (topic-subspace ablation ~0 on overall
within-CE): is topic's effect concentrated on FIRST-MENTION/content positions? Ablate the class subspace and
the (continuous, rank-64) content subspace at L15, and measure the within-CE (content loss) change SPLIT by
position type (first-mention / seen-other / inductable). If content-subspace ablation raises within-CE on
FIRST-MENTION positions much more than on inductable positions, topic/content drives the new-word choice there
specifically (reconciling the diluted overall number). Use a rank-64 CONTINUOUS content subspace (not the 11
cluster-means, which §922 showed carry nothing).

REGISTERED PREDICTIONS:
  (0) SANITY: baseline within-CE first-mention >> inductable (§879);
  (a) CONTENT DRIVES FIRST-MENTION WORD: ablating the content subspace raises within-CE on FIRST-MENTION
      positions more than on inductable positions (content's word-role is concentrated where a new content word
      is needed); class-subspace ablation hurts all positions (dominant);
  (b) report within-CE increase by position type for each ablation + random null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'word_recombination_positions_results.json'
CONTENT_L = 15; NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; RCLASS = 8; RCONTENT = 64
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
    content = (R-g) - ((R-g)@Ucp)@Ucp.T
    Ucontent = torch.linalg.svd(content - content.mean(0, keepdim=True), full_matrices=False)[2][:RCONTENT].T.contiguous()
    gd = torch.Generator(device=DEV).manual_seed(0); Urnd = torch.linalg.qr(torch.randn(D, RCONTENT, generator=gd, device=DEV))[0]
    # position type masks (per next-token target)
    V = int(m.lm_head.weight.shape[0]); tok2cls = np.full(V, 7)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    inductable = np.zeros((nb, SEQ-1), bool); firstment = np.zeros((nb, SEQ-1), bool)
    for r in range(nb):
        seen = set(); big = {}
        for p in range(SEQ-1):
            cur = int(S[r, p]); nx = int(S[r, p+1]); firstment[r, p] = nx not in seen
            if cur in big and big[cur] == nx: inductable[r, p] = True
            big[cur] = nx; seen.add(cur)
    inductable = inductable.reshape(-1); firstment = firstment.reshape(-1) & ~inductable; other = ~inductable & ~firstment
    hh = m.transformer.h[CONTENT_L].register_forward_hook(abl_hook)
    def within_by_pos(U, on):
        ABL['U'] = U if U is not None else Uclass; ABL['on'] = on; wv = []
        for i in range(0, nb, 4):
            bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            lp = F.log_softmax(forward_logits(idx).float(), -1); pcl = lp.exp() @ Cmat
            tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1]); tcli = cidx[tf]
            lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]; lp_cls = (pcl.reshape(-1, len(CLASSES))[torch.arange(tf.shape[0], device=DEV), tcli]+1e-12).log()
            wv.append((-(lp_tok-lp_cls)).cpu().numpy())
        ABL['on'] = False; w = np.concatenate(wv)
        return {'first_mention': float(w[firstment].mean()), 'seen_other': float(w[other].mean()), 'inductable': float(w[inductable].mean())}
    base = within_by_pos(None, False)
    ablC = within_by_pos(Uclass, True); ablK = within_by_pos(Ucontent, True); ablR = within_by_pos(Urnd, True)
    hh.remove()
    def delta(a): return {k: round(a[k]-base[k], 3) for k in base}
    out = {'baseline_within_ce': {k: round(base[k], 3) for k in base},
           'class_ablation_delta': delta(ablC), 'content_ablation_delta': delta(ablK), 'random_ablation_delta': delta(ablR),
           'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_content_drives_firstmention'] = bool(out['content_ablation_delta']['first_mention'] > out['content_ablation_delta']['inductable'] + 0.1 and
                                                     out['content_ablation_delta']['first_mention'] > out['random_ablation_delta']['first_mention'] + 0.1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"baseline within-CE: first-mention {base['first_mention']:.2f} seen {base['seen_other']:.2f} inductable {base['inductable']:.2f}", flush=True)
    print(f"CLASS-ablation Δwithin-CE: {out['class_ablation_delta']}", flush=True)
    print(f"CONTENT-ablation Δwithin-CE: {out['content_ablation_delta']}", flush=True)
    print(f"RANDOM-ablation Δwithin-CE: {out['random_ablation_delta']}", flush=True)
    print(f"(a) content drives first-mention word (content-ablation hits first-mention >> inductable): {out['pred_a_content_drives_firstmention']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
