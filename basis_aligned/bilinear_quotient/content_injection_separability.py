"""GENERATIVE demonstration of the CORE two-machine claim -- content is long-range/pooled, grammar is local -- from
the input side (extends §1016/§1017). Inject a topical content word W at a FAR position (3) vs a NEAR position (148),
and at query 150 measure BOTH: (a) the CONTENT shift = topic-neighbor Δlog-prob (§1016 metric), and (b) the GRAMMAR
shift = total-variation distance between the injected and baseline next-token CLASS distributions. If the two machines
are separable as claimed, a FAR content injection shifts CONTENT (broad pooling reaches the query) but barely shifts
GRAMMAR (grammar is set by the local tokens around the query, unchanged); a NEAR injection (adjacent to the query)
DOES disrupt grammar -- the positive control that the grammar metric can move.

REGISTERED PREDICTIONS:
  (0) POSITIVE CONTROL: a NEAR injection (pos 148, adjacent to query 150) produces a MUCH larger grammar shift
      (class-distribution TV) than a FAR injection (pos 3) -> the grammar metric is sensitive; grammar is LOCAL.
  (a) SEPARABILITY: a FAR content injection produces a LARGE content shift (topic-neighbor Δlp, ~0.5 per §1016) but a
      SMALL grammar shift (class-TV), ratio content-effect/grammar-effect high -> content is pooled long-range while
      grammar at the query is untouched by a distant content word: the two machines are separable, shown generatively;
  (b) report far/near content-shift and grammar-shift (class-TV)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_separability_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; FAR = 3; NEAR = 148; NNEIGH = 20
CONTENT_WORDS = [' football', ' hospital', ' ocean', ' music', ' science', ' army', ' church', ' garden']
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}


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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k):
    W = m.lm_head.weight.float(); wv = W[wid] / (W[wid].norm() + 1e-9)
    sims = (W / (W.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def shifts(blocks, wid, neigh, inj_pos, Cmat):
    # returns (mean content-shift = neighbor Δlp, mean grammar-shift = class-dist TV) at query
    cont = 0.0; gram = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous()
        inj_idx = base_idx.clone(); inj_idx[:, inj_pos] = wid
        p_base = F.softmax(forward_logits(base_idx).float()[:, -1], -1)
        p_inj = F.softmax(forward_logits(inj_idx).float()[:, -1], -1)
        has = (base_idx == wid).any(1)
        # content shift (log-prob of neighbors)
        dlp = (p_inj[:, neigh].clamp_min(1e-12).log() - p_base[:, neigh].clamp_min(1e-12).log()).mean(1)
        # grammar shift: class distribution TV
        cb = p_base @ Cmat; ci = p_inj @ Cmat  # (B, C)
        tv = 0.5 * (ci - cb).abs().sum(1)
        cont += float(dlp[~has].sum()); gram += float(tv[~has].sum()); n += int((~has).sum())
    return cont/max(n, 1), gram/max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); d = dec()
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    t2c = np.full(V, 7, np.int64)
    for t in np.unique(rows.cpu().numpy().reshape(-1)): t2c[int(t)] = CLASSES.index(classify(d(int(t))))
    Cmat = F.one_hot(torch.tensor(t2c, device=DEV), C).float()
    far_c = []; far_g = []; near_c = []; near_g = []
    for w in CONTENT_WORDS:
        wid = tid(w)
        if wid is None: continue
        neigh = neighbors(wid, NNEIGH)
        fc, fg = shifts(rows, wid, neigh, FAR, Cmat); nc, ng = shifts(rows, wid, neigh, NEAR, Cmat)
        far_c.append(fc); far_g.append(fg); near_c.append(nc); near_g.append(ng)
        print(f"{w!r}: FAR content {fc:.3f} grammar-TV {fg:.4f} | NEAR content {nc:.3f} grammar-TV {ng:.4f}", flush=True)
    out = {'far_content': round(float(np.mean(far_c)), 4), 'far_grammar_tv': round(float(np.mean(far_g)), 4),
           'near_content': round(float(np.mean(near_c)), 4), 'near_grammar_tv': round(float(np.mean(near_g)), 4)}
    out['pred_0_near_grammar_gt_far'] = bool(out['near_grammar_tv'] > 2*out['far_grammar_tv'])
    out['pred_a_separable'] = bool(out['far_content'] > 0.2 and out['far_content'] > 3*out['far_grammar_tv'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"FAR: content {out['far_content']} grammar-TV {out['far_grammar_tv']} | NEAR: content {out['near_content']} grammar-TV {out['near_grammar_tv']}", flush=True)
    print(f"pred_0 near-grammar>far {out['pred_0_near_grammar_gt_far']} | pred_a separable {out['pred_a_separable']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
