"""FRESH: §998 localized the content POOLING to layers L3-5 (banding their attention window costs the most content),
and §981-984 analyzed per-head ROUTING at L8. But WHICH HEADS in the L3-5 gathering band do the long-range content
pooling? Localize to head resolution: for each head index h (0-8), restrict ONLY that head's attention to a K=8 window
in ALL of L3-5 (other heads keep full context), and measure the within-CE (content) cost. Heads whose banding costs
the most content are the long-range content-gatherers.

REGISTERED PREDICTIONS:
  (0) NULL: banding NO heads == original CE; banding ALL heads in L3-5 reproduces §998's L3-5 within-cost (~0.52).
  (a) POOLING IS CONCENTRATED IN A FEW HEADS: a minority of the 9 heads account for most of the L3-5 content-
      gathering cost (top-3 heads > 60% of the all-heads cost) -> the long-range content pooling is head-localized,
      not uniform across heads;
  (b) report per-head within-CE cost + the all-heads reference + the top-3 share."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_pooling_heads_results.json'
NEVAL = 160; SEQ = 256; K = 8; POOL_LAYERS = {3, 4, 5}
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
BAND = {'K': K, 'layers': POOL_LAYERS, 'heads': set()}


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


def banded_squared_attention(self, q, k, v, q2, k2):
    B, T, H, Dh = q.shape
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
    scores2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pattern = (scores / Dh) * (scores2 / Dh)
    i = torch.arange(T, device=pattern.device)
    causal = i[:, None] >= i[None, :]                       # (T,T)
    keep = causal.unsqueeze(0).expand(H, T, T).clone()      # (H,T,T)
    if self._bl in BAND['layers'] and BAND['heads']:
        band_dist = (i[:, None] - i[None, :]) < BAND['K']   # (T,T)
        for h in BAND['heads']:
            keep[h] = keep[h] & band_dist
    pattern = pattern.masked_fill(~keep.unsqueeze(0), 0.0)  # broadcast over batch
    return torch.einsum('bhqk,bkhd->bhqd', pattern, v)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    return {'full_ce': round(tot/n, 4), 'class_ce': round(totc/n, 4), 'within_ce': round((tot-totc)/n, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES); H = m.transformer.h[0].attn.n_head
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for L, a in enumerate(attns): a._bl = L; a.squared_attention = types.MethodType(banded_squared_attention, a)
    BAND['heads'] = set(); base = split_ce(blocks, cidx, C); print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'per_head': {}}
    for h in range(H):
        BAND['heads'] = {h}
        r = split_ce(blocks, cidx, C)
        out['per_head'][str(h)] = round(r['within_ce'] - base['within_ce'], 4)
        print(f"head {h}: within-cost +{out['per_head'][str(h)]}", flush=True)
    BAND['heads'] = set(range(H)); rall = split_ce(blocks, cidx, C)
    out['all_heads_within_cost'] = round(rall['within_ce'] - base['within_ce'], 4)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    costs = sorted(out['per_head'].values(), reverse=True)
    out['top3_share'] = round(sum(costs[:3]) / max(sum(c for c in costs if c > 0), 1e-6), 3)
    out['per_head_sum'] = round(sum(out['per_head'].values()), 4)
    out['pred_a_head_concentrated'] = bool(out['top3_share'] > 0.6)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"all-heads within-cost {out['all_heads_within_cost']} (§998 L3-5 ~0.52) | per-head sum {out['per_head_sum']} | top3 share {out['top3_share']}", flush=True)
    print(f"pred_a head-concentrated {out['pred_a_head_concentrated']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
