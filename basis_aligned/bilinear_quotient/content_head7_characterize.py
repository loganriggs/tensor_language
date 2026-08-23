"""CHARACTERIZE head 7, the dominant L3-5 content-gathering head (§1006). Three questions:
 (1) WHICH layer's h7 matters -- band h7 in L3, L4, L5 individually and jointly (content within-CE + grammar class-CE
     split);
 (2) is it really a CONTENT head -- banding h7 should cost within-CE (content) >> class-CE (grammar);
 (3) does it pool BROADLY (bag-of-words) or LOCALLY -- capture h7's raw squared-attention pattern in L3-5 and measure
     the fraction of its (absolute) attention mass at distance >8 (broad) vs <=8 (local), and the correlation of its
     per-key mass with distance (recency).

REGISTERED PREDICTIONS:
  (0) NULL: banding no layer == original CE.
  (a) CONTENT HEAD: banding h7 (jointly L3-5) costs within-CE (content) MUCH more than class-CE (grammar), ratio > 3;
  (b) BROAD POOLER: a large fraction (> ~0.5) of h7's attention mass is at distance > 8 -> it pools broad context
      (bag-of-words §932/§995), NOT a local head; report the mass>8 fraction and the layer breakdown;
  (c) report per-layer (L3,L4,L5) h7 banding within/class cost to localize."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_head7_characterize_results.json'
NEVAL = 160; SEQ = 256; K = 8; HEAD = 7; POOL_LAYERS = [3, 4, 5]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
BAND = {'K': K, 'layers': set(), 'head': HEAD}
CAP = {'layer': None, 'pat': None}


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


def patched_squared_attention(self, q, k, v, q2, k2):
    B, T, H, Dh = q.shape
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
    scores2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pattern = (scores / Dh) * (scores2 / Dh)
    i = torch.arange(T, device=pattern.device)
    causal = i[:, None] >= i[None, :]
    pattern = pattern.masked_fill(~causal.unsqueeze(0).unsqueeze(0), 0.0)
    if self._bl == CAP['layer']:
        CAP['pat'] = pattern[:, BAND['head']].detach()   # (B,T,T) raw pattern for head
    if self._bl in BAND['layers']:
        band_dist = (i[:, None] - i[None, :]) < BAND['K']
        keep = torch.ones(H, T, T, dtype=torch.bool, device=pattern.device)
        keep[BAND['head']] = band_dist
        pattern = pattern.masked_fill(~keep.unsqueeze(0), 0.0)
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
def mass_profile(blocks, layer):
    # fraction of |pattern| mass at distance>8, and mean |mass| by distance decade, for head7 at `layer`
    CAP['layer'] = layer; BAND['layers'] = set()
    far = 0.0; near = 0.0
    for i in range(0, min(blocks.shape[0], 40), 8):
        forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
        p = CAP['pat'].abs()  # (B,T,T)
        B, T, _ = p.shape; ii = torch.arange(T, device=DEV)
        dist = (ii[:, None] - ii[None, :])  # (T,T) query-key distance (>=0 causal)
        farmask = (dist > 8) & (dist <= ii[:, None]); nearmask = (dist >= 0) & (dist <= 8)
        far += float((p * farmask).sum()); near += float((p * nearmask).sum())
    CAP['layer'] = None
    return round(far/max(far+near, 1e-9), 3)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for L, a in enumerate(attns): a._bl = L; a.squared_attention = types.MethodType(patched_squared_attention, a)
    CAP['layer'] = None; BAND['layers'] = set(); base = split_ce(blocks, cidx, C); print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'per_layer': {}, 'mass_gt8': {}}
    for L in POOL_LAYERS:
        BAND['layers'] = {L}
        r = split_ce(blocks, cidx, C)
        out['per_layer'][str(L)] = {'within_cost': round(r['within_ce']-base['within_ce'], 4),
                                    'class_cost': round(r['class_ce']-base['class_ce'], 4)}
        print(f"h7 @ L{L}: within +{out['per_layer'][str(L)]['within_cost']} class +{out['per_layer'][str(L)]['class_cost']}", flush=True)
    BAND['layers'] = set(POOL_LAYERS); rj = split_ce(blocks, cidx, C)
    out['joint'] = {'within_cost': round(rj['within_ce']-base['within_ce'], 4), 'class_cost': round(rj['class_ce']-base['class_ce'], 4)}
    for L in POOL_LAYERS: out['mass_gt8'][str(L)] = mass_profile(blocks, L)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    jw = out['joint']['within_cost']; jc = out['joint']['class_cost']
    out['content_grammar_ratio'] = round(jw/max(abs(jc), 1e-6), 2)
    out['mean_mass_gt8'] = round(float(np.mean(list(out['mass_gt8'].values()))), 3)
    out['pred_a_content_head'] = bool(jw > 3*abs(jc))
    out['pred_b_broad_pooler'] = bool(out['mean_mass_gt8'] > 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"joint h7 L3-5: within +{jw} class +{jc} (ratio {out['content_grammar_ratio']}) | mass>8 {out['mass_gt8']} mean {out['mean_mass_gt8']}", flush=True)
    print(f"pred_a content-head {out['pred_a_content_head']} | pred_b broad-pooler {out['pred_b_broad_pooler']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
