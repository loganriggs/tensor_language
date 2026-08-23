"""VERIFY §1006/§1007 (L5 head 7 is the dominant content-gatherer) with an INDEPENDENT instrument. §1006/1007 used
attention-WINDOW BANDING (restrict a head to a local window). Here use OUTPUT MEAN-ABLATION: replace each L5 head's
per-head attention output z[:,h] with its calibration mean (removing that head's position-varying contribution
entirely), and measure the chain-rule CE split. If L5 h7 is the content-gatherer, its output-ablation should cost
within-CE (content) far more than any other L5 head and far more than its own class-CE (grammar) -- confirming §1007
via a different intervention.

REGISTERED PREDICTIONS:
  (0) NULL: ablating NO head == original CE.
  (a) L5 h7 DOMINATES CONTENT (independent confirmation): among the 9 L5 heads, h7's output mean-ablation has the
      LARGEST within-CE (content) cost, and h7's within-cost > its class-cost -> h7 is the content head, confirmed by
      output-ablation as well as banding (§1006/1007);
  (b) report per-L5-head within-CE and class-CE cost + h7's rank and content/grammar ratio."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_L5_head_ablation_results.json'
NCAL = 48; NEVAL = 160; SEQ = 256; L5 = 5
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
MEANZ = {'v': None}     # (H, d) calibration mean of per-head z at L5
ABL = {'head': None, 'capture': False, 'buf': []}


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
    z = torch.einsum('bhqk,bkhd->bhqd', pattern, v)   # (B,H,T,d)
    if self._bl == L5:
        if ABL['capture']:
            ABL['buf'].append(z.mean(dim=(0, 2)).detach())   # (H,d) per-head mean over batch,positions
        elif ABL['head'] is not None:
            h = ABL['head']; z = z.clone(); z[:, h] = MEANZ['v'][h].to(z.dtype).view(1, 1, -1)
    return z


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
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES); H = m.transformer.h[0].attn.n_head
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for L, a in enumerate(attns): a._bl = L; a.squared_attention = types.MethodType(patched_squared_attention, a)
    # calibrate per-head z mean at L5
    ABL['capture'] = True; ABL['buf'] = []
    for i in range(0, calib.shape[0], 8): forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
    MEANZ['v'] = torch.stack(ABL['buf'], 0).mean(0)   # (H,d)
    ABL['capture'] = False; ABL['head'] = None
    base = split_ce(blocks, cidx, C); print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'per_head': {}}
    for h in range(H):
        ABL['head'] = h
        r = split_ce(blocks, cidx, C)
        out['per_head'][str(h)] = {'within_cost': round(r['within_ce']-base['within_ce'], 4),
                                   'class_cost': round(r['class_ce']-base['class_ce'], 4)}
        print(f"L5 h{h} mean-ablate: within +{out['per_head'][str(h)]['within_cost']} class +{out['per_head'][str(h)]['class_cost']}", flush=True)
    ABL['head'] = None
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    within = {h: out['per_head'][h]['within_cost'] for h in out['per_head']}
    top = max(within, key=within.get)
    h7 = out['per_head']['7']
    out['top_content_head'] = top
    out['h7_content_grammar_ratio'] = round(h7['within_cost']/max(abs(h7['class_cost']), 1e-6), 2)
    out['pred_a_h7_dominates_content'] = bool(top == '7' and h7['within_cost'] > abs(h7['class_cost']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top content head (output-ablation) L5 h{top} | h7 within {h7['within_cost']} class {h7['class_cost']} ratio {out['h7_content_grammar_ratio']}", flush=True)
    print(f"pred_a h7 dominates content (independent confirm) {out['pred_a_h7_dominates_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
