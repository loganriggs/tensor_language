"""ROBUSTNESS of the content-gathering localization (§998/§1001, ALL banding-based) under the cleaner OUTPUT-ABLATION
instrument (§1008 showed banding can overstate a component's unique role ~30x vs output-ablation). Mean-ablate each
3-layer group's ATTENTION OUTPUT (replace the block's attention contribution with its calibration mean) and measure
the chain-rule CE split. Does L3-5 remain the top content-gathering group, and are the costs lower than banding
(redundancy)?

REGISTERED PREDICTIONS:
  (0) NULL: ablating NO group == original CE.
  (a) LOCATION ROBUST: L3-5 remains the TOP content-gathering group (largest within-CE cost) under output-ablation,
      confirming the §998 localization is not a banding artifact;
  (b) REDUNDANCY: the output-ablation within-CE costs are LOWER than the banding costs (§998 L3-5 ~0.52) -> the
      content gathering is redundant/compensable at the band level too (consistent with §1008 for h7);
  (c) content>grammar in the gathering band; report per-group within/class cost + comparison to banding."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_gathering_ablate_results.json'
NCAL = 48; NEVAL = 160; SEQ = 256
GROUPS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
AMEAN = {}          # per-layer attention-output mean (1,1,D)
ABL = {'layers': set(), 'capture': False, 'buf': None}


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


def attn_hook_factory(L):
    def h(mo, i_, o_):
        y = o_[0] if isinstance(o_, tuple) else o_       # attention contribution (B,T,D)
        if ABL['capture']:
            ABL['buf'][L] = ABL['buf'].get(L, 0.0) + y.float().sum(dim=(0, 1)); ABL['cnt'] = ABL.get('cnt', 0) + y.shape[0]*y.shape[1]
            return o_
        if L in ABL['layers']:
            ny = AMEAN[L].to(y.dtype).expand_as(y)
            return (ny,) + tuple(o_[1:]) if isinstance(o_, tuple) else ny
        return o_
    return h


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
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    hooks = [m.transformer.h[L].attn.register_forward_hook(attn_hook_factory(L)) for L in range(18)]
    # calibrate per-layer attention-output mean
    ABL['capture'] = True; ABL['buf'] = {}; ABL['cnt'] = 0
    for i in range(0, calib.shape[0], 8): forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
    for L in range(18): AMEAN[L] = (ABL['buf'][L] / ABL['cnt']).view(1, 1, D)
    ABL['capture'] = False; ABL['layers'] = set()
    base = split_ce(blocks, cidx, C); print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'per_group': {}, 'banding_ref': {'L0_2': 0.061, 'L3_5': 0.520, 'L6_8': 0.071, 'L9_11': 0.026, 'L12_14': 0.045, 'L15_17': 0.019}}
    for a, b in GROUPS:
        ABL['layers'] = set(range(a, b+1)); tag = f'L{a}_{b}'
        r = split_ce(blocks, cidx, C)
        out['per_group'][tag] = {'within_cost': round(r['within_ce']-base['within_ce'], 4), 'class_cost': round(r['class_ce']-base['class_ce'], 4)}
        print(f"{tag} attn-output-ablate: within +{out['per_group'][tag]['within_cost']} class +{out['per_group'][tag]['class_cost']} (banding ref {out['banding_ref'][tag]})", flush=True)
    for h in hooks: h.remove()
    wc = {t: out['per_group'][t]['within_cost'] for t in out['per_group']}
    top = max(wc, key=wc.get)
    out['top_group'] = top
    out['pred_a_L35_top'] = bool(top == 'L3_5')
    out['pred_b_ablate_lower_than_band'] = bool(wc['L3_5'] < out['banding_ref']['L3_5'])
    g35 = out['per_group']['L3_5']
    out['pred_c_content_gt_grammar'] = bool(g35['within_cost'] > abs(g35['class_cost']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top group (output-ablation) {top} | L3-5 within {wc['L3_5']} vs banding 0.520", flush=True)
    print(f"pred_a L3-5 top {out['pred_a_L35_top']} | pred_b ablate<band {out['pred_b_ablate_lower_than_band']} | pred_c content>grammar {out['pred_c_content_gt_grammar']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
