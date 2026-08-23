"""WHERE across the stack does the function-word HEDGE form? §973/§975: on hard/grammar-error positions the model's
top-1 is a common function word (a hedge). Localize it with the logit lens (transform-invariant argmax per layer,
§944): on positions that end up as GRAMMAR-ERRORS at the final layer, at each layer read the output head off that
layer's residual and measure the fraction whose top-1 is a FUNCTION-class word. If the hedge is present from the
FRONT (where grammar/class is written, §915) and persists, the hedge is a front-grammar-machine behavior; if it
appears only late, it is a readout behavior. Report the same curve on HIT positions for contrast.

REGISTERED PREDICTIONS:
  (0) SANITY: at the final layer, grammar-error positions have ~1.0 (by their §975 definition top-1 is often
      function) and the curve is defined at every layer.
  (a) HEDGE IS FRONT-FORMED: on grammar-error positions the logit-lens top-1-function-word fraction is already
      HIGH in the front layers and stays high -> the hedge is written by the front grammar machine (§915/§952),
      not invented at the readout;
  (b) report per-layer top-1-function-frac for grammar-error vs hit positions."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'hedge_depth_results.json'
NEVAL = 160; SEQ = 256; NLAYER = 18
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


def readout(x): return m.lm_head(F.rms_norm(x, (D,)))  # argmax invariant to the monotone clamp


def forward_all(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None; outs = []
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0); outs.append(x)
    return outs


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    is_func = torch.tensor(np.array([CLASSES[c] in FUNCTION for c in tok2cls]), device=DEV)
    # accumulate per-layer func-top1 counts split by final-position error type
    func_g = np.zeros(NLAYER); func_h = np.zeros(NLAYER); ng = 0; nh = 0
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        outs = forward_all(idx); tf = tgt.reshape(-1)
        final_pred = readout(outs[-1]).float().reshape(-1, V).argmax(1)
        is_hit = final_pred == tf; is_gram = (~is_hit) & (cidx[final_pred] != cidx[tf])
        gm = is_gram; hm = is_hit
        ng += int(gm.sum()); nh += int(hm.sum())
        for L in range(NLAYER):
            pred_L = readout(outs[L]).float().reshape(-1, V).argmax(1)
            fL = is_func[pred_L]
            func_g[L] += float(fL[gm].sum()); func_h[L] += float(fL[hm].sum())
    out = {'n_grammar_err': ng, 'n_hit': nh,
           'grammar_err_func_frac_by_layer': [round(func_g[L]/max(ng,1), 3) for L in range(NLAYER)],
           'hit_func_frac_by_layer': [round(func_h[L]/max(nh,1), 3) for L in range(NLAYER)]}
    g = out['grammar_err_func_frac_by_layer']
    out['front_L0_2_mean'] = round(float(np.mean(g[:3])), 3); out['back_L15_17_mean'] = round(float(np.mean(g[15:])), 3)
    out['pred_a_hedge_front_formed'] = bool(out['front_L0_2_mean'] > 0.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print("grammar-err func-top1 by layer:", g, flush=True)
    print("hit         func-top1 by layer:", out['hit_func_frac_by_layer'], flush=True)
    print(f"front(L0-2) {out['front_L0_2_mean']} vs back(L15-17) {out['back_L15_17_mean']}", flush=True)
    print(f"(a) hedge is front-formed: {out['pred_a_hedge_front_formed']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
