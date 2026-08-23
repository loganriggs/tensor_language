"""DEFINITIVE behavioral test of the §941-vs-§990 resolution (§991): is the bilinear MLP's multiplicative INTERACTION
term behaviorally INERT at the FRONT but LOAD-BEARING in the MIDDLE? Use §941's certified LOSS instrument, not raw
output variance.

The MLP computes Down[(Left.x)*(Right.x)] + bias. Split the product Lx*Rx exactly (a=mean(Lx), b=mean(Rx) over a
calibration set; u=Lx-a, w=Rx-b):
  Lx*Rx = a*b + (a*w + b*u) + (u*w)   =  const + LINEAR-in-x  +  INTERACTION
        = (a*Rx + b*Lx - a*b)          + (u*w)
Ablate ONLY the interaction: replace the MLP output with Down(a*Rx + b*Lx - a*b) + bias (drops u*w), and measure the
CE cost vs baseline. If §991's resolution is right, at the FRONT this costs ~nothing (the interaction is present and
dominant in the raw output, §990, but behaviorally inert), while in the MIDDLE it costs substantially (the
interaction is the load-bearing content computation, §941 loss-frac 0.38).

REGISTERED PREDICTIONS:
  (0a) NULL: replacing the output with the FULL recompute Down(a*Rx+b*Lx-a*b + u*w)+bias = Down(Lx*Rx)+bias
       reproduces baseline CE exactly (hook math correct, cost ~0).
  (0b) SANITY: per-layer interaction-ablation cost >= 0 everywhere (removing a real term cannot help).
  (a) FRONT INERT / MIDDLE LOAD-BEARING: the interaction-ablation CE cost at the FRONT (L0-2) is NEAR-ZERO and MUCH
      smaller than at the MIDDLE (L8-11), directly confirming §941/§991 on the behavioral (loss) instrument -> the
      front's near-linearity (§941) is that its interaction is behaviorally inert, NOT that it is absent (§990);
  (b) CONTENT: the middle-band interaction-ablation cost falls mostly on WITHIN-class CE (content), not class CE
      (grammar), matching the content=middle=multiplicative account;
  (c) report per-layer cost, front-band vs middle-band cost, and the chain-rule split of each band."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'interaction_loss_ablation_results.json'
NCAL = 64; NEVAL = 160; SEQ = 256
LAYERS = [0, 1, 2, 4, 8, 11, 15, 17]
FRONT = [0, 1, 2]; MIDDLE = [8, 11]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
AB = {}    # per-layer (a,b) constants
MODE = {'on': False, 'null': False, 'layers': set()}


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


def ablate_hook_factory(L):
    mlp = m.transformer.h[L].mlp
    def h(mo, i_, o_):
        if not MODE['on'] or L not in MODE['layers']: return None  # leave output unchanged
        x = i_[0] if isinstance(i_, tuple) else i_
        Lx = mlp.Left(x); Rx = mlp.Right(x)
        a, b = AB[L]
        const_linear = a*Rx + b*Lx - a*b            # const + linear-in-x
        prod = const_linear + (Lx - a)*(Rx - b) if MODE['null'] else const_linear  # null re-adds interaction
        bias = mlp.Down.bias if mlp.Down.bias is not None else 0
        return F.linear(prod, mlp.Down.weight) + bias
    return h


@torch.no_grad()
def calibrate(blocks):
    # per-layer running mean of Left(x), Right(x)
    sums = {L: None for L in LAYERS}; cnt = 0; caps = {}
    hs = []
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L, mlp):
            def h(mo, i_, o_):
                x = i_[0] if isinstance(i_, tuple) else i_
                caps[L] = (mlp.Left(x).detach().float().reshape(-1, mlp.Left.weight.shape[0]),
                           mlp.Right(x).detach().float().reshape(-1, mlp.Right.weight.shape[0]))
            return h
        hs.append(mlp.register_forward_hook(mk(L, mlp)))
    for i in range(0, blocks.shape[0], 8):
        forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
        for L in LAYERS:
            lx, rx = caps[L]; s = (lx.sum(0), rx.sum(0), lx.shape[0])
            if sums[L] is None: sums[L] = [s[0].clone(), s[1].clone(), s[2]]
            else: sums[L][0] += s[0]; sums[L][1] += s[1]; sums[L][2] += s[2]
    for h in hs: h.remove()
    for L in LAYERS:
        n = sums[L][2]; AB[L] = (sums[L][0].div(n).view(1, -1), sums[L][1].div(n).view(1, -1))


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    full = tot/n; classce = totc/n
    return {'full_ce': round(full, 4), 'class_ce': round(classce, 4), 'within_ce': round(full-classce, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    calibrate(calib)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(ablate_hook_factory(L)) for L in LAYERS]
    out = {}
    MODE['on'] = False
    base = split_ce(blocks, cidx, C); out['baseline'] = base
    print(f"baseline {base}", flush=True)
    # NULL: full recompute at all layers -> must equal baseline
    MODE['on'] = True; MODE['null'] = True; MODE['layers'] = set(LAYERS)
    out['null_full_recompute'] = split_ce(blocks, cidx, C)
    print(f"null(full recompute all) {out['null_full_recompute']} (should ~= baseline)", flush=True)
    # per-layer interaction ablation
    MODE['null'] = False; out['per_layer'] = {}
    for L in LAYERS:
        MODE['layers'] = {L}
        r = split_ce(blocks, cidx, C)
        r['cost'] = round(r['full_ce'] - base['full_ce'], 4)
        r['within_cost'] = round(r['within_ce'] - base['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base['class_ce'], 4)
        out['per_layer'][str(L)] = r
        print(f"L{L:>2} interaction-ablate: cost {r['cost']} (within +{r['within_cost']} class +{r['class_cost']})", flush=True)
    # bands
    for tag, band in [('front', FRONT), ('middle', MIDDLE)]:
        MODE['layers'] = set(band)
        r = split_ce(blocks, cidx, C)
        r['cost'] = round(r['full_ce'] - base['full_ce'], 4)
        r['within_cost'] = round(r['within_ce'] - base['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base['class_ce'], 4)
        out[f'band_{tag}'] = r
        print(f"BAND {tag} {band} interaction-ablate: cost {r['cost']} (within +{r['within_cost']} class +{r['class_cost']})", flush=True)
    for h in hooks: h.remove()
    fc = out['band_front']['cost']; mc = out['band_middle']['cost']
    nullcost = abs(out['null_full_recompute']['full_ce'] - base['full_ce'])
    out['pred_0a_null_ok'] = bool(nullcost < 0.02)
    out['pred_0b_costs_nonneg'] = bool(all(out['per_layer'][str(L)]['cost'] > -0.02 for L in LAYERS))
    out['pred_a_front_inert_mid_loadbearing'] = bool(fc < 0.05 and mc > fc + 0.05)
    out['pred_b_mid_content'] = bool(out['band_middle']['within_cost'] > abs(out['band_middle']['class_cost']))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front-band cost {fc} vs middle-band cost {mc} | null cost {nullcost:.4f}", flush=True)
    print(f"pred_a front-inert/mid-loadbearing {out['pred_a_front_inert_mid_loadbearing']} | pred_b mid=content {out['pred_b_mid_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
