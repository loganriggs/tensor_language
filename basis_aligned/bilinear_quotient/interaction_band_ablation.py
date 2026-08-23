"""FRONTIER (content machine): §992 found that deleting the multiplicative interaction at any SINGLE middle layer
costs almost nothing (~0.03 nats), because the middle is a REDUNDANT, DISTRIBUTED band (§940) -- each layer's content
contribution is masked by the others. So the middle's TRUE content magnitude is only visible when the interaction is
removed across the WHOLE band JOINTLY. Measure the super-additivity: joint-band interaction-ablation cost vs the sum
of per-layer costs. Uses §941/§992's certified LOSS instrument; interaction = u*w = (Lx-a)*(Rx-b), ablated by
replacing the MLP output with Down(a*Rx + b*Lx - a*b) + bias (drops u*w).

REGISTERED PREDICTIONS:
  (0a) NULL: full recompute (all target layers) reproduces baseline CE (hook math correct, cost ~0).
  (0b) SANITY: joint-band cost >= max single-layer cost in the band (removing more cannot help).
  (a) MIDDLE SUPER-ADDITIVE: the joint MIDDLE-band (L4,8,11,15) interaction-ablation cost is MUCH larger than the
      sum of the individual per-layer costs (ratio > ~1.5) AND substantial in absolute nats (> ~0.3) -> the middle
      content computation is real but redundantly distributed, so single-layer ablation understates it (confirms
      §940/§992);
  (b) CONTENT: the joint middle-band cost falls mostly on WITHIN-class CE (content), not class CE (grammar);
  (c) FRONT for contrast: the FRONT band (L0,1,2) is high-stakes per layer (§933), so its joint cost is closer to
      ADDITIVE (ratio nearer 1) -> report both bands' super-additivity ratios."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'interaction_band_ablation_results.json'
NCAL = 64; NEVAL = 160; SEQ = 256
FRONT = [0, 1, 2]; MIDDLE = [4, 8, 11, 15]
ALL = sorted(set(FRONT + MIDDLE))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
AB = {}
MODE = {'null': False, 'layers': set()}


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
        if L not in MODE['layers']: return None
        x = i_[0] if isinstance(i_, tuple) else i_
        Lx = mlp.Left(x); Rx = mlp.Right(x); a, b = AB[L]
        const_linear = a*Rx + b*Lx - a*b
        prod = const_linear + (Lx - a)*(Rx - b) if MODE['null'] else const_linear
        bias = mlp.Down.bias if mlp.Down.bias is not None else 0
        return F.linear(prod, mlp.Down.weight) + bias
    return h


@torch.no_grad()
def calibrate(blocks):
    sums = {L: None for L in ALL}; caps = {}; hs = []
    for L in ALL:
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
        for L in ALL:
            lx, rx = caps[L]
            if sums[L] is None: sums[L] = [lx.sum(0), rx.sum(0), lx.shape[0]]
            else: sums[L][0] += lx.sum(0); sums[L][1] += rx.sum(0); sums[L][2] += lx.shape[0]
    for h in hs: h.remove()
    for L in ALL:
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
    hooks = [m.transformer.h[L].mlp.register_forward_hook(ablate_hook_factory(L)) for L in ALL]

    def cost(layers, null=False):
        MODE['layers'] = set(layers); MODE['null'] = null
        r = split_ce(blocks, cidx, C)
        r['cost'] = round(r['full_ce'] - base['full_ce'], 4)
        r['within_cost'] = round(r['within_ce'] - base['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base['class_ce'], 4)
        return r

    MODE['layers'] = set(); base = split_ce(blocks, cidx, C)
    print(f"baseline {base}", flush=True)
    out = {'baseline': base}
    out['null_full_recompute'] = cost(ALL, null=True)
    print(f"null(full recompute all) cost {out['null_full_recompute']['cost']} (~0)", flush=True)
    # per-layer
    out['per_layer'] = {}
    for L in ALL:
        r = cost([L]); out['per_layer'][str(L)] = r
        print(f"L{L:>2} single: cost {r['cost']} (within +{r['within_cost']})", flush=True)
    # bands joint
    out['bands'] = {}
    for tag, band in [('front', FRONT), ('middle', MIDDLE)]:
        joint = cost(band)
        sum_single = round(sum(out['per_layer'][str(L)]['cost'] for L in band), 4)
        ratio = round(joint['cost']/max(sum_single, 1e-6), 2)
        out['bands'][tag] = {'joint': joint, 'sum_single': sum_single, 'superadd_ratio': ratio}
        print(f"BAND {tag} {band}: joint cost {joint['cost']} (within +{joint['within_cost']} class +{joint['class_cost']}) | sum-single {sum_single} | super-add ratio {ratio}", flush=True)
    for h in hooks: h.remove()
    mid = out['bands']['middle']; fr = out['bands']['front']
    out['pred_0a_null_ok'] = bool(abs(out['null_full_recompute']['cost']) < 0.02)
    out['pred_0b_joint_ge_max_single'] = bool(mid['joint']['cost'] >= max(out['per_layer'][str(L)]['cost'] for L in MIDDLE) - 0.01)
    out['pred_a_middle_superadditive'] = bool(mid['superadd_ratio'] > 1.5 and mid['joint']['cost'] > 0.3)
    out['pred_b_middle_content'] = bool(mid['joint']['within_cost'] > abs(mid['joint']['class_cost']))
    out['pred_c_front_more_additive'] = bool(fr['superadd_ratio'] < mid['superadd_ratio'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"MIDDLE joint {mid['joint']['cost']} vs sum-single {mid['sum_single']} (ratio {mid['superadd_ratio']}) | FRONT ratio {fr['superadd_ratio']}", flush=True)
    print(f"pred_a mid super-additive {out['pred_a_middle_superadditive']} | pred_b mid=content {out['pred_b_middle_content']} | pred_c front more additive {out['pred_c_front_more_additive']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
