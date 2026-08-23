"""Do the deep-middle content MLPs SHARPEN content (Left~=Right, a self-product content^2) or CONJOIN two different
content criteria (Left != Right)? bilin18 MLP = Down[(Left.x) * (Right.x)] + bias (pure bilinear). §842 found mlp0
(grammar) is a SELF-PRODUCT (sharpening). The deep-middle content MLPs are untested -- this refines §1041's content x
content. Two tests per MLP (mlp0/4/8/10/12): (A) WEIGHT cosine between corresponding Left/Right rows (per neuron), mean
-- high => self-product; (B) FUNCTIONAL: force self-product by replacing Right(x) with Left(x) (output = Down[Left(x)^2]
+bias) and measure per-MLP CE cost -- small => the MLP is ~self-product (Right adds little beyond Left); large => Right
carries distinct info (conjunction).

REGISTERED PREDICTIONS:
  (0) SANITY: mlp0 weight-cosine high and force-self-product cost small (reproduces §842 self-product).
  (a) DEEP-MIDDLE IS CONJUNCTION (two criteria): deep-middle MLPs have LOWER Left/Right weight cosine than mlp0 AND a
      LARGE force-self-product CE cost (>~0.3 nats) -> they multiply two DIFFERENT content features, not content^2;
  (b) OR SHARPENING: deep-middle also ~self-product (high cosine, small cost). Report weight cosine + force-self-product
      cost per MLP + a force-Right^2 control + a random-neuron-permute null."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_gate_leftright_results.json'
NEVAL = 200; SEQ = 256; LAYERS = [0, 4, 8, 10, 12, 16]
SUB = {'L': None, 'mode': None}   # mode: None | 'LL' (Left^2) | 'RR' (Right^2) | 'perm' (permute Right neurons)


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook(L):
    mlp = m.transformer.h[L].mlp
    def h(mo, i_, o_):
        if SUB['L'] != L or SUB['mode'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        Lx = mlp.Left(x); Rx = mlp.Right(x)
        if SUB['mode'] == 'LL': g = Lx * Lx
        elif SUB['mode'] == 'RR': g = Rx * Rx
        elif SUB['mode'] == 'perm':
            p = torch.randperm(Rx.shape[-1], device=DEV); g = Lx * Rx[..., p]
        ny = mlp.Down(g) + mlp.Down_bias
        return ny.to((o_[0] if isinstance(o_, tuple) else o_).dtype)
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(fwd(idx).float(), -1); tf = tgt.reshape(-1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tf.shape[0], device=DEV), tf].sum()); n += tf.shape[0]
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous()
    hooks = [m.transformer.h[L].mlp.register_forward_hook(hook(L)) for L in LAYERS]
    SUB['L'] = None; base = ce(blocks)
    out = {'base_ce': round(base, 4), 'per_mlp': {}}
    for L in LAYERS:
        mlp = m.transformer.h[L].mlp
        Lw = mlp.Left.weight.data.float(); Rw = mlp.Right.weight.data.float()   # (HID, D)
        cos = F.cosine_similarity(Lw, Rw, dim=1)                                # per-neuron
        SUB['L'] = L
        SUB['mode'] = 'LL'; ce_ll = ce(blocks)
        SUB['mode'] = 'RR'; ce_rr = ce(blocks)
        SUB['mode'] = 'perm'; ce_perm = ce(blocks)
        SUB['mode'] = None; SUB['L'] = None
        out['per_mlp'][str(L)] = {
            'weight_cos_mean': round(float(cos.mean()), 4), 'weight_cos_abs_mean': round(float(cos.abs().mean()), 4),
            'force_Lsq_cost': round(ce_ll - base, 4), 'force_Rsq_cost': round(ce_rr - base, 4),
            'perm_right_null_cost': round(ce_perm - base, 4)}
        print(f"mlp{L}: wcos {out['per_mlp'][str(L)]['weight_cos_mean']} (|.| {out['per_mlp'][str(L)]['weight_cos_abs_mean']}) | force L^2 cost {out['per_mlp'][str(L)]['force_Lsq_cost']} | R^2 {out['per_mlp'][str(L)]['force_Rsq_cost']} | perm-null {out['per_mlp'][str(L)]['perm_right_null_cost']}", flush=True)
    for h in hooks: h.remove()
    mid = [8, 10, 12]
    out['mlp0_force_Lsq_cost'] = out['per_mlp']['0']['force_Lsq_cost']
    out['midmean_force_Lsq_cost'] = round(sum(out['per_mlp'][str(L)]['force_Lsq_cost'] for L in mid)/len(mid), 4)
    out['mlp0_wcos'] = out['per_mlp']['0']['weight_cos_mean']
    out['midmean_wcos'] = round(sum(out['per_mlp'][str(L)]['weight_cos_mean'] for L in mid)/len(mid), 4)
    out['pred_a_midconjunction'] = bool(out['midmean_force_Lsq_cost'] > 0.3 and out['midmean_wcos'] < out['mlp0_wcos'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"mlp0 wcos {out['mlp0_wcos']} force-L^2 {out['mlp0_force_Lsq_cost']} | mid wcos {out['midmean_wcos']} force-L^2 {out['midmean_force_Lsq_cost']} | pred_a mid-conjunction {out['pred_a_midconjunction']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
