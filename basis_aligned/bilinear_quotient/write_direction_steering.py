"""WRITE DIRECTION STEERING -- the positive control closing 619-621.

619-621 established: supervised readout PROBES (d = mean class output
minus generic) recover the READ axis, which is orthogonal to the WRITE
axis (the unembedding row W_U[class]), so pushing a probe cannot steer
its own feature -- P(class) drops or moves like noise. The decisive
complement: push the WRITE axis itself. If adding alpha*W_U[class] to
the final residual raises P(class) monotonically -- while the read
probe d does NOT (reproducing 619/621) and random does not -- then the
read/write distinction is proven causally: the write axis steers, the
read axis does not.

For newline and article, three steering directions are compared at the
same scale (alpha*0.25*||resid||*dir added to the final residual):
  WRITE = mean unit-normalized W_U[class] rows (the write axis);
  READ  = d_class, the supervised probe (618, the read axis);
  RAND  = a random matched-norm direction.

REGISTERED PREDICTIONS:
  (0) IDENTITY: alpha=0 is the clean forward pass;
  (a) WRITE STEERS FORWARD (the control): P(class) increases
      monotonically with alpha along W_U[class], for BOTH newline and
      article;
  (b) CONTRAST: along the READ probe d, P(class) does NOT increase
      (reproduces the 619/621 reversal), and along RANDOM it does not
      systematically increase -- so only the write axis steers;
  (c) report all three P(class) curves per class;
  NULL: the random direction is not monotonic increasing."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'write_direction_steering_results.json'
NFRESH = 48
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
CLASSES = {'newline': [198, 628], 'article': [257, 281, 262, 383]}


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    WU = m.lm_head.weight.detach().float()

    # capture mlp17 output once for the read-probe directions
    mlp17 = m.transformer.h[17].mlp
    cap = []
    hk = mlp17.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    hk.remove()
    O = torch.cat(cap, 0)
    nxt = fresh[:, 1:257].reshape(-1).numpy()

    resid_norm = None

    def run(direction, alpha, toks):
        nonlocal resid_norm
        pc = torch.zeros(NFRESH, T)
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous(); B = bb.shape[0]
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
            if resid_norm is None:
                resid_norm = float(x.norm(dim=-1).mean())
            xs = x + alpha * 0.25 * resid_norm * direction[None, None]
            lg = (30 * torch.tanh(m.lm_head(F.rms_norm(xs, (D,))) / 30)).float()
            p = F.softmax(lg, dim=-1)
            pc[i:i + B] = p[..., toks].sum(-1).cpu()
        return float(pc.mean())

    def curve(direction, toks):
        return [round(run(direction, a, toks), 5) for a in ALPHAS]

    def mono_inc(ps):
        return all(ps[i] <= ps[i + 1] + 1e-6 for i in range(len(ps) - 1)) \
            and ps[-1] > ps[0]

    results = {}
    all_write_ok = True
    for cname, toks in CLASSES.items():
        # WRITE axis: mean unit-normalized unembedding rows
        w = sum(WU[t] / WU[t].norm() for t in toks)
        w = (w / w.norm()).to(DEV)
        # READ axis: supervised probe
        isc = np.isin(nxt, toks)
        d = O[torch.tensor(isc)].mean(0) - O.mean(0)
        d = (d / d.norm()).to(DEV)
        # RANDOM
        gg = torch.Generator(device=DEV).manual_seed(0)
        r = torch.randn(D, generator=gg, device=DEV); r = r / r.norm()

        cw = curve(w, toks)
        cd = curve(d, toks)
        cr = curve(r, toks)
        write_ok = mono_inc(cw)
        read_ok = mono_inc(cd)
        rand_ok = mono_inc(cr)
        all_write_ok = all_write_ok and write_ok
        cos_wd = float((w @ d).cpu())
        results[cname] = {'write_curve': cw, 'read_curve': cd,
                          'rand_curve': cr, 'write_steers_fwd': write_ok,
                          'read_steers_fwd': read_ok, 'rand_steers_fwd': rand_ok,
                          'cos_write_read': round(cos_wd, 4)}
        print(f'{cname}: cos(write,read) {cos_wd:+.3f}', flush=True)
        print(f'  WRITE {cw} mono-inc {write_ok}', flush=True)
        print(f'  READ  {cd} mono-inc {read_ok}', flush=True)
        print(f'  RAND  {cr} mono-inc {rand_ok}', flush=True)

    pa = all_write_ok
    pb = all(not results[c]['read_steers_fwd'] and not results[c]['rand_steers_fwd']
             for c in CLASSES)
    print(f'\n(a) WRITE steers forward for all classes: {pa}', flush=True)
    print(f'(b) READ and RANDOM do not steer forward: {pb}', flush=True)

    out = {'results': results, 'pred_a_write_steers': bool(pa),
           'pred_b_read_rand_dont': bool(pb), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
