"""CAUSAL CAPSTONE of the middle-attention account (§1099 kernel 58% + §1141 saturation-mass + §1144 near-field
templates — all named via probes, i.e. correlational/decodable so far). Convert the account into a GENERATIVE
STAND-IN and score it causally, held-out: pattern_hat[h,q,k] = alpha_hat(q) · k_h(q−k) + Σ_{j=1,2} c_hat_j(q) ·
t_j(q−k), where k_h = per-head distance kernel (fit on half A), t_j = the §1144 head-mean row templates (fit on
A), and alpha_hat / c_hat_j are MLP probes from the query residual (trained on A). Values stay dynamic. Applied
to the middle band L6-14 on half B; compared against const (band floor) and kernel-only (§1099 baseline 0.583).

REGISTERED PREDICTIONS:
  (0) SANITY: kernel-only reproduces ~0.58 on half B; const floor comparable to §1099's +1.15.
  (a) ACCOUNT CERTIFIES CAUSALLY: the named three-term stand-in recovers >= 0.75 of the band's collective value
      -> the middle-attention band joins the benchmark's understood set with a fully NAMED generative model
      (kernel × saturation-mass + near-field shaping); benchmark/dossier updated;
  (b) NAMES DON'T CONVERT: recovery <= 0.65 -> the probes' decodability is correlational only (the real pattern
      uses the state differently than the probes do) — report plainly; the §1141/§1144 names stay
      representation-level;
  (c) ablation of terms: report kernel+mass (no templates) and kernel+templates (no mass) to attribute the gain."""
import json, time, sys, types, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_named_standin_results.json'
NSEQ = 256; SEQ = 256; W = 96; BAND = list(range(6, 15))
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
CTL = {'mode': None}
ST = {}
CAPX = {}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capx_hook(L):
    def h(mo, args): CAPX[L] = args[0].detach()
    return h


@torch.no_grad()
def pattern_for(attn, x):
    B, T, C = x.shape
    q = attn.c_q(x).view(B, T, NH, HD); k = attn.c_k(x).view(B, T, NH, HD)
    q2 = attn.c_q2(x).view(B, T, NH, HD); k2 = attn.c_k2(x).view(B, T, NH, HD)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (HD,)), F.rms_norm(k, (HD,))
    q, k = MOD.apply_rotary_emb(q, cos, sin), MOD.apply_rotary_emb(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (HD,)), F.rms_norm(k2, (HD,))
    q2, k2 = MOD.apply_rotary_emb(q2, cos, sin), MOD.apply_rotary_emb(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pat = (s1/HD)*(s2/HD)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return pat.masked_fill_(mask.logical_not(), 0.0)


class Probe(torch.nn.Module):
    def __init__(self, dout, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = torch.nn.Linear(D, 256); self.l2 = torch.nn.Linear(256, dout)
        with torch.no_grad():
            self.l1.weight.copy_(torch.randn(256, D, generator=g)*0.02)
            self.l2.weight.copy_(torch.randn(dout, 256, generator=g)*0.02)
    def forward(self, x): return self.l2(torch.relu(self.l1(x)))


def train_probe(X, Y, seed):
    net = Probe(Y.shape[1], seed).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    with torch.enable_grad():
        for step in range(2500):
            ii = torch.randint(0, X.shape[0], (4096,), device=DEV)
            loss = ((net(X[ii]) - Y[ii])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def make_sq(attn, L):
    orig = attn.squared_attention
    def patched(self, q, k, v, q2, k2):
        B, T, Hh, Dh = q.shape
        md = CTL['mode']
        if md is None or L not in CTL['band']:
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k); s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
            pat = (s1/Dh)*(s2/Dh)
            cm = torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool))
            pat = pat.masked_fill(cm.logical_not(), 0.0)
        else:
            pat = ST[f'kern_full_{L}'][:, :T, :T].unsqueeze(0).expand(B, NH, T, T).clone().to(v.dtype)
            if md in ('mass', 'full'):
                a = ST['alpha_cur'][L]                     # B,T predicted mass
                pat = pat * a.unsqueeze(1).unsqueeze(-1).to(pat.dtype)
            if md in ('tmpl', 'full'):
                c = ST['coeff_cur'][L]                     # B,T,2 predicted template coefficients
                tm = ST[f'tmpl_full_{L}']                  # 2,T,T banded template matrices
                add = torch.einsum('btj,jtk->btk', c.to(pat.dtype), tm[:, :T, :T].to(pat.dtype))
                pat = pat + add.unsqueeze(1)
        return torch.einsum('bhqk,bkhd->bhqd', pat, v)
    return orig, types.MethodType(patched, attn)


def const_hook(L):
    def h(mo, args):
        if CTL['mode'] != 'const' or L not in CTL['band']: return None
        y = args[0].clone()
        for hh in range(NH):
            y[..., hh*HD:(hh+1)*HD] = ST[f'means_{L}'][hh].view(1, 1, HD).to(y.dtype)
        return (y,) + tuple(args[1:])
    return h


def state_hook(L):
    """at eval time: compute alpha/coeff predictions from the live attn input"""
    def h(mo, args):
        if CTL['mode'] in ('mass', 'tmpl', 'full') and L in CTL['band']:
            x = args[0].detach().float()
            Xz = (x - ST['mu_x'][L])/ST['sd_x'][L]
            with torch.no_grad():
                a = ST[f'probeA_{L}'](Xz.reshape(-1, D)).view(x.shape[0], x.shape[1])
                ST['alpha_cur'][L] = (a*ST['a_sd'][L] + ST['a_mu'][L]).clamp(0.1, 10.0)
                cc = ST[f'probeC_{L}'](Xz.reshape(-1, D)).view(x.shape[0], x.shape[1], 2)
                ST['coeff_cur'][L] = cc*ST['c_sd'][L] + ST['c_mu'][L]
        return None
    return h


@torch.no_grad()
def ce(blocks):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    A = rows[:NSEQ//2]; B = rows[NSEQ//2:]
    T = SEQ - 1
    di = torch.arange(T, device=DEV).view(-1, 1) - torch.arange(T, device=DEV).view(1, -1)

    hs = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in BAND]
    means = {L: torch.zeros(NH, HD, device=DEV) for L in BAND}
    hm = []
    for L in BAND:
        def mk(L):
            def h(mo, args): means[L] += args[0].detach().float().reshape(-1, NH, HD).sum(0)
            return h
        hm.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    ksum = {L: torch.zeros(NH, T, device=DEV) for L in BAND}
    kcnt = torch.zeros(T, device=DEV)
    rowsA = {L: [] for L in BAND}; statesA = {L: [] for L in BAND}; massA = {L: [] for L in BAND}
    npos = 0
    for i in range(0, A.shape[0], 8):
        idx = A[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx); npos += idx.numel()
        for L in BAND:
            pat = pattern_for(H[L].attn, CAPX[L])
            for dd in range(0, T, 1):
                mask = (di == dd); nel = int(mask.sum())
                if nel == 0: continue
                ksum[L][:, dd] += pat[:, :, mask].sum((0, 2))
                if L == BAND[0]: kcnt[dd] += nel*pat.shape[0]
            pm = pat.mean(1)                                # head-mean B,T,T
            x = CAPX[L].float()
            for b in range(x.shape[0]):
                for qpos in range(W, T, 4):
                    rowsA[L].append(pm[b, qpos, qpos-W+1:qpos+1])
                    statesA[L].append(x[b, qpos])
            # per-position row mass ratio target for alpha (all positions)
            rm = pat.abs().sum(-1).mean(1)                  # B,T head-mean row mass
            massA[L].append((rm, x))
            del pat
    for h in hm: h.remove()
    for h in hs: h.remove()

    for L in BAND:
        ST[f'means_{L}'] = means[L]/npos
        kern = ksum[L]/kcnt.clamp_min(1)                    # NH,T
        km = torch.zeros(NH, T, T, device=DEV)
        for dd in range(T):
            iidx = torch.arange(dd, T, device=DEV)
            km[:, iidx, iidx-dd] = kern[:, dd].unsqueeze(1)
        ST[f'kern_full_{L}'] = km
        # kernel head-mean row mass by position (for alpha ratio)
        kmass_pos = torch.cumsum(kern.abs().mean(0), dim=0) # T
        # templates from row residuals (head-mean, window)
        R = torch.stack(rowsA[L], 0); S = torch.stack(statesA[L], 0)
        rowsA[L] = None; statesA[L] = None
        mean_row = R.mean(0)
        Rc = R - mean_row
        _, _, Vt2 = torch.linalg.svd(Rc, full_matrices=False)
        tmpl = Vt2[:2]                                       # 2,W
        tmats = torch.zeros(2, T, T, device=DEV)
        for j in range(2):
            for dd in range(W):
                iidx = torch.arange(max(dd, W-1), T, device=DEV)
                # template index: offset dd from query means key at q-(W-1-dd)... align: row vector covers offsets W-1..0
                pass
        # simpler: banded fill — row[q, q-W+1+r] = tmpl[j, r]
        for j in range(2):
            for r in range(W):
                qs = torch.arange(W-1, T, device=DEV)
                ks2 = qs - (W-1) + r
                tmats[j, qs, ks2] = tmpl[j, r]
        ST[f'tmpl_full_{L}'] = tmats
        coeffs = Rc @ tmpl.T                                 # N,2
        # probes (standardized targets)
        mu_x = S.mean(0); sd_x = S.std(0).clamp_min(1e-6)
        ST['mu_x'] = ST.get('mu_x', {}); ST['sd_x'] = ST.get('sd_x', {})
        ST['mu_x'][L] = mu_x; ST['sd_x'][L] = sd_x
        Xz = (S - mu_x)/sd_x
        c_mu = coeffs.mean(0); c_sd = coeffs.std(0).clamp_min(1e-6)
        ST['c_mu'] = ST.get('c_mu', {}); ST['c_sd'] = ST.get('c_sd', {})
        ST['c_mu'][L] = c_mu; ST['c_sd'][L] = c_sd
        ST[f'probeC_{L}'] = train_probe(Xz, (coeffs-c_mu)/c_sd, seed=L)
        # alpha targets: real head-mean row mass / kernel mass at each captured full-window position
        rm_all = []; xs_all = []
        for (rm, x) in massA[L]:
            for b in range(x.shape[0]):
                rm_all.append(rm[b, W:]); xs_all.append(x[b, W:])
        massA[L] = None
        rmv = torch.cat(rm_all, 0); xsv = torch.cat(xs_all, 0).reshape(-1, D)
        alph = rmv / kmass_pos[W:].repeat(len(rm_all)).clamp_min(1e-6)
        a_mu = alph.mean(); a_sd = alph.std().clamp_min(1e-6)
        ST['a_mu'] = ST.get('a_mu', {}); ST['a_sd'] = ST.get('a_sd', {})
        ST['a_mu'][L] = a_mu; ST['a_sd'][L] = a_sd
        Xz2 = (xsv - mu_x)/sd_x
        ST[f'probeA_{L}'] = train_probe(Xz2, ((alph-a_mu)/a_sd).unsqueeze(1), seed=100+L)
        print(f"L{L}: kernel+templates+probes fit", flush=True)

    # eval on half B
    origs = {}
    for L in BAND:
        o, p = make_sq(H[L].attn, L); origs[L] = o; H[L].attn.squared_attention = p
    hstate = [H[L].attn.register_forward_pre_hook(state_hook(L)) for L in BAND]
    hconst = [H[L].attn.c_proj.register_forward_pre_hook(const_hook(L)) for L in BAND]
    ST['alpha_cur'] = {}; ST['coeff_cur'] = {}
    CTL['band'] = set(BAND); CTL['mode'] = None
    base = ce(B)
    res = {}
    for md in ['const', 'kernel', 'mass', 'tmpl', 'full']:
        CTL['mode'] = md if md != 'kernel' else 'kern'
        if md == 'kernel': CTL['mode'] = 'kern'
        # 'kern' mode: kernel only — implement as mode not in (mass/tmpl/full) but in band → need explicit
        CTL['mode'] = {'const': 'const', 'kernel': 'kern', 'mass': 'mass', 'tmpl': 'tmpl', 'full': 'full'}[md]
        res[md] = round(ce(B) - base, 4)
        CTL['mode'] = None
        print(f"{md:>7}: cost +{res[md]}", flush=True)
    for L in BAND: H[L].attn.squared_attention = origs[L]
    for h in hstate + hconst: h.remove()

    fl = max(res['const'], 1e-6)
    rec = {md: round(1 - res[md]/fl, 3) for md in ['kernel', 'mass', 'tmpl', 'full']}
    out = {'base_ce': round(base, 4), 'costs': res, 'recovery': rec, 'kernel_ref_1099': 0.583}
    out['pred_a_certifies'] = bool(rec['full'] >= 0.75)
    out['pred_b_names_dont_convert'] = bool(rec['full'] <= 0.65)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"recovery: {rec}", flush=True)
    print(f"pred_a certifies {out['pred_a_certifies']} | pred_b no-convert {out['pred_b_names_dont_convert']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
