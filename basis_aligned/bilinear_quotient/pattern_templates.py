"""OPEN-PROBLEM #2 (the last designed thread): the middle pool's key-SELECTION criterion — ~33% of the band's
collective value, surviving distance/content-sim/induction/mass (§1108 scoreboard). NEW INSTRUMENT, applying the
§1140 lesson (the mass driver was 'unnamed' only until a nonlinear probe replaced hand-features): decompose each
head's attention ROWS into offset-aligned TEMPLATES, then ask whether the template MIXTURE is decodable from the
query's residual. Method: for middle heads, collect each query's key-weight row aligned by relative offset
(window 96); subtract the head's mean row (= the §1099 distance kernel); SVD the residuals → row-templates.
Then: (i) how much residual row variance do the top-2 templates carry? (ii) are their per-query coefficients
DECODABLE from the query residual (MLP probe, held-out)?

REGISTERED PREDICTIONS:
  (0) SANITY: template-0 of the RAW rows ≈ the distance kernel (cos >= 0.8 with §1099's kernel shape);
      residual templates orthogonal to it.
  (a) NAMEABLE SELECTION: top-2 residual templates carry >= 40% of residual row variance AND their coefficients
      are decodable (MLP probe R² >= 0.4) -> the selection criterion = state-dependent mixing of a few named
      row-shapes; naming step next (what the templates DO: where they put mass, what predicts their use);
  (b) INTERACTION-IRREDUCIBLE: templates carry variance but coefficients are NOT decodable from the query state
      (R² < 0.2) -> selection genuinely lives in query-key interactions; the pause becomes a justified boundary;
  (c) UNSTRUCTURED: long-tail template spectrum (top-2 < 20%) -> row-level selection has no low-rank structure
      (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pattern_templates_results.json'
NSEQ = 64; SEQ = 256; W = 96; LMID = [8, 10, 12]
H = m.transformer.h
MOD = sys.modules[type(H[0].attn).__module__]
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    T = SEQ - 1
    hs = [H[L].attn.register_forward_pre_hook(capx_hook(L)) for L in LMID]

    rows_all = {L: [] for L in LMID}; states = {L: [] for L in LMID}
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); fwd(idx)
        for L in LMID:
            x = CAPX[L].float()
            pat = pattern_for(H[L].attn, CAPX[L]).mean(1)      # B,T,T head-mean (band criterion, §1085 style)
            # offset-aligned rows for queries with full window
            for b in range(x.shape[0]):
                for qpos in range(W, T, 3):                    # stride 3 to keep N manageable
                    rows_all[L].append(pat[b, qpos, qpos-W+1:qpos+1].cpu())
                    states[L].append(x[b, qpos].cpu())
    for h in hs: h.remove()

    out = {'per_layer': {}}
    agg_named = 0; agg_layers = 0
    for L in LMID:
        R = torch.stack(rows_all[L], 0).to(DEV)                # N, W
        S = torch.stack(states[L], 0).to(DEV)                  # N, D
        rows_all[L] = None; states[L] = None
        mean_row = R.mean(0)
        # sanity: mean row vs distance-kernel shape (monotone-decay proxy: corr with itself is trivial;
        # report decay corr with 1/dist proxy)
        dists = torch.arange(W, 0, -1, device=DEV).float()
        kern_proxy = 1.0/dists
        c0 = float(F.cosine_similarity(mean_row.unsqueeze(0), kern_proxy.unsqueeze(0)).squeeze())
        Rc = R - mean_row
        U2, S2, Vt2 = torch.linalg.svd(Rc, full_matrices=False)
        var = (S2**2)
        top2_frac = float(var[:2].sum()/var.sum())
        coeffs = Rc @ Vt2[:2].T                                 # N,2 template coefficients
        # decodability probe per template coefficient
        N = S.shape[0]; ntr = int(0.7*N)
        perm = torch.randperm(N, generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
        tr, te = perm[:ntr], perm[ntr:]
        Xz = (S - S[tr].mean(0))/S[tr].std(0).clamp_min(1e-6)
        r2s = []
        for j in range(2):
            y = coeffs[:, j]; yz = (y - y[tr].mean())/y[tr].std().clamp_min(1e-6)
            net = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            with torch.enable_grad():
                for step in range(2500):
                    ii = tr[torch.randint(0, ntr, (4096,), device=DEV)]
                    loss = ((net(Xz[ii]).squeeze(-1) - yz[ii])**2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pr = torch.cat([net(Xz[te][i2:i2+8192]).squeeze(-1) for i2 in range(0, te.shape[0], 8192)], 0)
            r2s.append(round(1 - float(((pr - yz[te])**2).mean()/(yz[te]**2).mean()), 3))
        # template shapes for naming: where do they put mass (near/mid/far thirds of the window)?
        shape = {}
        for j in range(2):
            t2 = Vt2[j]
            thirds = [float(t2[:32].abs().sum()), float(t2[32:64].abs().sum()), float(t2[64:].abs().sum())]
            ssum = sum(thirds)
            shape[f'template{j}'] = [round(v/ssum, 3) for v in thirds]  # far/mid/near mass shares
        named = bool(top2_frac >= 0.4 and max(r2s) >= 0.4)
        agg_named += int(named); agg_layers += 1
        out['per_layer'][str(L)] = {'mean_row_decay_cos': round(c0, 3), 'top2_var_frac': round(top2_frac, 3),
                                    'coeff_probe_r2': r2s, 'template_mass_far_mid_near': shape, 'named': named}
        print(f"L{L}: mean-row decay-cos {c0:.3f} | top2 var {top2_frac:.3f} | coeff probe R2 {r2s} | shapes {shape}", flush=True)
        del R, S, Rc

    out['pred_a_nameable'] = bool(agg_named >= 2)
    out['pred_b_interaction'] = bool(agg_named == 0 and all(out['per_layer'][str(L)]['top2_var_frac'] >= 0.2 for L in LMID)
                                     and all(max(out['per_layer'][str(L)]['coeff_probe_r2']) < 0.2 for L in LMID))
    out['pred_c_unstructured'] = bool(all(out['per_layer'][str(L)]['top2_var_frac'] < 0.2 for L in LMID))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a nameable {out['pred_a_nameable']} | pred_b interaction {out['pred_b_interaction']} | pred_c unstructured {out['pred_c_unstructured']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
