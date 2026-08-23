"""THE DEFINITIVE LADDER (registered §1148): rerun every intervention grain under the SENSITIVE readout —
position-matched log-probs of each DONOR row's actual next tokens (does the eval run start predicting the
donor's words?). 4 donor rows; per donor, conditions: coherent-K16 (donor's traveling pattern) | shuffled-pos
(coherence destroyed) | uniform-K16 (donor's mean coords broadcast) | atoms-8 (donor's 8 SAE values broadcast) |
rotated-basis null. Specificity = own-donor target gain vs mean other-donor target gain (4x4 cross matrix per
condition).

REGISTERED PREDICTIONS:
  (0) SANITY: rotated null ~0 specificity; the readout DETECTS something for coherent (else this protocol
      cannot support any conclusion and says so).
  (a) COHERENCE LAW RE-CONFIRMED PROPERLY: coherent is donor-specific (own >= 2x other-donor) while shuffled,
      uniform, and atoms-8 are not -> §1147 restored with a valid instrument; §1146's grain claim restored in
      sensitive form;
  (b) RANK SUFFICES: uniform-K16 also donor-specific -> the style conclusion falls; a passage's mean coords
      already carry its address;
  (c) NOTHING SPECIFIC: even coherent fails here -> transport does not replicate in this setup; §1059-60's
      conditions differ in something still unidentified (report plainly)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grain_kl_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64; NATOM = 256; TOPK = 8; STEPS = 3000
H = m.transformer.h
enc = tiktoken.get_encoding('gpt2')
STEER = {'vec': None, 'alpha': 0.0}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def steer_hook(L):
    def h(mo, i_, o_):
        if STEER['vec'] is None: return None
        x = (i_[0] if isinstance(i_, tuple) else i_)
        if STEER['vec'] == 'field':
            fld = STEER['field'][:, :x.shape[1]] if STEER['field'].shape[0] >= x.shape[0] else None
            xm = x + STEER['field_cur'].to(x.dtype)
        else:
            d = STEER['delta']
            xm = x + (d.unsqueeze(-1)*STEER['vec']).to(x.dtype)
        y = mo.Down(mo.Left(xm)*mo.Right(xm)) + mo.Down_bias
        return y.to(o_.dtype)
    return h


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(n, d, generator=g)*0.1)
        self.Dm = torch.nn.Parameter(torch.randn(d, n, generator=g)*0.1)
        self.k = k
    def forward(self, x):
        a = x @ self.E.T
        top = a.topk(self.k, -1)
        code = torch.zeros_like(a).scatter_(-1, top.indices, top.values)
        return code @ self.Dm.T, code


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); nb = NSEQ; T = SEQ - 1

    # capture REF mlp inputs only
    capR = {L: [] for L in REF}
    hs = []
    for L in REF:
        def mkr(L):
            def h(mo, i_, o_): capR[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[L].mlp.register_forward_hook(mkr(L)))
    idsL = []
    for i in range(0, nb, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))

    devsum = None
    for L in REF:
        X = torch.cat(capR[L], 0); capR[L] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        dv = X - (xb/cn.clamp_min(1).unsqueeze(1))[tok]
        devsum = dv if devsum is None else devsum + dv; del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False)
    Uc = Vt[:K].T.contiguous()
    Cc = (dev @ Uc).contiguous(); del dev, devsum

    sae = TopKSAE(K, NATOM, TOPK, 0).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    with torch.enable_grad():
        for step in range(STEPS):
            ii = torch.randint(0, Cc.shape[0], (4096,), device=DEV)
            xh, _ = sae(Cc[ii]); loss = ((xh - Cc[ii])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    _, code = sae(Cc)
    usage = (code != 0).float().mean(0)
    atoms = usage.argsort(descending=True)[:8].tolist()

    # affinity sets: top tokens by mean atom activation at the PRECEDING position
    aff = {}
    nextok = torch.roll(tok, -1)
    for a in atoms:
        act = code[:, a]
        sums = torch.zeros(V, device=DEV); sums.index_add_(0, nextok, act)
        mean_act = sums / cn.clamp_min(5)
        mean_act[cn < 5] = -1e9
        aff[a] = mean_act.argsort(descending=True)[:100]
    ov = []
    al = list(atoms)
    for i2 in range(len(al)):
        for j2 in range(i2+1, len(al)):
            ov.append(len(set(aff[al[i2]].tolist()) & set(aff[al[j2]].tolist()))/100)
    print(f"atoms {atoms} | affinity pairwise overlap mean {sum(ov)/len(ov):.3f}", flush=True)

    # ATOM INTERCHANGE at REF mlp inputs
    dirs = {a: F.normalize(Uc @ sae.Dm[:, a].detach(), dim=0) for a in atoms}
    hks = [H[L].mlp.register_forward_hook(steer_hook(L)) for L in REF]
    ev_n = 64
    ev_blocks = blocks[:ev_n]
    code_ev = code[:ev_n*T].view(ev_n, T, NATOM)
    g = torch.Generator(device=DEV).manual_seed(0)
    @torch.no_grad()
    def mean_logprob_sets(sets):
        lp_sums = {nm: 0.0 for nm in sets}; ncount = 0
        for i in range(0, ev_n, 8):
            idx = ev_blocks[i:i+8].to(DEV)[:, :-1].contiguous()
            if STEER['vec'] == 'field':
                STEER['field_cur'] = STEER['field'][i:i+8]
            elif STEER['vec'] is not None:
                STEER['delta'] = STEER['delta_full'][i:i+8]
            lp = F.log_softmax(fwd(idx).float(), -1)
            for nm, tset in sets.items():
                lp_sums[nm] += float(lp[..., tset].mean())
            ncount += 1
        return {nm: lp_sums[nm]/ncount for nm in sets}
    sets = {str(a): aff[a] for a in atoms}
    STEER['vec'] = None
    base_lp = mean_logprob_sets(sets)
    # sensitive-readout ladder
    Cc_ev = Cc[:ev_n*T].view(ev_n, T, K)
    kk = 16; Usub = Uc[:, :kk]
    code_rows = code[:ev_n*T].view(ev_n, T, NATOM).mean(1)
    g3 = torch.Generator(device=DEV).manual_seed(5)
    donors = []
    for j, a in enumerate(atoms[:4]):
        donors.append(int(code_rows[:, a].argsort(descending=True)[j]))
    # donor target tokens = the donor row's own next-token at each position (teacher-forced): tok[t+1]
    @torch.no_grad()
    def donor_lp():
        """mean log-prob assigned by the (possibly intervened) eval runs to each donor's next tokens, matched positions"""
        sums = {dj: 0.0 for dj in donors}; ncount = 0
        for i in range(0, ev_n, 8):
            idx = ev_blocks[i:i+8].to(DEV)[:, :-1].contiguous()
            if STEER['vec'] == 'field':
                STEER['field_cur'] = STEER['field'][i:i+8]
            lp = F.log_softmax(fwd(idx).float(), -1)          # 8,T,V
            for dj in donors:
                tgt = ev_blocks[dj].to(DEV)[1:idx.shape[1]+1]  # donor next tokens, positions 1..T
                sums[dj] += float(lp.gather(-1, tgt.view(1, -1, 1).expand(lp.shape[0], -1, 1)).mean())
            ncount += 1
        return {dj: sums[dj]/ncount for dj in donors}
    STEER['vec'] = None
    base_dlp = donor_lp()
    g4 = torch.Generator(device=DEV).manual_seed(17)
    Urot = (Uc @ torch.linalg.qr(torch.randn(K, K, generator=g4, device=DEV))[0])[:, :kk]
    res_g = {}
    for nm2 in ['coherent', 'shuffled_pos', 'uniform', 'atoms8', 'rotated']:
        owns = []; oths = []
        for j, dj in enumerate(donors):
            donor_pattern = Cc_ev[dj, :, :kk]
            cur = Cc_ev[..., :kk]
            if nm2 == 'coherent':
                dd = (donor_pattern.view(1, T, kk) - cur) @ Usub.T
            elif nm2 == 'shuffled_pos':
                perm = torch.randperm(T, generator=g3, device=DEV)
                dd = (donor_pattern[perm].view(1, T, kk) - cur) @ Usub.T
            elif nm2 == 'uniform':
                dd = (donor_pattern.mean(0).view(1, 1, kk) - cur) @ Usub.T
            elif nm2 == 'atoms8':
                _, dcode = sae(Cc_ev[dj])
                dmean = dcode.mean(0)
                deltaD = torch.zeros(ev_n, T, D, device=DEV)
                for a2 in atoms:
                    dsc = float((Uc @ sae.Dm[:, a2].detach()).norm())
                    code_ev_a = code[:ev_n*T].view(ev_n, T, NATOM)[..., a2]
                    deltaD += ((float(dmean[a2]) - code_ev_a)*dsc).unsqueeze(-1)*dirs[a2]
                dd = deltaD
            else:
                dd = (donor_pattern.view(1, T, kk) - cur) @ Urot.T
            STEER['field'] = dd.expand(ev_n, T, D).contiguous() if dd.shape[0] == 1 else dd
            STEER['vec'] = 'field'
            dlp = donor_lp()
            STEER['vec'] = None; STEER['field'] = None
            owns.append(dlp[dj] - base_dlp[dj])
            oths.append(sum(dlp[dk] - base_dlp[dk] for dk in donors if dk != dj)/(len(donors)-1))
        mo_ = sum(owns)/len(owns); mt_ = sum(oths)/len(oths)
        res_g[nm2] = {'own_donor': round(mo_, 4), 'other_donor': round(mt_, 4),
                      'specificity': round(mo_/max(abs(mt_), 1e-4), 2)}
        print(f"{nm2:>13}: own-donor {mo_:+.4f} | other-donor {mt_:+.4f} | specificity {res_g[nm2]['specificity']}", flush=True)
    for h in hks: h.remove()

    out = {'atoms': atoms, 'donors': donors, 'conditions': res_g}
    rc = res_g['coherent']['specificity']
    out['pred_a_coherence_reconfirmed'] = bool(rc >= 2 and res_g['shuffled_pos']['specificity'] < 1.5
                                               and res_g['uniform']['specificity'] < 1.5)
    out['pred_b_rank_suffices'] = bool(res_g['uniform']['specificity'] >= 2)
    out['pred_c_nothing'] = bool(rc < 1.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ladder {out['conditions']} | pred_a reconfirmed {out['pred_a_coherence_reconfirmed']} | pred_b rank {out['pred_b_rank_suffices']} | pred_c nothing {out['pred_c_nothing']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
