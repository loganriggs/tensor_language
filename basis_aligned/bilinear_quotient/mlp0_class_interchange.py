"""GOLD-STANDARD instrument for §1104 (registered there): deletion can't show per-class causality because the
class variable is redundantly multiplexed (attn0 copy + x0 re-injection compensate). INTERCHANGE changes the
VALUE instead of removing the wire — it worked where ablation/steering failed for class (§892, IIA 0.25) and
topic (§894). Here at PER-CLASS granularity: at mlp0's output, swap each position's CLASS-PACKAGE projection
(top-24 subspace of token-mean outputs) with that of a DONOR position of a DIFFERENT class (pair positions of
class A with donors of class B). Readout: the predicted NEXT-TOKEN-CLASS distribution. Per §828, next-class
follows the class bigram; if the swapped-in code carries CLASS-B-ness, the next-class prediction at A-positions
should move TOWARD the class-bigram successor profile of B. Metric: interchange accuracy = fraction of swapped
positions where the next-class distribution moves closer (KL) to the DONOR-class successor profile than to the
original-class profile. NULLS: (i) random-subspace swap of matched rank (should do ~nothing class-shaped);
(ii) same-class donor swap (should not move the profile). Pairs tested: determiner<->number,
punctuation<->pronoun, preposition<->aux (6 directed pairs). NSEQ=192.

REGISTERED PREDICTIONS:
  (0) SANITY: same-class donor swaps move next-class KL little; base successor profiles differ strongly across
      classes (bigram check).
  (a) PER-CLASS CAUSAL CODE: directed interchange moves the next-class prediction toward the DONOR class's
      successor profile in >= 5/6 directed pairs (accuracy > 0.5 vs the toward-original baseline, and KL-shift
      >= 3x the random-subspace null) -> the within-package coordinates carry class-specific causal content:
      the §1098 clusters are causally distinct VALUES of the class variable (the strong story, via the right
      instrument);
  (b) if interchange also fails class-specifically, the class information used downstream is NOT carried by
      mlp0's output at all (fully re-derived from x0 downstream) — report plainly; then the §1098 map is a
      write-side epiphenomenon and the story changes."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_class_interchange_results.json'
NSEQ = 192; SEQ = 256; MIN_OCC = 8; KPKG = 24
H = m.transformer.h
enc = tiktoken.get_encoding('gpt2')
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'each', 'every', 'no', 'all', 'both'}
PREP = {'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'into', 'over', 'under', 'about', 'after',
        'before', 'between', 'through', 'during', 'against', 'without', 'within', 'upon', 'across', 'off', 'up', 'down', 'out'}
PRON = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'his', 'its', 'their',
        'my', 'your', 'our', 'who', 'whom', 'which', 'what', 'himself', 'herself', 'itself', 'themselves'}
CONJ = {'and', 'or', 'but', 'so', 'because', 'if', 'while', 'although', 'though', 'when', 'where', 'as', 'than',
        'whether', 'nor', 'yet', 'since', 'unless'}
AUX = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'has', 'have', 'had', 'do', 'does', 'did',
       'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'not', "n't"}
CLASSES = ['determiner', 'preposition', 'pronoun', 'conjunction', 'aux/be/neg', 'number', 'punctuation',
           'Capitalized', 'subword-piece', 'content word']
PAIRS = [('determiner', 'number'), ('number', 'determiner'), ('punctuation', 'pronoun'),
         ('pronoun', 'punctuation'), ('preposition', 'aux/be/neg'), ('aux/be/neg', 'preposition')]
SWAP = {'mode': None, 'U': None, 'donor_pool': None, 'mask_cls': None, 'g': None}


def label_token(tid):
    raw = enc.decode([tid]); s = raw.strip()
    if s == '': return None
    low = s.lower()
    if re.fullmatch(r"[0-9][0-9,\.]*", s): return 'number'
    if re.fullmatch(r"[^\w\s]+", s): return 'punctuation'
    if low in DET: return 'determiner'
    if low in PREP: return 'preposition'
    if low in PRON: return 'pronoun'
    if low in CONJ: return 'conjunction'
    if low in AUX: return 'aux/be/neg'
    if s[0].isupper(): return 'Capitalized'
    if not raw.startswith(' ') and s.isalpha(): return 'subword-piece'
    if s.isalpha(): return 'content word'
    return None


def fwd(idx):
    CUR['idx'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


CUR = {}


def swap_hook(mo, i_, o_):
    if SWAP['mode'] is None: return None
    idx = CUR['idx']; U = SWAP['U'].to(o_.dtype)          # D x k
    cls = SWAP['cls_of'][idx]                              # B,T class ids
    msk = cls == SWAP['mask_cls']                          # positions to modify (class A)
    if not msk.any(): return None
    y = o_.clone()
    coords = (o_ - SWAP['mu'].to(o_.dtype)) @ U            # B,T,k
    if SWAP['mode'] == 'donor':
        pool = SWAP['donor_pool']                          # N,k donor coords (class B)
        sel = torch.randint(0, pool.shape[0], (int(msk.sum()),), generator=SWAP['g'], device=o_.device)
        new = pool[sel].to(o_.dtype)
    elif SWAP['mode'] == 'same':
        pool = SWAP['same_pool']
        sel = torch.randint(0, pool.shape[0], (int(msk.sum()),), generator=SWAP['g'], device=o_.device)
        new = pool[sel].to(o_.dtype)
    else:  # random-subspace: replace coords in a RANDOM matched-rank subspace with donor-pool-style values
        return None
    y[msk] = o_[msk] + (new - coords[msk]) @ U.T
    return y


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    cls_of = torch.full((V,), -1, dtype=torch.long)
    for t in range(min(V, 50257)):
        try: lab = label_token(t)
        except Exception: continue
        if lab is not None: cls_of[t] = CLASSES.index(lab)
    cls_of = cls_of.to(DEV); SWAP['cls_of'] = cls_of

    # class package subspace + per-class donor coord pools (from a capture pass)
    cap = []
    hook = H[0].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)) or None)
    ids = []
    for i in range(0, 96, 8):
        x = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); ids.append(x.reshape(-1)); fwd(x)
    tokc = torch.cat(ids, 0); X = torch.cat(cap, 0); cap.clear()
    xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
    xb.index_add_(0, tokc, X); cn.index_add_(0, tokc, torch.ones_like(tokc, dtype=torch.float))
    mu = X.mean(0)
    keep = ((cn >= MIN_OCC) & (cls_of >= 0)).nonzero().squeeze(1)
    tm_c = xb[keep]/cn[keep].unsqueeze(1); tm_c = tm_c - tm_c.mean(0)
    _, _, Vt = torch.linalg.svd(tm_c, full_matrices=False)
    U = Vt[:KPKG].T.contiguous()
    coords_all = (X - mu) @ U; cls_pos = cls_of[tokc]
    pools = {c: coords_all[cls_pos == CLASSES.index(c)] for c in set(a for p in PAIRS for a in p)}
    del X
    hook.remove()
    SWAP['U'] = U; SWAP['mu'] = mu; SWAP['g'] = torch.Generator(device=DEV).manual_seed(0)
    g2 = torch.Generator(device=DEV).manual_seed(1)
    Ur = torch.linalg.qr(torch.randn(D, KPKG, generator=g2, device=DEV))[0]

    # class successor profiles (bigram over the data): P(next-class | current-class), from labels
    nxt = torch.zeros(len(CLASSES), len(CLASSES), device=DEV)
    for i in range(0, NSEQ, 8):
        bb = blocks[i:i+8].to(DEV); a = cls_of[bb[:, :-1].reshape(-1)]; b = cls_of[bb[:, 1:].reshape(-1)]
        ok = (a >= 0) & (b >= 0)
        nxt.index_put_((a[ok], b[ok]), torch.ones(int(ok.sum()), device=DEV), accumulate=True)
    prof = nxt / nxt.sum(1, keepdim=True).clamp_min(1)

    hk = H[0].mlp.register_forward_hook(swap_hook)

    @torch.no_grad()
    def nextclass_dist(mode, A, B):
        """mean predicted next-CLASS distribution at class-A positions under the given swap mode"""
        SWAP['mode'] = mode; SWAP['mask_cls'] = CLASSES.index(A)
        SWAP['donor_pool'] = pools[B] if B else None; SWAP['same_pool'] = pools[A]
        acc = torch.zeros(len(CLASSES), device=DEV); n = 0
        for i in range(0, NSEQ, 8):
            bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous()
            p = F.softmax(fwd(idx).float(), -1)
            msk = cls_of[idx] == CLASSES.index(A)
            if not msk.any(): continue
            pm = p[msk]                                    # n, V
            for kk in range(len(CLASSES)):
                acc[kk] += pm[:, cls_of == kk].sum()
            n += int(msk.sum())
        SWAP['mode'] = None
        d = acc / acc.sum()
        return d

    def kl(p, q): return float((p * (p.clamp_min(1e-8)/q.clamp_min(1e-8)).log()).sum())

    res = {}; wins = 0; shifts = []; rand_shifts = []
    for A, B in PAIRS:
        base_d = nextclass_dist(None, A, None)
        don_d = nextclass_dist('donor', A, B)
        same_d = nextclass_dist('same', A, B)
        pa, pb = prof[CLASSES.index(A)], prof[CLASSES.index(B)]
        # does the swap move the distribution toward B's successor profile?
        d_to_B_base = kl(pb, base_d); d_to_B_don = kl(pb, don_d)
        d_to_A_base = kl(pa, base_d); d_to_A_don = kl(pa, don_d)
        toward_B = d_to_B_base - d_to_B_don                # positive = moved toward donor profile
        away_A = d_to_A_don - d_to_A_base                  # positive = moved away from own profile
        same_move = kl(pb, same_d) - d_to_B_base           # should be ~0
        win = bool(toward_B > 0 and away_A > -0.005)
        wins += int(win); shifts.append(toward_B)
        res[f'{A}->{B}'] = {'toward_donor_profile': round(toward_B, 4), 'away_from_own': round(away_A, 4),
                            'same_class_control': round(-same_move, 4), 'win': win}
        print(f"{A:>13} <- {B:<13}: toward-donor {toward_B:+.4f} | away-own {away_A:+.4f} | same-ctrl {-same_move:+.4f} | win {win}", flush=True)
    # random-subspace null on one pair (det<-number): swap coords in a random 24-dim subspace
    SWAP['U'] = Ur
    coordsR = None  # donor pool in random subspace: reuse capture? approximate: use gaussian with matched std
    # build random-subspace donor pool from a fresh capture of class-B outputs
    capR = []
    hookR = H[0].mlp.register_forward_hook(lambda mo, i_, o_: capR.append(o_.detach().float().reshape(-1, D)) or None)
    SWAP['mode'] = None
    for i in range(0, 32, 8): fwd(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    hookR.remove()
    XR = torch.cat(capR, 0); capR.clear()
    cpR = (XR - mu) @ Ur; clsR = cls_of[torch.cat([blocks[i:i+8].to(DEV)[:, :-1].reshape(-1) for i in range(0, 32, 8)], 0)]
    pools_r = {'number': cpR[clsR == CLASSES.index('number')]}
    del XR
    pools_backup = pools['number']; pools['number'] = pools_r['number']
    base_d = nextclass_dist(None, 'determiner', None)
    don_d = nextclass_dist('donor', 'determiner', 'number')
    pb = prof[CLASSES.index('number')]
    rand_toward = kl(pb, base_d) - kl(pb, don_d)
    pools['number'] = pools_backup; SWAP['U'] = U
    hk.remove()

    out = {'pairs': res, 'wins': wins, 'random_subspace_toward_donor_detnum': round(rand_toward, 4),
           'mean_toward_shift': round(sum(shifts)/len(shifts), 4)}
    out['pred_a_per_class_causal'] = bool(wins >= 5 and out['mean_toward_shift'] >= 3*abs(rand_toward))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"wins {wins}/6 | mean toward-shift {out['mean_toward_shift']} | random-subspace null {rand_toward:+.4f}", flush=True)
    print(f"pred_a per-class-causal {out['pred_a_per_class_causal']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
