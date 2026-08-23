"""CLEAN capacity measurement, resolving §1023's confound. §1023 found per-topic retention GROWS with N (to 2.23 at
N=6) but flagged this as confounded: injecting more CONTENT words raises content predictions GENERALLY (§1020), and
topic-neighbors are content words riding that shift. Here isolate the TOPIC-SPECIFIC component by subtracting the
neighbor-boost of topic-UNRELATED CONTROL content words (held out, never injected). Clean topic-boost = injected-topic
neighbor-boost - control-content neighbor-boost. This removes the general content-density shift.

REGISTERED PREDICTIONS:
  (0) SANITY: at N=1 the control-subtracted topic boost > 0 (topic-specific effect survives the subtraction).
  (a) CLEAN CAPACITY IS ADDITIVE/FLAT (not super-linear): after subtracting the control-content shift, per-topic
      clean retention stays ~1 across N (topics add independently, no dilution and no genuine super-addition) up to
      N=6 -> the raw §1023 growth WAS the content-density confound; the true bag is high-capacity ADDITIVE;
  (b) report raw vs control-subtracted (clean) retention vs N, and the control content-density boost vs N."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_injection_capacity_clean_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; NNEIGH = 20
POSITIONS = [3, 12, 24, 36, 48, 60]
WORDS = [' football', ' ocean', ' hospital', ' music', ' science', ' church']  # injected topics
CONTROLS = [' kitchen', ' lawyer']  # held-out, topically-unrelated content words (never injected)
NS = [1, 2, 3, 4, 6]


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def neighbors(wid, k):
    W = m.lm_head.weight.float(); wv = W[wid] / (W[wid].norm() + 1e-9)
    sims = (W / (W.norm(dim=1, keepdim=True) + 1e-9)) @ wv; sims[wid] = -1e9
    return torch.topk(sims, k).indices


@torch.no_grad()
def boosts_for(blocks, inj_wids, inj_pos, neigh_list, exclude_wids):
    # returns mean Δlp for EACH neighbor-set in neigh_list (all measured on the same injected forward pass)
    sums = [0.0]*len(neigh_list); n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous(); inj_idx = base_idx.clone()
        for w, p in zip(inj_wids, inj_pos): inj_idx[:, p] = w
        lb = F.log_softmax(forward_logits(base_idx).float()[:, -1], -1)
        li = F.log_softmax(forward_logits(inj_idx).float()[:, -1], -1)
        has = torch.zeros(base_idx.shape[0], dtype=torch.bool, device=DEV)
        for w in exclude_wids: has |= (base_idx == w).any(1)
        keep = ~has
        for j, ng in enumerate(neigh_list): sums[j] += float((li[:, ng] - lb[:, ng]).mean(1)[keep].sum())
        n += int(keep.sum())
    return [s/max(n, 1) for s in sums]


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    wids = [tid(w) for w in WORDS]; neigh = [neighbors(w, NNEIGH) for w in wids]
    cwids = [tid(w) for w in CONTROLS]; cneigh = [neighbors(w, NNEIGH) for w in cwids]
    # solo clean boost per topic: inject only topic j; boost_j - mean control boost
    solo_clean = []
    for j in range(len(WORDS)):
        res = boosts_for(rows, [wids[j]], [POSITIONS[0]], [neigh[j]] + cneigh, [wids[j]] + cwids)
        solo_clean.append(res[0] - float(np.mean(res[1:])))
    print(f"solo clean boosts: {[round(s,3) for s in solo_clean]}", flush=True)
    out = {'solo_clean': [round(s, 4) for s in solo_clean], 'by_N': {}}
    for N in NS:
        inj_wids = wids[:N]; inj_pos = POSITIONS[:N]
        raw_rets = []; clean_rets = []; ctrl_boosts = []
        for j in range(N):
            res = boosts_for(rows, inj_wids, inj_pos, [neigh[j]] + cneigh, [wids[j]] + cwids)
            topic_boost = res[0]; ctrl = float(np.mean(res[1:]))
            raw_rets.append(topic_boost / max(solo_clean[j] + 0, 1e-6) if False else topic_boost)  # raw boost
            clean_rets.append((topic_boost - ctrl) / max(solo_clean[j], 1e-6))
            ctrl_boosts.append(ctrl)
        out['by_N'][str(N)] = {'raw_boost': round(float(np.mean(raw_rets)), 3),
                               'clean_retention': round(float(np.mean(clean_rets)), 3),
                               'control_density_boost': round(float(np.mean(ctrl_boosts)), 3)}
        print(f"N={N}: raw-boost {out['by_N'][str(N)]['raw_boost']} | clean-retention {out['by_N'][str(N)]['clean_retention']} | control-density {out['by_N'][str(N)]['control_density_boost']}", flush=True)
    c2 = out['by_N']['2']['clean_retention']; c6 = out['by_N']['6']['clean_retention']
    out['clean_retention_2'] = c2; out['clean_retention_6'] = c6
    out['pred_0_topic_survives'] = bool(solo_clean and np.mean(solo_clean) > 0.05)
    out['pred_a_clean_additive_flat'] = bool(0.6 < c6 < 1.4 and abs(c6 - c2) < 0.4)  # flat-ish near 1, no super-linear growth
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"clean retention N=2 {c2} vs N=6 {c6} | control-density grows: {[out['by_N'][str(N)]['control_density_boost'] for N in NS]}", flush=True)
    print(f"pred_0 topic-survives {out['pred_0_topic_survives']} | pred_a clean-additive-flat {out['pred_a_clean_additive_flat']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
