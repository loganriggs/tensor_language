#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_classes_cover pred_b_03_continuation_queries_enriched pred_c_07_punctuation_keys_not_enriched pred_d_11_of_queries_not_enriched pred_e_15_determiner_queries_not_enriched
"""Embedding-forward folding, rung 50 (v662): where the program heads' pattern mass actually falls — a class census on real rows.

v660/v661: readings taken from the largest entries of the gate tables were mostly false under restricted edits. The honest replacement:
for each of the twelve program heads of blocks 0-1 (0.3 0.4 0.6 0.7 0.8 | 1.0 1.1 1.3 1.5 1.6 1.7 1.8), on 64 fit rows (queries >= 8), the share of
its real off-diagonal |pattern| mass by QUERY class and by KEY class, divided by the class's share of positions (enrichment). Classes by GPT-2
token string: word-initial alphabetic (leading space, lowercase), capitalised word-initial, continuation (no leading space, alphabetic),
punctuation, digit, other. Response statistics; no edits. The four falsified readings and the one supported reading predict the enrichments.
PREDICTIONS (scored as written; failures preserved)
    pred_a_classes_cover                    the six classes cover >= 0.95 of positions (instrument)
    pred_b_03_continuation_queries_enriched 0.3's query-mass enrichment on continuation tokens >= 2 (consistent with v660's 2.6x cost). Prior: likely
    pred_c_07_punctuation_keys_not_enriched 0.7's key-mass enrichment on punctuation <= 1.5 (consistent with v660). Prior: likely
    pred_d_11_of_queries_not_enriched       1.1's query-mass enrichment on of / possessive tokens (a seventh class for this test) <= 1.5 (consistent with v661). Prior: likely
    pred_e_15_determiner_queries_not_enriched 1.5's query-mass enrichment on determiners (an eighth class) <= 1.5 (consistent with v661). Prior: likely
PRICE (registered maximum): 2 capture forwards (64 rows); 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_mass_census_v662_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
CANDIDATE_ID = "embedding_forward.mass_census_v662"
FORWARDS_MAX = 4
EBATCH, N_CAP, Q_MIN = 32, 64, 8
HEADS = {0: (3, 4, 6, 7, 8), 1: (0, 1, 3, 5, 6, 7, 8)}
COVER, ENRICH_MIN, ENRICH_MAX = 0.95, 2.0, 1.5
PREDICTIONS = {"pred_a_classes_cover": ">= 0.95", "pred_b_03_continuation_queries_enriched": ">= 2", "pred_c_07_punctuation_keys_not_enriched": "<= 1.5",
               "pred_d_11_of_queries_not_enriched": "<= 1.5", "pred_e_15_determiner_queries_not_enriched": "<= 1.5"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"cover": COVER, "enrich_min": ENRICH_MIN, "enrich_max": ENRICH_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    V = model.config.vocab_size; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long()
    OF = {" of", " his", " her", " their", " your", " its", " our", " my"}; DET = {" a", " an", " the", " some", " these", " those", " this", " that", " several", " many", " few", " each", " every"}
    CLASSES = ("word_initial", "capitalised", "continuation", "punctuation", "digit", "other", "of_possessive", "determiner")
    cls = torch.full((V,), 5, dtype=torch.long); ofq = torch.zeros(V, dtype=torch.bool); det = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        try:
            s = enc.decode([i])
        except Exception:
            continue
        st = s.strip()
        if s.startswith(" ") and len(s) > 1 and s[1:].isalpha():
            cls[i] = 1 if s[1].isupper() else 0
        elif s and not s.startswith(" ") and s.isalpha():
            cls[i] = 2
        elif st and all(not c.isalnum() and not c.isspace() for c in st):
            cls[i] = 3
        elif st and st.isdigit():
            cls[i] = 4
        if s in OF:
            ofq[i] = True
        if s in DET:
            det[i] = True
    cls, ofq, det = cls.to(dev), ofq.to(dev), det.to(dev)
    with torch.no_grad():
        state = {"capture": {0: [], 1: []}}
        natives = {l: blocks[l].attn.squared_attention for l in (0, 1)}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                state["capture"][l].append(pat.detach().clone())
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in (0, 1):
            blocks[l].attn.squared_attention = make_patched(l)
        idxs = []
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1; idxs.append(idx)
        for l in (0, 1):
            blocks[l].attn.squared_attention = natives[l]
        idx = torch.cat(idxs); P = {l: torch.cat(state["capture"][l]) for l in (0, 1)}
        Tn = idx.shape[1]; pos = torch.arange(Tn, device=dev); off = (pos[:, None] > pos[None, :])[None].expand(idx.shape[0], -1, -1); qm = (pos >= Q_MIN)
        c_tok = cls[idx]                                                                                 # [B, T]
        masks_q = {name: ((c_tok == i) if i < 6 else (ofq[idx] if i == 6 else det[idx])) for i, name in enumerate(CLASSES)}
        pos_share = {name: float(m[:, qm].float().mean()) for name, m in masks_q.items()}
        key_share = {name: float((m[:, None, :].expand_as(off) & off)[:, qm].float().sum() / off[:, qm].float().sum()) for name, m in masks_q.items()}
        cover = sum(pos_share[c] for c in CLASSES[:6])
        report = {"position_share": pos_share, "key_share": key_share, "cover": cover, "heads": {}}
        for l, hs in HEADS.items():
            for h in hs:
                m = P[l][:, h].abs() * off; m = m[:, qm]; tot = float(m.sum())
                qmass = {c: float((m * masks_q[c][:, qm, None]).sum() / tot) for c in CLASSES}
                kmass = {c: float((m * masks_q[c][:, None, :]).sum() / tot) for c in CLASSES}
                report["heads"][f"{l}.{h}"] = {"query_enrichment": {c: qmass[c] / max(pos_share[c], 1e-9) for c in CLASSES}, "key_enrichment": {c: kmass[c] / max(key_share[c], 1e-9) for c in CLASSES},
                                               "query_mass": qmass, "key_mass": kmass}
                qe, ke = report["heads"][f"{l}.{h}"]["query_enrichment"], report["heads"][f"{l}.{h}"]["key_enrichment"]
                print(f"head {l}.{h}: query enrichment " + " ".join(f"{c[:5]}={qe[c]:.2f}" for c in CLASSES) + " | key enrichment " + " ".join(f"{c[:5]}={ke[c]:.2f}" for c in CLASSES))
        print("position shares:", {c: round(v, 3) for c, v in pos_share.items()}, "| cover of the six base classes:", round(cover, 3))
    H_ = report["heads"]
    predictions = {"pred_a_classes_cover": cover >= COVER, "pred_b_03_continuation_queries_enriched": H_["0.3"]["query_enrichment"]["continuation"] >= ENRICH_MIN,
                   "pred_c_07_punctuation_keys_not_enriched": H_["0.7"]["key_enrichment"]["punctuation"] <= ENRICH_MAX,
                   "pred_d_11_of_queries_not_enriched": H_["1.1"]["query_enrichment"]["of_possessive"] <= ENRICH_MAX,
                   "pred_e_15_determiner_queries_not_enriched": H_["1.5"]["query_enrichment"]["determiner"] <= ENRICH_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_mass_census_result_v662", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
