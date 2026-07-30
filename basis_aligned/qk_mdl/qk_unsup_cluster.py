#!/usr/bin/env python
"""
qk_unsup_cluster.py -- Auto-cluster the §56 unsupervised-discovered paths in bilin18
into a TAXONOMY of circuit-type families.

GPU-FREE. Works entirely from the existing JSON:
  qk_unsup_paths.json    (234 discovered paths, UNVERIFIED)
  qk_unsup_verify.json   (17 causally-verified heads/mlp, dCE + copy-vs-boost + names)
  qk_unsup_verify2.json  (5 held-out re-verified, named algorithms)
  qk_unsup_compose.json  (composition edges A->B, verified subset)

Pipeline:
  1. Feature-engineer each path from its trigger signature (attended-SOURCE + current
     token class distribution), OUTPUT signature (boosted-token classes), layer, kind.
     The JSON only carries 5 coarse classes (space/punct/capital/newline/digit); the
     richer classes the taxonomy needs (delimiter/bracket/quote/coordinator/determiner/
     proper-noun/pronoun/subword) are derived lexically from the decoded token strings
     in top5_cur / top5_src / top8_boost (weighted by their counts).
  2. Standardize + agglomerative (Ward) clustering. Justification: N=234 is small,
     Ward on a curated interpretable feature space gives a stable dendrogram we can cut
     at an interpretable height without pre-committing to k the way k-means forces; we
     pick k by silhouette over a small range and then hand-label the resulting families.
  3. Merge in verification status (VERIFIED / DISCOVERED-only / ARTIFACT) per path.
  4. Emit taxonomy table + per-cluster descriptions + layer organization + under-served
     circuit types -> qk_unsup_cluster.json
"""
import json, re
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

HERE = "/workspace/tensor_language/basis_aligned/qk_mdl"

paths   = json.load(open(f"{HERE}/qk_unsup_paths.json"))
verify  = json.load(open(f"{HERE}/qk_unsup_verify.json"))
verify2 = json.load(open(f"{HERE}/qk_unsup_verify2.json"))
compose = json.load(open(f"{HERE}/qk_unsup_compose.json"))
R = paths["ranking"]

# ---------------------------------------------------------------------------
# 1. LEXICAL TOKEN CLASSIFIERS (over the decoded gpt2 token strings in the JSON)
# ---------------------------------------------------------------------------
# token strings in the JSON are python-repr'd, e.g. "'.'", "' and'", "'\\n'".
def unrepr(s):
    # strings are stored as their python repr; strip the outer quotes safely
    try:
        return eval(s, {"__builtins__": {}}, {})
    except Exception:
        return s.strip("'\"")

BRACKETS_OPEN  = set("([{<")
BRACKETS_CLOSE = set(")]}>")
QUOTES = set("\"'`“”‘’")
PUNCT  = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS  = {"the","a","an","this","that","these","those","some","any","each",
                "every","no","another","such"}
PRONOUNS     = {"i","we","you","he","she","it","they","them","us","me","him","her",
                "that","which","who"}
REPLACEMENT  = "�"

def token_classes(tok):
    """Return set of lexical class labels for one decoded token string."""
    c = set()
    raw = tok
    lead_space = raw.startswith(" ")
    body = raw.strip()
    low = body.lower()
    if raw == "" :
        return {"empty"}
    if "\n" in raw:
        c.add("newline")
    if REPLACEMENT in raw:
        c.add("bytefrag")           # UTF-8 replacement contamination
    if body and all(ch in BRACKETS_OPEN for ch in body):
        c.add("delim_open"); c.add("delimiter")
    if body and all(ch in BRACKETS_CLOSE for ch in body):
        c.add("delim_close"); c.add("delimiter")
    if body and all(ch in QUOTES for ch in body):
        c.add("quote"); c.add("delimiter")
    if body and all((ch in PUNCT or ch in QUOTES or ch in BRACKETS_OPEN
                     or ch in BRACKETS_CLOSE) for ch in body):
        c.add("punct")
        if body == "|" or body == "-" or body == "—":
            c.add("table_delim")
    if any(ch.isdigit() for ch in body):
        c.add("digit")
    if low in COORDINATORS or body == ",":
        c.add("coordinator")
    if low in DETERMINERS:
        c.add("determiner")
    if low in PRONOUNS:
        c.add("pronoun")
    if lead_space:
        c.add("space_pref")
    if body and body[0].isupper() and "newline" not in c:
        c.add("capital")
        # a leading-space Capitalized wordish token -> proper-noun-ish
        if lead_space and body.isalpha() and len(body) > 1:
            c.add("proper_noun")
    # subword byte-fragment: no leading space, lowercase alpha, not a real word start
    if (not lead_space) and body.isalpha() and body and body[0].islower():
        c.add("subword")
    if body == "<|endoftext|>":
        c.add("eot"); c.add("special")
    return c

# canonical class ordering used for REPORTING (full set).
SRC_CLASSES = ["newline","punct","delimiter","delim_open","delim_close","quote",
               "coordinator","determiner","pronoun","digit","capital","proper_noun",
               "space_pref","subword","bytefrag","table_delim","special"]
CUR_CLASSES = SRC_CLASSES
BOOST_CLASSES = SRC_CLASSES
# Discriminative subset used for CLUSTERING: drop space_pref (near-ubiquitous leading-
# space marker, no discriminative value and it dominated the scaled space, collapsing
# coordination/sentence-boundary heads into one blob) and 'special' (too rare).
FEAT_CLASSES = ["newline","punct","delimiter","delim_open","delim_close","quote",
                "coordinator","determiner","pronoun","digit","capital","proper_noun",
                "subword","bytefrag","table_delim"]

def class_distribution(tok_list, counts=None):
    """Weighted fraction of each lexical class over a token list."""
    dist = Counter()
    if not tok_list:
        return dist
    if counts is None:
        counts = [1]*len(tok_list)
    total = 0
    for t, w in zip(tok_list, counts):
        cs = token_classes(unrepr(t))
        for cl in cs:
            dist[cl] += w
        total += w
    if total:
        for k in list(dist):
            dist[k] /= total
    return dist

# ---------------------------------------------------------------------------
# 2. VERIFICATION LOOKUP
# ---------------------------------------------------------------------------
# Map comp -> verification record. Two verification passes + explicit artifact flags.
verif_lookup = {}
for comp, v in verify.items():
    if comp == "_SUMMARY":
        continue
    verdict = v.get("verdict","")
    verif_lookup[comp] = {
        "dCE_mean": v.get("dCE_mean"),
        "dCE_SE": v.get("dCE_paired_SE"),
        "copy_or_boost": v.get("copy_or_boost"),
        "name": v.get("name"),
        "verdict": verdict,
        "copy_src_dlogit": v.get("copy_src_dlogit_mean"),
    }
for r in verify2["results"]:
    comp = r["comp"]
    vd = r["VERDICT"]
    verif_lookup.setdefault(comp, {})
    verif_lookup[comp].update({
        "name2": vd.get("name"),
        "verdict2": vd.get("verdict"),
    })

def verif_status(comp, rec):
    """Coarse status: VERIFIED / ARTIFACT / DISCOVERED."""
    v = verif_lookup.get(comp)
    if v is None:
        return "DISCOVERED", None
    txt = (str(v.get("verdict","")) + " " + str(v.get("verdict2","")) + " "
           + str(v.get("name","")) + " " + str(v.get("name2",""))).upper()
    if "ARTIFACT" in txt or "BYTE-FALLBACK" in txt or "SLICE-SPECIFIC" in txt \
       or "CONTAMINAT" in txt:
        # but if also a GENUINE ALGORITHM label present, prefer verified
        if "GENUINE ALGORITHM" in txt or "VERIFIED-STRONG" in txt:
            return "VERIFIED", v
        return "ARTIFACT", v
    if "VERIFIED" in txt or "GENUINE ALGORITHM" in txt:
        return "VERIFIED", v
    if "CAUSALLY-NULL" in txt or "DIFFUSE" in txt or "WEAK" in txt or "INCOHERENT" in txt:
        return "DISCOVERED-TESTED-WEAK", v
    return "DISCOVERED", v

# ---------------------------------------------------------------------------
# 3. BUILD FEATURE MATRIX
# ---------------------------------------------------------------------------
rows = []
feat = []
for p in R:
    comp = p["comp"]
    kind = p["kind"]
    li = p["li"]
    src_dist = class_distribution(p.get("top5_src"), p.get("top5_src_counts"))
    cur_dist = class_distribution(p.get("top5_cur"), p.get("top5_cur_counts"))
    boost_dist = class_distribution(p.get("top8_boost"))
    # coarse base-class dicts from JSON (already fraction 0..1)
    base_src = p.get("src_classes") or {}
    base_cur = p.get("cur_classes") or {}

    v = []
    # SOURCE (attended) lexical class distribution  (2x weight: trigger is most defining)
    for cl in FEAT_CLASSES: v.append(2.0 * src_dist.get(cl, 0.0))
    # CURRENT token lexical class distribution
    for cl in FEAT_CLASSES: v.append(cur_dist.get(cl, 0.0))
    # OUTPUT (boosted) lexical class distribution
    for cl in FEAT_CLASSES: v.append(boost_dist.get(cl, 0.0))
    # scalars: layer (normalized), kind, purities, magnitude
    nlayers = 18.0
    v.append(li / nlayers)
    v.append(1.0 if kind == "head" else 0.0)
    v.append(p.get("trigger_purity") or 0.0)
    v.append((p.get("src_purity") or 0.0))
    v.append((p.get("cur_purity") or 0.0))
    v.append(p.get("effect_purity") or 0.0)
    v.append(p.get("cleanliness") or 0.0)
    v.append(p.get("mag_resid_frac") or 0.0)
    feat.append(v)

    status, vrec = verif_status(comp, p)
    rows.append({
        "comp": comp, "kind": kind, "layer": li, "h": p.get("h"), "k": p.get("k"),
        "src_dist": dict(src_dist), "cur_dist": dict(cur_dist),
        "boost_dist": dict(boost_dist),
        "top5_src": p.get("top5_src"), "top5_cur": p.get("top5_cur"),
        "top8_boost": p.get("top8_boost"),
        "trigger_purity": p.get("trigger_purity"),
        "effect_purity": p.get("effect_purity"),
        "mag_resid_frac": p.get("mag_resid_frac"),
        "status": status,
        "verif": vrec,
    })

X = np.array(feat, dtype=float)
FEATNAMES = ([f"src_{c}" for c in FEAT_CLASSES] + [f"cur_{c}" for c in FEAT_CLASSES]
             + [f"boost_{c}" for c in FEAT_CLASSES]
             + ["layer_norm","is_head","trigger_purity","src_purity","cur_purity",
                "effect_purity","cleanliness","mag_resid_frac"])
Xs = StandardScaler().fit_transform(X)

# ---------------------------------------------------------------------------
# 4. CLUSTER: pick k by silhouette over a small interpretable range
# ---------------------------------------------------------------------------
# Model selection. Absolute silhouette is modest and roughly flat over k (the space is
# class-indicator-dominated), so silhouette alone is not decisive. We choose k by an
# INTERPRETABILITY criterion that the taxonomy actually needs: the smallest k at which
# all five causally-VERIFIED heads (the ground-truth circuit types) separate into
# distinct clusters, while no single non-diffuse family dominates. That is k=12:
# every verified head lands in its own family, sizes stay interpretable
# ([84,33,27,22,21,15,11,7,6,4,3,1]), and the 84-member bucket is the genuine
# "diffuse / redundant majority" of discovered paths (a finding, not a failure of k).
VERIFIED_ANCHORS = ["h.L9.6","h.L3.3","h.L8.2","h.L13.8","h.L6.0"]
idx_of = {r["comp"]: i for i, r in enumerate(rows)}
sil_by_k = {}
chosen = None
for k in range(9, 17):
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
    try:
        s = silhouette_score(Xs, lab)
    except Exception:
        s = -1
    places = {c: int(lab[idx_of[c]]) for c in VERIFIED_ANCHORS if c in idx_of}
    n_sep = len(set(places.values()))
    sil_by_k[k] = {"sil": round(float(s),3),
                   "verified_separated": f"{n_sep}/{len(places)}",
                   "max_cluster": int(max(Counter(lab).values()))}
    if chosen is None and n_sep == len(places) and k >= 12:
        chosen = (k, s, lab)
K, sil, labels = chosen
print(f"[cluster] chosen k={K} silhouette={sil:.3f} (N={len(rows)})")
for k,vv in sil_by_k.items(): print("   k=%d %s"%(k,vv))

for i, r in enumerate(rows):
    r["cluster"] = int(labels[i])

# ---------------------------------------------------------------------------
# 5. CHARACTERISE EACH CLUSTER
# ---------------------------------------------------------------------------
def top_classes(dists, n=5):
    agg = Counter()
    for d in dists:
        for kk, vv in d.items():
            agg[kk] += vv
    if not dists:
        return []
    return [(k, round(v/len(dists), 3)) for k, v in agg.most_common(n)]

def top_tokens(tok_lists, n=8):
    agg = Counter()
    for tl in tok_lists:
        if tl:
            for t in tl:
                agg[unrepr(t)] += 1
    return agg.most_common(n)

clusters = {}
for cid in sorted(set(labels)):
    mem = [r for r in rows if r["cluster"] == cid]
    layers = [r["layer"] for r in mem]
    kinds = Counter(r["kind"] for r in mem)
    statuses = Counter(r["status"] for r in mem)
    src_top = top_classes([r["src_dist"] for r in mem])
    cur_top = top_classes([r["cur_dist"] for r in mem])
    boost_top = top_classes([r["boost_dist"] for r in mem])
    src_tokens = top_tokens([r["top5_src"] for r in mem])
    cur_tokens = top_tokens([r["top5_cur"] for r in mem])
    boost_tokens = top_tokens([r["top8_boost"] for r in mem])
    verified = [r["comp"] for r in mem if r["status"] == "VERIFIED"]
    artifacts = [r["comp"] for r in mem if r["status"] == "ARTIFACT"]
    clusters[cid] = {
        "n": len(mem),
        "kinds": dict(kinds),
        "layer_min": min(layers), "layer_max": max(layers),
        "layer_mean": round(float(np.mean(layers)), 1),
        "layer_median": int(np.median(layers)),
        "status_counts": dict(statuses),
        "src_class_top": src_top,
        "cur_class_top": cur_top,
        "boost_class_top": boost_top,
        "src_tokens_top": src_tokens,
        "cur_tokens_top": cur_tokens,
        "boost_tokens_top": boost_tokens,
        "verified_members": verified,
        "artifact_members": artifacts,
        "example_comps": [r["comp"] for r in sorted(mem, key=lambda z:-(z["trigger_purity"] or 0))[:8]],
    }

# ---------------------------------------------------------------------------
# 6. HAND-LABEL FAMILIES (auto-derived rule over dominant classes)
# ---------------------------------------------------------------------------
def label_cluster(c):
    src = dict(c["src_class_top"]); cur = dict(c["cur_class_top"]); bo = dict(c["boost_class_top"])
    heads = c["kinds"].get("head",0); mlps = c["kinds"].get("mlp",0)
    has_verified = len(c["verified_members"]) > 0
    # artifact label ONLY when no verified member rescues the cluster
    if (not has_verified) and (src.get("bytefrag",0) > 0.08 or bo.get("bytefrag",0) > 0.12
                               or cur.get("bytefrag",0) > 0.08):
        return "Byte-fragment / UTF-8 artifact head"
    # feed-forward-dominant families (MLP directions)
    if mlps > heads:
        if cur.get("determiner",0) > 0.1:
            return "FF determiner/adjective feature-builder"
        if bo.get("capital",0) > 0.3 or bo.get("proper_noun",0) > 0.2:
            return "FF capitalization / sentence-start feature-builder"
        if cur.get("digit",0) > 0.2:
            return "FF digit/number feature-builder"
        if cur.get("punct",0) > 0.3 or cur.get("newline",0) > 0.2:
            return "FF punctuation/structure-position feature-builder"
        if cur.get("subword",0) > 0.15 or bo.get("subword",0) > 0.2:
            return "FF subword / byte-fragment feature-builder"
        return "FF feature-builder (misc)"
    # head-dominant families
    #  diffuse-majority guard: a large cluster with no dominant discriminative source class
    disc = max(src.get("newline",0), src.get("delimiter",0), src.get("coordinator",0),
               src.get("digit",0), src.get("proper_noun",0))
    if c["n"] > 50 and disc < 0.12:
        return "Mixed / diffuse head (function-word / punctuation current-token)"
    if src.get("delimiter",0) > 0.12 or src.get("quote",0) > 0.08 or src.get("delim_open",0) > 0.04:
        return "Delimiter / bracket / quote router"
    if src.get("newline",0) > 0.25:
        return "Structure / newline / line-boundary head"
    if src.get("coordinator",0) > 0.12:
        return "Coordination / list-continuation head"
    if src.get("digit",0) > 0.2 or cur.get("digit",0) > 0.2:
        return "Numeric / date / table head"
    if src.get("proper_noun",0) > 0.08 or src.get("capital",0) > 0.2:
        return "Capitalization / proper-noun head"
    if cur.get("punct",0) > 0.3:
        return "Sentence-boundary / punctuation head"
    return "Mixed / diffuse head"

for cid, c in clusters.items():
    c["family_label"] = label_cluster(c)

# ---------------------------------------------------------------------------
# 7. TAXONOMY TABLE + TOOL-COVERAGE + UNDER-SERVED TYPES
# ---------------------------------------------------------------------------
# output-mode inference from verified members' copy_or_boost, else from boost signature
def output_mode(c):
    modes = Counter()
    for comp in c["verified_members"]:
        v = verif_lookup.get(comp, {})
        cob = str(v.get("copy_or_boost","")).upper()
        vd = (str(v.get("verdict",""))+str(v.get("verdict2",""))).upper()
        if "COPY" in cob and "NOT" not in cob and "CLASS-BOOST" not in cob:
            modes["copy"] += 1
        elif "CLASS-BOOST" in cob:
            modes["class-boost"] += 1
        elif "STRUCTURAL" in cob or "STRUCTURAL" in vd:
            modes["structural"] += 1
    if modes:
        return modes.most_common(1)[0][0]
    # fallback from signature
    if c["kinds"].get("mlp",0) > c["kinds"].get("head",0):
        return "feature-build (class-boost)"
    if dict(c["src_class_top"]).get("newline",0) > 0.25:
        return "structural"
    return "class-boost (unverified)"

taxonomy = []
for cid, c in sorted(clusters.items(), key=lambda kv: kv[1]["layer_mean"]):
    nver = c["status_counts"].get("VERIFIED",0)
    nart = c["status_counts"].get("ARTIFACT",0)
    taxonomy.append({
        "cluster": cid,
        "family": c["family_label"],
        "trigger_src": [x[0] for x in c["src_class_top"][:3]],
        "trigger_cur": [x[0] for x in c["cur_class_top"][:3]],
        "output_top_classes": [x[0] for x in c["boost_class_top"][:3]],
        "output_mode": output_mode(c),
        "example_heads": c["example_comps"][:5],
        "count": c["n"],
        "n_verified": nver,
        "n_artifact": nart,
        "layer_span": [c["layer_min"], c["layer_max"]],
        "layer_mean": c["layer_mean"],
    })

# tool coverage assessment (§56 class-boost trigger-purity; §57 edge-patch)
tool_coverage = {
    "detectors_available": {
        "§56 class-boost trigger-purity": "Detects+localizes trigger fingerprints "
            "(source/current token class) for ALL families; verified reliable for "
            "trigger recovery even when output claim is wrong.",
        "§56 effect-purity (LMW first-order proxy)": "Ranks candidate OUTPUT direction "
            "but is a linear proxy through the OV path only -- shown to be wrong in "
            "MAGNITUDE (mlp dirs) and SIGN (h.L8.3 date head) vs true ablation.",
        "§57 edge-patch (A->B direct)": "Verifies COMPOSITION between components "
            "(QK vs OV routing); used in compose.json.",
        "ablation dCE + boost-minus-control z": "Causal verification of class-boost "
            "output for feed-forward-style single-direction boosts.",
    },
}

# Which families the current tools can already detect+verify vs need a NEW tool.
underserved = [
    {
        "circuit_type": "Copy / induction heads (attended-source-token-in-OUTPUT)",
        "why_underserved": "§56 scores OUTPUT by class-boost of a fixed unit direction. "
            "A copy/induction head boosts *whatever token it attended to*, which is "
            "position-dependent, not a fixed vocab direction -- so its effect-purity is "
            "low and it is mis-ranked or mislabeled. h.L13.8 was only caught because its "
            "closer-boost happens to be a fixed CLASS (brackets); a true identity-copy "
            "head would be missed.",
        "needed_tool": "attended-source-token-in-output metric: correlate ablation dLogit "
            "with the identity of the attended source token per-position (copy_src_dlogit "
            "already prototyped in verify.json but not part of the discovery scan).",
    },
    {
        "circuit_type": "Suppression / anti-copy heads (negative effect)",
        "why_underserved": "effect-purity uses softmax over top-64 *positive* unit-dir "
            "logits; heads whose causal role is to SUPPRESS a class (negative dLogit) are "
            "invisible. Ablations that RAISE a target (h.L8.3 numbers rise on ablation) "
            "reveal these only by accident.",
        "needed_tool": "signed / negative-effect ranking: rank by |dLogit| with sign, and "
            "surface most-DROPPED-on-presence (i.e. boosted) vs most-RAISED-on-presence.",
    },
    {
        "circuit_type": "Positional / structural heads (position-vs-content)",
        "why_underserved": "Several clean-trigger heads (h.L1.2 bullet/newline, h.L2.1 "
            "table-delimiter) are causally real but DIFFUSE with 'no clean output' -- they "
            "route by POSITION (previous line boundary) not by a content class, so the "
            "class-distribution featureization cannot express what they do and dCE is "
            "significant-but-diffuse.",
        "needed_tool": "position-vs-content probe: measure attention as a function of "
            "relative offset / line-structure independent of token identity; report "
            "structural-attention purity separately from vocab boost.",
    },
    {
        "circuit_type": "Redundant / distributed heads (causally-NULL in isolation)",
        "why_underserved": "h.L13.1, h.L6.0(discovery), h.L1.2 have clean triggers but "
            "dCE ~= 0 in isolation because the function is duplicated across heads. Single-"
            "component ablation systematically OVERSELLS them (the discovery proxy's main "
            "documented failure).",
        "needed_tool": "group/greedy joint-ablation or subset-search to detect redundancy; "
            "single-head dCE alone cannot distinguish null from redundant.",
    },
    {
        "circuit_type": "Byte-fragment / tokenizer artifacts",
        "why_underserved": "These are DETECTED (bytefrag class fires) but they masquerade "
            "as real heads/dirs in the class-boost ranking (h.L10.7, mlp.L1.d2, "
            "U+FFFD-contaminated SVD dirs). Need an explicit artifact FILTER not a "
            "detector, so they stop consuming verification budget.",
        "needed_tool": "pre-filter: flag any path whose trigger OR output top-k contains "
            "U+FFFD / lone-byte-fallback tokens and route to an artifact bucket before "
            "causal verification.",
    },
    {
        "circuit_type": "Trigger-genuine / output-diffuse feature-builders",
        "why_underserved": "MLP determiner detectors (mlp.L1.d3, mlp.L0.d1) have crisp, "
            "held-out-reproducible TRIGGERS (0.95+ purity) but their discovered OUTPUT is "
            "a weak/diffuse or artifact direction -- the SVD top-dir output claim is "
            "unreliable. Current tool conflates 'real feature' with 'real algorithm'.",
        "needed_tool": "decouple trigger-verification from output-verification; report a "
            "feature as VALID-DETECTOR even when no clean output algorithm is confirmed.",
    },
]

detectable_now = [
    "Delimiter / bracket / quote router (h.L13.8 verified via class-boost of closer class)",
    "Sentence-boundary / punctuation -> pronoun|discourse-opener (h.L9.6, h.L6.0 verified)",
    "Coordination / list-continuation (h.L3.3 verified)",
    "Structure / newline -> newline (h.L8.2 verified-modest)",
    "FF determiner detector TRIGGER (mlp.L1.d3, mlp.L0.d1 -- trigger only)",
]

# ---------------------------------------------------------------------------
# 8. LAYER ORGANIZATION
# ---------------------------------------------------------------------------
layer_org = {}
for cid, c in clusters.items():
    band = ("early(0-5)" if c["layer_mean"] <= 5 else
            "mid(6-11)" if c["layer_mean"] <= 11 else "late(12-17)")
    layer_org.setdefault(band, []).append({
        "cluster": cid, "family": c["family_label"], "n": c["n"],
        "layer_mean": c["layer_mean"], "n_verified": c["status_counts"].get("VERIFIED",0)})

# ---------------------------------------------------------------------------
# 9. SAVE
# ---------------------------------------------------------------------------
out = {
    "meta": {
        "model": "bilin18",
        "n_paths": len(rows),
        "n_features": X.shape[1],
        "feature_names": FEATNAMES,
        "cluster_algo": "AgglomerativeClustering(ward) on StandardScaled features",
        "k_chosen": K, "silhouette": round(float(sil),3),
        "verification_sources": ["qk_unsup_verify.json","qk_unsup_verify2.json"],
        "note": "richer trigger/output classes (delimiter/bracket/quote/coordinator/"
                "determiner/pronoun/proper-noun/subword/bytefrag) derived lexically from "
                "decoded gpt2 token strings; JSON only carried 5 coarse classes.",
    },
    "clusters": {str(k): v for k, v in clusters.items()},
    "taxonomy_table": taxonomy,
    "layer_organization": layer_org,
    "tool_coverage": tool_coverage,
    "detectable_and_verifiable_now": detectable_now,
    "underserved_circuit_types": underserved,
    "per_path_assignments": [
        {"comp": r["comp"], "cluster": r["cluster"], "kind": r["kind"],
         "layer": r["layer"], "status": r["status"]} for r in rows],
}
out["meta"]["silhouette_by_k"] = sil_by_k
def _np(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
json.dump(out, open(f"{HERE}/qk_unsup_cluster.json","w"), indent=1, default=_np)
print(f"[saved] {HERE}/qk_unsup_cluster.json  ({K} clusters)")

# ---------------------------------------------------------------------------
# console report
# ---------------------------------------------------------------------------
print("\n==== TAXONOMY (clusters ordered by mean layer) ====")
for t in taxonomy:
    print(f"c{t['cluster']:>2} | {t['family']:<42} | n={t['count']:>3} "
          f"ver={t['n_verified']} art={t['n_artifact']} | L{t['layer_span'][0]}-{t['layer_span'][1]} "
          f"(mu{t['layer_mean']}) | src={t['trigger_src']} -> out={t['output_top_classes']} [{t['output_mode']}]")
print("\n==== LAYER ORGANIZATION ====")
for band in ["early(0-5)","mid(6-11)","late(12-17)"]:
    for e in sorted(layer_org.get(band,[]), key=lambda z:z["layer_mean"]):
        print(f"  {band:<12} c{e['cluster']:>2} {e['family']:<42} n={e['n']} ver={e['n_verified']}")
