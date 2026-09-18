#!/usr/bin/env python3
# BQGATE: LIBRARY -- emits the standard follow-up runner files for one readout line (review-9 efficiency item).
"""`python dod_line.py <battery_runner_module> <stem> <first_version> [--natural|--only-natural postok,negtok rows_v<N>.json:congruent1,congruent2 ...]`

Given an existing fresh-row battery runner module (one that defines CANDIDATE_ID, HEADS and build() -> rows, pos, neg, ...), writes:
  run_<stem>_dod_random_set_null_v<N>.py     (v72's body; matched-count random four-head-set null on the battery rows)
  run_<stem>_dod_response_census_v<N+1>.py   (v74's body; exact response census from the set's lowest block)
and, per --natural entry, run_<stem>_dod_<natural|pile>_v<M>.py (dod_natural_line body) for a mined rows receipt. Every file carries the
literal PREDICTIONS dict the gate reads and a docstring naming its parent; the band for the null's replay check is read from the battery
receipt if it exists. Files are written, not enqueued: read them, then `bash ops/enqueue.sh` each. This replaces the shell templates used
for v87-v90, v98-v99, v106-v109."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / "ops"

NULL = '''#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_set_beats_every_random_quadruple pred_c_random_quadruples_are_not_live pred_d_set_fraction_within_band_of_v71
"""{stem} DoD (v{v}): matched-count random four-head-set null on the {mod} fresh rows (SIMPLE). Body: v72's; 16 random quadruples from the 158
other heads, each removed along its own weight-only readout direction. Emitted by dod_line.py.
PREDICTIONS: pred_a instrument <= 1e-4; pred_b set damage > max random quadruple; pred_c no random quadruple live; pred_d pooled set fraction
within {frac} +/- 0.05 (replay of the battery; the scored key keeps v72's literal name).
PRICE (registered maximum): batches x (native + producer + set + 16 random) forwards; 0 backwards; 0 fits. Bar <= 60.
"""
from __future__ import annotations
import run_pronoun_gender_dod_random_set_null_v72 as v72
import {mod} as line

PREDICTIONS = {{"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_set_beats_every_random_quadruple": "> random max",
               "pred_c_random_quadruples_are_not_live": "none live", "pred_d_set_fraction_within_band_of_v71": "{frac} +/- 0.05 (this line's band)"}}
v72.OUT = v72.ROOT / "circuits/followups/{stem}_dod_random_set_null_v{v}_result.json"
v72.CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_random_set_null_v{v}"
v72.EXCLUDED = set(line.HEADS)
v72.POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in v72.EXCLUDED]
v72.SETS = [tuple(sorted(v72.random.Random(2026_09_18_{v} + s).sample(v72.POOL, 4))) for s in range(16)]
v72.V71_FRACTION, v72.FORWARDS_MAX = {frac}, 60
v72.v71 = type("Line", (), {{"build": staticmethod(lambda: line.build()[:3]), "HEADS": line.HEADS}})

if __name__ == "__main__":
    v72.main()
'''

CENSUS = '''#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recurrence_closure pred_b_linearization_remainder_small pred_c_direct_terms_carry_most pred_d_downstream_net_is_small pred_e_largest_responder_is_an_mlp
"""{stem} DoD (v{v}): RESPONSE CENSUS of the {mod} set removal on its fresh rows (better_circuits §3.3). Body: v74's from the set's lowest block.
DIRECT = the attention modules of the set's own blocks (block deltas also contain other heads' responses; stated). Emitted by dod_line.py.
PREDICTIONS: pred_a closure <= 1e-3; pred_b |remainder| <= 0.10 x |exact|; pred_c direct >= 0.80 of linear; pred_d |downstream net| <= 0.25 x
direct; pred_e largest downstream responder is an MLP. Prior for c/d: unsure (pronoun and person sets were direct; in/of and have/has relayed).
PRICE (registered maximum): batches x (native + edited trace) forwards; 0 backwards; 0 fits. Bar <= 8.
"""
from __future__ import annotations
import run_pronoun_gender_dod_response_census_v74 as v74
import {mod} as line

PREDICTIONS = {{"pred_a_recurrence_closure": "<= 1e-3", "pred_b_linearization_remainder_small": "<= 0.10", "pred_c_direct_terms_carry_most": ">= 0.80",
               "pred_d_downstream_net_is_small": "<= 0.25 x direct", "pred_e_largest_responder_is_an_mlp": "mlp:*"}}
v74.OUT = v74.ROOT / "circuits/followups/{stem}_dod_response_census_v{v}_result.json"
v74.CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_response_census_v{v}"
v74.FORWARDS_MAX = 8
v74.FROM = min(l for l, _ in line.HEADS)
v74.MODULES = [f"{{kind}}:{{layer:02d}}" for layer in range(v74.FROM, 18) for kind in ("attn", "mlp")]
v74.DIRECT = [f"attn:{{l:02d}}" for l in sorted({{l for l, _ in line.HEADS}})]
v74.v71 = type("Line", (), {{"build": staticmethod(lambda: line.build()[:3]), "HEADS": line.HEADS}})

if __name__ == "__main__":
    v74.main()
'''

NATURAL = '''#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""{stem} DoD (v{v}): the {mod} set on {desc} (PREDICTS OOD). Body: `dod_natural_line.run`; rows receipt `circuits/followups/{rowsf}` (outcome-blind
miner; any-sense cue filter stated). Congruent cells {cong} are the readout test; the other cells are corpus counter-cases. Emitted by dod_line.py.
PREDICTIONS (bars frozen from v20; failures preserved): pred_a instrument <= 1e-4; pred_b congruent capability >= 0.75 per cell; pred_c congruent mean
damage >= 0.15 and >= 60% positive; pred_d > null max; pred_e three gates; pred_f incongruent mean damage <= 0 (None if the cells are empty).
PRICE (registered maximum): batches x (native + producer + set + 16 nulls) forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import {mod} as line

CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_{kind}_v{v}"
ROWS = N.ROOT / "circuits/followups/{rowsf}"
OUT = "{stem}_dod_{kind}_v{v}_result.json"
CONGRUENT = {cong}
HEADS = {heads}
PREDICTIONS = {{"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}}


def main() -> None:
    rows, sha, pos, neg = N.rows_from_receipt(ROWS, "{poslab}", "{postok}", "{negtok}", lambda r: f"natural_{{r['cue']}}_{{r['label']}}")
    N.run(CANDIDATE_ID, OUT, rows, sha, pos, neg, HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
'''


def main():
    mod, stem, v0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
    line = __import__(mod)
    heads = "line.HEADS"
    if "--heads" in sys.argv:
        i = sys.argv.index("--heads"); heads = "(" + ", ".join(f"({a})" for a in sys.argv[i + 1].split(";")) + ")"; del sys.argv[i:i + 2]
    frac = None
    for f in (ROOT / "circuits/followups").glob(f"{stem}_dod_battery_v*_result.json"):
        frac = round(json.loads(f.read_text())["joint"]["target_damage_fraction"], 2)
    if frac is None and not (len(sys.argv) > 4 and sys.argv[4] == "--only-natural"):
        raise SystemExit("battery receipt not found; run the battery first (the null replays its fraction)")
    written = []
    args = sys.argv[4:]
    only_natural = args and args[0] == "--only-natural"
    if only_natural:
        args = ["--natural"] + args[1:]; v = v0
    else:
        p = OPS / f"run_{stem}_dod_random_set_null_v{v0}.py"; p.write_text(NULL.format(stem=stem, v=v0, mod=mod, frac=frac)); written.append(p)
        p = OPS / f"run_{stem}_dod_response_census_v{v0 + 1}.py"; p.write_text(CENSUS.format(stem=stem, v=v0 + 1, mod=mod)); written.append(p)
        v = v0 + 2
    if args and args[0] == "--natural":
        postok, negtok = args[1].split(",")
        for spec in args[2:]:
            rowsf, cong = spec.split(":"); cong = tuple(cong.split(","))
            kind = "pile" if "pile" in rowsf else "natural"
            desc = "Pile rows (out-of-corpus)" if kind == "pile" else "natural FineWeb rows (training corpus, out-of-panel)"
            p = OPS / f"run_{stem}_dod_{kind}_v{v}.py"
            p.write_text(NATURAL.format(stem=stem, v=v, mod=mod, desc=desc, rowsf=rowsf, cong=repr(cong), kind=kind, poslab=postok.strip(), postok=postok, negtok=negtok, heads=heads)); written.append(p); v += 1
    for p in written:
        p.chmod(0o755); print("wrote", p.name)


if __name__ == "__main__":
    main()
