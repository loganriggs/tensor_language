"""Assemble the multi-family results explainer (general.html) with figures embedded.

Usage: python build_gen.py   (after analysis_general.py etc.; reads figures/ and seeds.json)
"""

import base64
import json
from pathlib import Path

FIGURES = Path("figures")


def uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((FIGURES / name).read_bytes()).decode()


seed_rows = ""
if Path("seeds.json").exists():
    for row in json.loads(Path("seeds.json").read_text()):
        seed_rows += f"<tr><td>{row['condition']}</td><td>{row['seeds']}</td><td>{row['corr']}</td></tr>\n"

HTML = f"""<title>One model, many graphs — results</title>
<style>
  .doc {{
    --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --secondary: #52514e;
    --muted: #898781; --hairline: rgba(11,11,11,0.10); --accent: #2a78d6; --neg: #8c2b2b;
    background: var(--surface); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.55; padding: 36px 20px 64px; box-sizing: border-box;
  }}
  .doc .col {{ max-width: 900px; margin: 0 auto; }}
  .doc h1 {{ font-size: 24px; font-weight: 650; margin: 0 0 4px; text-wrap: balance; }}
  .doc p.dek {{ color: var(--secondary); font-size: 14.5px; margin: 0 0 24px; max-width: 70ch; }}
  .doc h2 {{ font-size: 17px; font-weight: 650; margin: 40px 0 10px; padding-top: 18px; border-top: 1px solid var(--hairline); }}
  .doc p {{ font-size: 14px; max-width: 74ch; margin: 8px 0; }}
  .doc code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
               background: #f3f2ee; border-radius: 3px; padding: 0 4px; }}
  .doc img {{ max-width: 100%; border: 1px solid var(--hairline); border-radius: 6px; margin: 10px 0; }}
  table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
  th, td {{ border: 1px solid var(--hairline); padding: 5px 12px; text-align: right; font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f3f2ee; font-weight: 600; }}
  td b {{ color: var(--accent); }}
  .verdict {{ background: #eef4fc; border: 1px solid var(--hairline); border-radius: 6px; padding: 10px 16px;
              font-size: 13.5px; margin: 12px 0; }}
  .cap {{ color: var(--muted); font-size: 12.5px; max-width: 80ch; margin-top: -4px; }}
</style>
<div class="doc"><div class="col">
<h1>One model, many graphs: does mixed training create real geometry?</h1>
<p class="dek">Single-task 2-layer models learned task-specific circuits and stored graph structure
"anti-organized" (neighbors pushed apart). We trained one model on six graph families at once —
ring, directed ring (&equiv; cycles), grid, cylinder, random tree, random 3-regular; torus and
Erd&#337;s&ndash;R&eacute;nyi held out entirely — and measured performance, in-context representation
geometry (Park et&nbsp;al. protocol), and circuits. Hypotheses were fixed before running; the full
trail is in <code>LOG.md</code>. Reproduce: <code>train_general.py / analysis_general.py /
mech_general.py</code>.</p>

<h2>1 · Performance: two architectures master everything, zero-shot included</h2>
<img src="{uri('gen_perf.png')}" alt="performance heatmap">
<img src="{uri('gen_curves.png')}" alt="loss and generalization curves">
<p class="cap">Training dynamics. The bilinear generalist goes through a phase transition at ~6k steps —
all families, held-out ones included, jump together (directed-ring at unseen length 27 goes 0.0 &rarr; 0.6+
at the same moment), and the train loss drops in lock-step. The softmax induction stack converges fast and
smoothly instead. Held-out curves (dashed red) track trained curves (solid blue) throughout — generalization
is not an end-of-training bonus, it appears the moment the algorithm does.</p>
<p class="cap">bilin-lerp-2L and softmax-add-3L reach ~1.00 legal on all six train families AND
on never-trained torus / ER graphs / larger sizes. Failures are informative: bilin-add-3L diverges
(the additive-residual depth fix is unstable here), softmax-lerp-2L fails only the deterministic
family (dring 0.10), and the weak archs drop to 0.00 on directed rings at unseen length.</p>

<p style="font-size:13.5px"><b>Interactive 3D version of the geometry comparison</b> (node means + per-position
datapoint clouds, drag to rotate, context slider):
<a href="https://claude.ai/code/artifact/6e85f3a9-aa05-4bbd-ba98-addbaaba3e16">geo_compare_3d</a>.</p>

<h2>2 · The headline: genuine Park-style geometry — neighbors stored nearby</h2>
<p>Park et&nbsp;al.'s Theorem&nbsp;5.1 gives a sharp test for real energy-minimizing organization:
the graph's spectral coordinates must appear in the <b>top two principal components</b> of the
windowed mean token representations. The multi-family bilinear model passes; the single-task model
and the softmax champion do not:</p>
<table>
<tr><th>model (grid docs, ctx 256)</th><th>|corr| PC1&harr;z&#8322; / PC2&harr;z&#8323;</th><th>top-2-PC Dirichlet energy (random &asymp; 2)</th><th>lattice's best PC</th></tr>
<tr><td>single-task grid (anti)</td><td>0.08 / 0.03</td><td>2.69</td><td>PC12</td></tr>
<tr><td><b>multi-family bilin-lerp-2L</b></td><td><b>0.81 / 0.80</b></td><td><b>0.49</b></td><td><b>PC1</b></td></tr>
<tr><td>multi-family softmax-add-3L (anti)</td><td>0.02 / 0.01</td><td>3.94</td><td>PC14</td></tr>
</table>
<img src="{uri('gen_park.png')}" alt="Park test side by side">
<p class="cap">Top-2-PC projections, Procrustes-rotated onto the true layout. The multi-family
model's map is a (warped) 4&times;5 sheet with local edges — actual neighbors-nearby geometry, not
a sign convention. The same model draws a clean phase heptagon for cycles (below), where the
anti-organizing softmax model draws a 7-pointed star.</p>
<img src="{uri('gen_circle.png')}" alt="cycle phase structure by arch">

<h2>3 · Not a sign flip, not an init lottery</h2>
<p>Two objections we tested. <b>(a) "Bilinear sign is gauge."</b> True for the attention pattern
(flip q&#8321; and o together), but the measurement lives on the shared residual stream, where
orthogonal reparameterizations leave the Gram matrix unchanged — and the observed difference is
not a sign at all: it's <em>where</em> the lattice sits in the variance spectrum (PC1&ndash;2 vs
buried at PC12 at 1.8% variance). No symmetry moves structure across the spectrum.
<b>(b) "It's init luck."</b> Partially right — and testing it sharpened the result. For
<em>single-family</em> training the sign IS a seed lottery. But two-family and six-family
conditions pin the mode reliably, in opposite directions — the data distribution selects the
geometry; single-family training merely fails to constrain it:</p>
<table>
<tr><th>condition</th><th>seeds</th><th>grid Gram&ndash;adjacency corr @ ctx 256</th></tr>
{seed_rows}
</table>

<h2>4 · What flips it: the mixture's stochastic diversity, not conflict</h2>
<p>Pre-registered guess: the deterministic directed-ring family (which punishes recency shortcuts)
drives the flip. <b>Falsified</b> — it does the opposite:</p>
<table>
<tr><th>bilin-lerp-2L trained on</th><th>grid corr @ ctx 256</th></tr>
<tr><td>grid only</td><td>&minus;0.14 … +0.67 (4 seeds — unconstrained)</td></tr>
<tr><td>grid + dring (deterministic)</td><td><span style="color:var(--neg)">&minus;0.55, &minus;0.70, &minus;0.72 (3 seeds) — reliably anti</span></td></tr>
<tr><td>grid + cylinder</td><td>+0.24 (1 seed)</td></tr>
<tr><td>grid + ring</td><td>+0.38 (1 seed)</td></tr>
<tr><td>grid + tree</td><td>+0.41 (1 seed)</td></tr>
<tr><td>all six families</td><td><b>+0.55 … +0.66 (3 seeds) — reliably positive</b></td></tr>
</table>
<div class="verdict"><b>Synthesis:</b> organization tracks the <em>algorithmic mode</em>.
Deterministic next-token copying selects induction-style circuits, which anti-organize — the
perfect softmax-add-3L is mechanically a textbook previous-token + induction stack (third layer
nearly idle) and anti-organizes even on the full stochastic mixture. Predicting
<em>neighborhoods as sets</em> across diverse stochastic families builds the positive,
Park-style map. Open question: both modes use K-composition matching, so exactly why induction
anti-organizes is not yet explained.</div>

<h2>5 · The circuit rewired (and organization by architecture)</h2>
<img src="{uri('gen_org.png')}" alt="organization by architecture">
<p class="cap">Gram&ndash;adjacency correlation vs context for all five architectures (left) and their
top-PC maps on grid and never-trained torus docs. The single-task reference (&minus;0.57) is the
dashed line.</p>
<img src="{uri('gen_mech.png')}" alt="attention profiles single vs multi">
<p class="cap">Same documents, single-task vs multi-family circuits. The multi model dropped the
cycle model's "suppress the last 3 tokens" elimination hack (removing layer 2 now scores 0.00 at
L=5, was 0.88), attends <em>positively</em> at induction offsets, relies on K-composition
absolutely (cutting it: 0.00), and abandoned V-composition (cutting it barely hurts — and even
helps deterministic cycles). One relational circuit for every family — and it coincides with the
positive geometry.</p>


<h2>6 · Session 2: WHY neighbors end up nearby (resolved)</h2>
<p>A node's representation must carry the model's <em>prediction</em> — positive neighbor-token
evidence in the unembedding basis (found in all 18 models tested; it IS the prediction). That
evidence alone always forms the positive map, because adjacent nodes' predictions overlap
<em>through each other</em>. Representations also carry own/recent-token content whose sign is
behaviorally free: "don't predict what can't follow" can be implemented as suppression in the
write paths (negative own content → anti-map) or in the static readout (positive writes,
cancelled at the logits → positive map). Training data pins the choice via <b>reversibility</b>:
walks that can never return to their recent past force suppression into the writes (directed
rings at any entropy: −0.38…−0.70) while any nonzero backtrack rate keeps the map positive
(biased ring with 12.5% backtracks and LOWER entropy: +0.67). Full trail: results_why.md / LOG.md.</p>
<img src="{uri('geo_why.png')}" alt="why neighbors end up nearby">
<p class="cap">Left: organization tracks the summed own-token write coefficient across all 18 models
(r = 0.76). Middle: projecting own-token directions out of each node's rep moves every model toward
the positive map — the positive map is always underneath. Right: the pinning variable is
reversibility, not entropy.</p>

<p style="color:var(--muted); font-size:12.5px; margin-top:28px">Companion files: results_general.md
(this content + numbers), LOG.md (hypothesis &rarr; verdict trail), graphs.py / train_general.py /
analysis_general.py / mech_general.py. Earlier reports: results.md (cycles), results_graphs.md
(lattices), mech.html (single-task circuits).</p>
</div></div>
"""

Path("general.html").write_text(HTML)
print(f"wrote general.html ({len(HTML) / 1024:.0f} kB)")
