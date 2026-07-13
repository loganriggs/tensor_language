"""Assemble the mechanistic explainer (mech.html) with figures embedded as data URIs.

Usage: python build_mech.py   (after mech.py; writes mech.html)
"""

import base64
from pathlib import Path

FIGURES = Path("figures")


def uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((FIGURES / name).read_bytes()).decode()


HTML = f"""<title>Mechanistics of the 2-layer bilinear models</title>
<style>
  .doc {{
    --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --secondary: #52514e;
    --muted: #898781; --hairline: rgba(11,11,11,0.10); --accent: #2a78d6; --neg: #8c2b2b;
    background: var(--surface); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.55; padding: 36px 20px 64px; box-sizing: border-box;
  }}
  .doc .col {{ max-width: 880px; margin: 0 auto; }}
  .doc h1 {{ font-size: 24px; font-weight: 650; margin: 0 0 4px; text-wrap: balance; }}
  .doc p.dek {{ color: var(--secondary); font-size: 14.5px; margin: 0 0 28px; max-width: 68ch; }}
  .doc h2 {{ font-size: 17px; font-weight: 650; margin: 40px 0 10px; padding-top: 18px; border-top: 1px solid var(--hairline); }}
  .doc h3 {{ font-size: 14px; font-weight: 600; margin: 22px 0 6px; }}
  .doc p {{ font-size: 14px; max-width: 72ch; margin: 8px 0; }}
  .doc code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
               background: #f3f2ee; border-radius: 3px; padding: 0 4px; }}
  .doc img {{ max-width: 100%; border: 1px solid var(--hairline); border-radius: 6px; margin: 10px 0; }}
  .paths {{ display: flex; flex-direction: column; gap: 10px; margin: 14px 0; }}
  .path {{ background: var(--panel); border: 1px solid var(--hairline); border-left: 3px solid var(--accent);
           border-radius: 6px; padding: 10px 14px; }}
  .path b {{ font-size: 13.5px; }}
  .path p {{ font-size: 13.5px; margin: 4px 0 0; color: var(--secondary); }}
  .path .wire {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--muted); }}
  table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
  th, td {{ border: 1px solid var(--hairline); padding: 5px 12px; text-align: right;
            font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f3f2ee; font-weight: 600; }}
  td.hit {{ color: var(--neg); font-weight: 600; }}
  .unknown {{ background: #fbf7ee; border: 1px solid var(--hairline); border-radius: 6px; padding: 12px 16px; }}
  .unknown li {{ font-size: 13.5px; margin: 5px 0; }}
  .cap {{ color: var(--muted); font-size: 12.5px; max-width: 78ch; margin-top: -4px; }}
</style>
<div class="doc"><div class="col">
<h1>How the two-layer bilinear models work</h1>
<p class="dek">A mechanistic account of the cycle model (2L·d64·1h) and the grid model (2L·d128·1h) —
bilinear attention only, no softmax, no norms, <code>lerp</code>-0.5 residual. Evidence: attention-offset
profiles, the two bilinear score factors, token→logit maps per weight path, and causal wire ablations
(every claim below has an ablation or a weight-space measurement behind it; reproduce with
<code>python mech.py</code>). The last section lists what we could <em>not</em> explain.</p>

<p>One architectural fact drives everything: with no softmax, attention weights are <b>signed</b> —
the pattern is a product of two dot-product score maps, <code>(q₁·k₁)(q₂·k₂)/d²</code>, so the model
freely uses <em>negative attention</em>, and a path's effect is the sign of the attention <em>times</em>
the sign of its copy circuit. Both models exploit this.</p>

<h2>1 · The cycle model: eliminate, then sign-flipped induction</h2>
<p>Task: a random cycle of L distinct tokens tiles the context; predict the next token.
The trained model decomposes into three additive paths into the logits
(stream = 0.25·embed + 0.25·L1-out + 0.5·L2-out):</p>

<div class="paths">
  <div class="path"><b>Path 0 · direct: "not the token I'm reading".</b>
    <p>The embed→unembed map has a negative diagonal (−0.32 ± 0.33): the current token's own logit is suppressed.</p>
    <span class="wire">0.25·W_U W_E, diag &lt; 0</span></div>
  <div class="path"><b>Path 1 · layer 1: eliminate the recent, shortlist the seen.</b>
    <p>Layer 1 attends <em>negatively</em> to offsets 0–3 (−0.71, −0.59, −0.38, −0.18) and mildly positively to offsets ~7–20,
    and its OV→logits map is diagonal (top-1 on the diagonal for 100 of 100 tokens). Net effect: subtract the logits of the
    last few tokens, add the logits of everything seen in the mid-range window. For a cycle of distinct tokens the next token
    is never among the last min(3, L−2) tokens, so this is a safe process of elimination — and for L=5 it is nearly sufficient
    alone: with layer 2 removed the model still scores <b>0.88</b> at L=5. This also explains why 1-layer models topped out
    exactly on short cycles in the original sweep.</p>
    <span class="wire">pattern₁(offsets 0–3) &lt; 0 × diag(W_U W_O1 W_V1 W_E) &gt; 0 ⇒ suppression</span></div>
  <div class="path"><b>Path 2 · layer 2: induction with two minus signs.</b>
    <p>Layer 2's attention spikes <em>negatively</em> at offsets kL−1 and kL — the previous occurrences of the current
    token and the slots right after them (mean pattern on "x<sub>s−1</sub>=x<sub>t</sub>" keys: −0.28; unrelated keys: +0.02).
    Its token-copy circuit also has a negative diagonal (−0.19). Negative attention × negative copy = <b>positive logit for
    the true next token</b>. The bilinear product factorizes cleanly: factor 2 is a <em>periodic same-phase detector</em>
    (peaks +1.1…+1.4 at offsets 0, L, 2L, 3L), factor 1 is a <em>recency envelope</em> (positive near, negative far);
    their product is the negative induction stripe.</p>
    <span class="wire">pattern₂(kL−1) &lt; 0 × diag(W_U W_O2 W_V2 W_E) &lt; 0 ⇒ boost x_{{t+1}}</span></div>
</div>

<img src="{uri('mech_cycle.png')}" alt="cycle model attention structure">
<p class="cap">Left: layer-1 signed attention by offset. Middle: layer-2 attention, negative spikes at kL−1/kL.
Right: the two bilinear factors — a period-L phase detector times a recency envelope.</p>

<img src="{uri('mech_cycle_ov.png')}" alt="cycle model OV maps">
<p class="cap">Token→logit maps along each weight path (30×30 corner). Diagonals: direct −, L1 +, L2-copy −,
L2∘L1 mixed. Note the diagonals are <em>per-token mixed in sign</em> (std ≫ |mean|) — the sign claims above are
about the average; see §3.</p>

<h3>Causal check: cut each wire</h3>
<table>
<tr><th>variant</th><th>L=5</th><th>L=10</th><th>L=15</th><th>L=20</th><th>L=25</th><th>L=30</th></tr>
<tr><td>full model</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.97</td><td>0.85</td><td>0.71</td></tr>
<tr><td>layer 1 removed</td><td class="hit">0.01</td><td class="hit">0.01</td><td class="hit">0.01</td><td class="hit">0.01</td><td class="hit">0.01</td><td class="hit">0.01</td></tr>
<tr><td>layer 2 removed</td><td>0.88</td><td class="hit">0.26</td><td class="hit">0.10</td><td class="hit">0.07</td><td class="hit">0.01</td><td class="hit">0.00</td></tr>
<tr><td>L2 queries from pre-L1 stream</td><td>0.83</td><td class="hit">0.38</td><td class="hit">0.26</td><td class="hit">0.18</td><td class="hit">0.04</td><td class="hit">0.03</td></tr>
<tr><td>L2 keys from pre-L1 stream</td><td>0.86</td><td class="hit">0.31</td><td class="hit">0.17</td><td class="hit">0.10</td><td class="hit">0.01</td><td class="hit">0.01</td></tr>
<tr><td>L2 values from pre-L1 stream</td><td>0.89</td><td class="hit">0.35</td><td class="hit">0.14</td><td class="hit">0.10</td><td class="hit">0.01</td><td class="hit">0.01</td></tr>
</table>
<p>Layer 1 is load-bearing for everything. Layer 2 is unnecessary at L=5 (elimination suffices) and essential beyond.
And <b>all three</b> of layer 2's inputs — queries, keys, <em>and</em> values — must read the layer-1-written local
window: the "phase" that factor 2 detects is literally the window of recent tokens layer 1 deposits at each position,
and part of the copied content comes through it too (K-, Q- and V-composition all matter).</p>

<h2>2 · The grid model: backtrack baseline + two copy routes</h2>
<p>Task: uniform random walk on a token-labeled lattice; predict the (stochastic) next token — score is
probability mass on the true neighbors. The circuit rhymes with the cycle one but chooses <em>positive</em>
attention, and it must recover neighbors in <em>both</em> walk directions:</p>

<div class="paths">
  <div class="path"><b>Path 0 · direct: strong self-suppression.</b>
    <p>Diagonal −4.12 ± 1.82 — a walk never stays put, and layer 2's routes below both leak the current token; this cancels it.</p></div>
  <div class="path"><b>Path 1 · layer 1: previous-token writer, and a backtrack baseline.</b>
    <p>Layer 1 is a small negative previous-token head (offset 1: −0.18). Through its own logit path this boosts
    the token you just came from — which is always a legal move: with layer 2 removed the model still has
    <b>0.84</b> legal rate (but only 0.40 mass). Its more important job is writing each position's predecessor
    into the stream for layer 2 to read.</p></div>
  <div class="path"><b>Path 2 · layer 2: attend to revisits; copy successors directly, predecessors via the window.</b>
    <p>Attention is positive and concentrates on <b>self-matches</b> (past occurrences of the current node, +0.080)
    and <b>successor slots</b> (positions right after them, +0.034; unrelated: +0.007). Two copy routes then extract
    both neighbor directions: the raw token-copy circuit (diag +2.59) turns attention on successor slots into successor
    logits; and the composite circuit through layer 1 (diag −0.65, times layer 1's −0.18 write) turns attention on
    self-matches into <b>predecessor</b> logits via a double negative.</p>
    <span class="wire">succ: p₂(+) × diag(U O₂V₂E) &gt; 0 · pred: p₂(+) × diag(U O₂V₂O₁V₁E) &lt; 0 × p₁(−) ⇒ +</span></div>
</div>

<img src="{uri('mech_grid.png')}" alt="grid model attention structure">
<img src="{uri('mech_grid_ov.png')}" alt="grid model OV maps">
<p class="cap">Grid model: layer-1 offset profile, layer-2 attention by key-token relation, and the per-path
token→logit maps (direct path uniformly negative on the diagonal; L2 raw copy positive).</p>

<h3>Causal check</h3>
<table>
<tr><th>variant</th><th>legal rate</th><th>neighbor mass</th></tr>
<tr><td>full model</td><td>0.989</td><td>0.762</td></tr>
<tr><td>layer 1 removed</td><td class="hit">0.200</td><td class="hit">0.191</td></tr>
<tr><td>layer 2 removed</td><td>0.839</td><td class="hit">0.400</td></tr>
<tr><td>L2 queries from pre-L1 stream</td><td>0.878</td><td><b>0.864</b></td></tr>
<tr><td>L2 keys from pre-L1 stream</td><td class="hit">0.601</td><td class="hit">0.357</td></tr>
<tr><td>L2 values from pre-L1 stream</td><td>0.711</td><td class="hit">0.529</td></tr>
</table>
<p>K-composition is how matches are found (cutting it: 0.60 legal); V-composition carries the predecessor route
(mass 0.76 → 0.53). Cutting Q-composition <em>raises</em> mass while lowering legal rate — see §3.</p>

<h2>3 · What we don't understand</h2>
<div class="unknown"><ul>
<li><b>The Q-composition paradox (grid).</b> Feeding layer 2's queries from the pre-L1 stream <em>improves</em> neighbor
mass (0.86 vs 0.76) while hurting argmax legality. The full model's queries evidently sharpen attention in a way that
helps the top-1 but miscalibrates the distribution. We measured it; we can't yet say why.</li>
<li><b>Per-token sign variation in the copy circuits.</b> The OV diagonals are near-perfectly diagonal (top-1 for
~100/100 tokens) but with large mixed-sign spread (e.g. +0.36 ± 2.55). The narrative above uses the mean sign; how the
per-token signs coordinate with the embedding geometry (and with the signed attention) is unexplained.</li>
<li><b>Why opposite signs across tasks?</b> The cycle model settles on negative attention × negative copy; the grid
model on positive × positive (plus a double-negative side route). Both work; we don't know what breaks the symmetry
during training.</li>
<li><b>Grid factor roles.</b> Unlike the cycle model's clean phase-detector × envelope, the grid model's two bilinear
factors are not individually interpretable — only their product is (e.g. successor slots get −0.22 × −0.12).</li>
<li><b>Small unexplained attention features</b>, e.g. the cycle model's positive layer-2 weight at offset 0 and at
offsets ~24–25, and the exact role of rotary inside the factors (we never isolated it).</li>
<li><b>Scope.</b> One seed per model; cylinder/torus not analyzed (the grid story plausibly transfers; unverified).
The link between this circuit and the anti-organized in-context representations (neighbors kept separable helps the
key-matching) is consistent but not established causally.</li>
</ul></div>

<p style="color:var(--muted); font-size:12.5px; margin-top:28px">Companion files: results.md (cycles),
results_graphs.md (lattices), mech.py (this analysis), figures/mech_*.png. Models: runs/L2_d64_h1,
runs_geo/grid_L2_d128_long.</p>
</div></div>
"""

Path("mech.html").write_text(HTML)
print(f"wrote mech.html ({len(HTML) / 1024:.0f} kB)")
