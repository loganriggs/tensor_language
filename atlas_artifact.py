import base64
import json
from pathlib import Path

REPO = Path("/workspace/tensor_language")
FIG = REPO / "figures"
OUT = Path("/tmp/claude-0/-workspace-tensor-language/9dd2caa2-0596-4379-9da3-1957a40d185f/scratchpad/circuit-atlas.html")

demo = json.loads((REPO / "runs_lm/circuit_demo.json").read_text())
ind_ex = json.loads((REPO / "runs_lm/induction_examples.json").read_text())
ng = json.loads((REPO / "runs_lm/ngram_demo.json").read_text())
khop = json.loads((REPO / "runs_hop/khop_examples.json").read_text())


def b64(name):
    return base64.b64encode((FIG / name).read_bytes()).decode()


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "⏎"))


css = """
:root{
  --bg:#fbfaf7; --panel:#f4f2ec; --ink:#171a1e; --sub:#5b6068; --hair:#e0ddd2;
  --accent:#2f6fd0; --neg:#b3372f; --ok:#2e7d4f; --mono-bg:#f1efe8; --hl:#fdeaa8;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#15171b; --panel:#1d2025; --ink:#e7e8e4; --sub:#9aa0a9; --hair:#2c2f35;
         --accent:#6ba3ee; --neg:#e2705f; --ok:#57b380; --mono-bg:#22252b; --hl:#5a4d1e; }
}
:root[data-theme="dark"]{ --bg:#15171b; --panel:#1d2025; --ink:#e7e8e4; --sub:#9aa0a9;
  --hair:#2c2f35; --accent:#6ba3ee; --neg:#e2705f; --ok:#57b380; --mono-bg:#22252b; --hl:#5a4d1e; }
:root[data-theme="light"]{ --bg:#fbfaf7; --panel:#f4f2ec; --ink:#171a1e; --sub:#5b6068;
  --hair:#e0ddd2; --accent:#2f6fd0; --neg:#b3372f; --ok:#2e7d4f; --mono-bg:#f1efe8; --hl:#fdeaa8; }

body{ background:var(--bg); color:var(--ink);
  font-family:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  line-height:1.55; margin:0; font-size:16.5px; }
.col{ max-width:920px; margin:0 auto; padding:40px 22px 90px; }
.mono, code, .ex, table, .chip, .eyebrow, .tok, .headchip, .tabs button, .probbar, .small-mono{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
.eyebrow{ font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); }
h1{ font-size:34px; line-height:1.15; margin:8px 0 6px; text-wrap:balance; font-weight:600;}
.lede{ color:var(--sub); font-size:17px; max-width:66ch; margin:0 0 8px; }
.meta{ color:var(--sub); font-size:13px; margin-bottom:18px;}
h2{ font-size:23px; margin:6px 0 4px; text-wrap:balance; font-weight:600;}
h3{ font-size:16.5px; margin:24px 0 8px; font-weight:600;}
p{ max-width:72ch; margin:10px 0; }
.tabs{ display:flex; flex-wrap:wrap; gap:8px; margin:22px 0 8px; position:sticky; top:0;
  background:var(--bg); padding:10px 0; z-index:5; border-bottom:1px solid var(--hair);}
.tabs button{ font-size:12.5px; letter-spacing:.04em; padding:7px 13px; border-radius:999px;
  border:1px solid var(--hair); background:var(--panel); color:var(--ink); cursor:pointer; }
.tabs button:hover{ border-color:var(--accent); }
.tabs button.on{ background:var(--accent); border-color:var(--accent); color:#fff; }
.tabs button:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }
section.tab{ display:none; padding-top:14px; }
section.tab.on{ display:block; }
.chip{ font-size:11px; letter-spacing:.1em; padding:2px 9px; border-radius:999px;
  border:1px solid; white-space:nowrap; position:relative; top:-2px;}
.chip.ok{ color:var(--accent); border-color:var(--accent); }
.ex{ font-size:13px; background:var(--mono-bg); border:1px solid var(--hair);
  border-radius:6px; padding:9px 12px; margin:8px 0; overflow-x:auto; }
.ex b{ color:var(--neg); font-weight:700; }
.ex mark{ background:var(--hl); color:inherit; border-radius:3px; padding:0 1px;}
.ex .ce{ color:var(--sub); }
.tablewrap{ overflow-x:auto; margin:14px 0; }
table{ border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
th,td{ border:1px solid var(--hair); padding:5px 11px; text-align:right; }
th:first-child,td:first-child{ text-align:left; }
th{ background:var(--panel); font-weight:600; }
td.hi{ color:var(--accent); font-weight:700; }
td.lo{ color:var(--neg); font-weight:700; }
figure{ margin:18px 0; } figure img{ max-width:100%; border:1px solid var(--hair); border-radius:8px; display:block;}
figcaption{ font-size:13px; color:var(--sub); margin-top:6px; max-width:80ch; }
.finding{ border-left:3px solid var(--accent); padding:2px 0 2px 16px; margin:16px 0; max-width:70ch; }
.small{ font-size:13.5px; color:var(--sub); }

/* vertical circuit view */
.circuit{ background:var(--panel); border:1px solid var(--hair); border-radius:10px;
  padding:18px; margin:16px 0; }
.stage{ margin:0 0 4px; }
.stagelabel{ font-family:ui-monospace,Menlo,monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--sub); margin:10px 0 6px; }
.probbar{ display:flex; align-items:center; gap:14px; }
.probbar .num{ font-size:26px; font-weight:700; min-width:120px;}
.probbar .track{ flex:1; height:14px; background:var(--mono-bg); border:1px solid var(--hair);
  border-radius:7px; overflow:hidden; position:relative;}
.probbar .fill{ height:100%; background:var(--accent); border-radius:7px 0 0 7px;
  transition:width .25s; }
.probbar .base{ position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--sub); }
.delta{ font-size:13px; color:var(--sub); min-height:20px; margin-top:4px;}
.headrow{ display:flex; flex-wrap:wrap; gap:8px; margin:6px 0 10px; }
.headchip{ font-size:12.5px; padding:8px 11px; border-radius:8px; border:1px solid var(--hair);
  background:var(--bg); cursor:pointer; text-align:left; color:var(--ink);}
.headchip .hp{ display:block; font-size:11px; color:var(--sub); }
.headchip:hover{ border-color:var(--accent); }
.headchip.sel{ border-color:var(--neg); background:var(--mono-bg); box-shadow:0 0 0 1px var(--neg); }
.headchip:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }
.ribbon{ display:flex; flex-wrap:wrap; gap:0; margin:6px 0 0; padding:0;}
.tok{ font-size:13px; padding:2px 0; border-radius:0; border:1px solid transparent;
  white-space:pre; background:none; }
.circuit.flush{ padding-left:6px; padding-right:6px; }
.wire{ position:absolute; inset:0; pointer-events:none; overflow:visible; }
.wired{ position:relative; }
.hbox{ display:inline-block; font-family:ui-monospace,Menlo,monospace; font-size:12.5px;
  border:1.5px solid var(--accent); border-radius:8px; padding:7px 12px; background:var(--bg);
  cursor:pointer; color:var(--ink); text-align:left;}
.hbox .role{ display:block; font-size:10.5px; color:var(--sub); letter-spacing:.05em;}
.hbox.sel{ background:var(--mono-bg); border-color:var(--neg); box-shadow:0 0 0 1px var(--neg);}
.hbox:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }
.hrow{ display:flex; gap:40px; margin:26px 0; justify-content:center; }
.ctxheads{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
.ctxheads .headchip{ font-size:11.5px; padding:5px 9px; }
.tok.q{ border-color:var(--neg); font-weight:700;}
.tok.match{ background:var(--hl); }
.tok.src{ background:var(--hl); border-color:var(--accent); font-weight:700;}
.tok.att{ border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
.tok.tgt{ border:1px dashed var(--ok); color:var(--ok); font-weight:700;}
.legend{ font-size:12px; color:var(--sub); margin-top:8px;}
.legend span{ margin-right:14px; }
.pathrow{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-size:13px;}
.arrow{ color:var(--sub); }
.axes{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:18px 0; }
.axes .cell{ background:var(--panel); border:1px solid var(--hair); border-radius:8px; padding:14px 16px; }
.axes h4{ margin:0 0 6px; font-size:15px; } .axes p{ font-size:14px; color:var(--sub); margin:0; }
.axes .tag{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--accent);
  letter-spacing:.08em; text-transform:uppercase; }
@media (max-width:640px){ .axes{ grid-template-columns:1fr; } h1{font-size:28px;} }
a{ color:var(--accent); }
"""

# ---------------- induction interactive data ----------------
start, q, j = demo["start"], demo["q"], demo["j"]
ribbon = demo["ribbon"]                       # tokens start..q inclusive
heads_js = {}
for name, h in demo["heads"].items():
    heads_js[name] = {"p": h["ablated_p"],
                      "att": [t["pos"] - start for t in h["top"] if t["pos"] >= start]}
IND = {"base": demo["base_p"], "target": demo["target"], "heads": heads_js,
       "q": q - start, "j": j - start, "src": j + 1 - start}

rib_html = ""
for i, t in enumerate(ribbon):
    cls = "tok"
    if i == IND["q"]: cls += " q"
    elif i == IND["j"]: cls += " match"
    elif i == IND["src"]: cls += " src"
    rib_html += f'<span class="{cls}" data-i="{i}">{esc(t)}</span>'
rib_html += f'<span class="tok tgt">{esc(demo["target"])} ?</span>'


def headchips(layer, desc):
    out = [f'<div class="stagelabel">Layer {layer} — {desc} (click a head to ablate it)</div><div class="headrow">']
    for hi in range(4):
        name = f"L{layer}H{hi}"
        p = demo["heads"][name]["ablated_p"]
        out.append(f'<button class="headchip" data-h="{name}">{name}'
                   f'<span class="hp">ablate → P = {p:.2f}</span></button>')
    out.append("</div>")
    return "".join(out)


xnor_rows = ""
for hi in range(4):
    name = f"L1H{hi}"
    h = demo["heads"][name]
    a = demo["xnor"][name]
    strong = ' class="hi"' if name == "L1H2" else ""
    xnor_rows += (f"<tr><td>{name}</td><td>{h['src_score']:+.3f}</td><td>{h['src_ov']:+.3f}</td>"
                  f"<td{strong}>{h['src_product']:+.3f}</td>"
                  f"<td{strong}>{a['mean_product']:+.3f}</td><td>{a['frac_positive']:.0%}</td></tr>")

ind_ex_html = ""
for e in ind_ex[:4]:
    ind_ex_html += (f'<div class="ex">…{esc(e["pre"])}<mark>{esc(e["rep"])}</mark>'
                    f'{esc(e["mid"])}<mark>{esc(e["cur"])}</mark> ⟶ <b>{esc(e["tgt"])}</b>'
                    f' <span class="ce">(attn2 CE {e["ce"]})</span></div>')

# ---------------- n-gram examples ----------------
ng_html = ""
for e in ng:
    toks = "".join(f'<span class="tok match">{esc(t)}</span>' for t in e["context_tail_tokens"][1:])
    lead = f'<span class="tok">{esc(e["context_tail_tokens"][0])}</span>'
    crows = "".join(f"<tr><td>P(tgt | last {c['order']})</td>"
                    f"<td>{c['p'] if c['p'] is not None else '—'}</td>"
                    f"<td class='small'>{c['count']:,} occurrences</td></tr>" for c in e["corpus"])
    top_heads = sorted(e["heads"].items(), key=lambda kv: kv[1]["ablated_p"])[:3]
    hrows = ""
    for k, v in top_heads:
        offs = ", ".join(f"offset −{t['off']} ({esc(t['tok'])!s}, {t['score']:+.2f})"
                         for t in v["top_offsets"])
        hrows += f"<tr><td>{k}</td><td>{v['ablated_p']:.2f}</td><td class='small' style='text-align:left'>{offs}</td></tr>"
    ng_html += f"""
<div class="circuit">
<div class="pathrow small" style="margin-bottom:8px">…{esc(e["prefix_text"])}</div>
<div class="pathrow">{lead}{toks}<span class="arrow">⟶</span>
<span class="tok tgt">{esc(e["target"])}</span>
<span class="small">&nbsp; P(target): attn2 <b>{e["p_attn2"]:.2f}</b> → attn3 <b>{e["p_attn3"]:.2f}</b></span></div>
<div class="tablewrap"><table>
<tr><th>train-corpus statistic</th><th>probability</th><th>context frequency</th></tr>{crows}</table></div>
<div class="tablewrap"><table>
<tr><th>most load-bearing heads (attn3)</th><th>ablate → P</th><th>top attention offsets from the prediction position</th></tr>{hrows}</table></div>
</div>"""

# ---------------- k-hop examples ----------------
khop_bind_str = "  ".join(f"{a}→{b}" for a, b in khop["bindings"][:14])
khop_ex_html = ('<p class="small">A concrete document (one random cycle) and three real queries '
                'from it — the answer is never stated next to the query; the model must chain '
                'lookups through the bindings:</p>'
                f'<div class="ex">bindings: {khop_bind_str} … '
                '<span class="ce">(continues to all 24 pairs, random order)</span></div>')
_sub = {1: "₁", 2: "₂", 3: "₃"}
for e in sorted(khop["examples"], key=lambda x: x["k"]):
    chain = e["chain"]
    steps = f"{chain[0]}"
    for i, c in enumerate(chain[1:], 1):
        hop = "1 hop" if i == 1 else f"{i} hops"
        node = f"<mark>{c}</mark>" if i == len(chain) - 1 else f"<b>{c}</b>"
        steps += f' →<span class="ce">({hop})</span> {node}'
    need = ("one lookup — induction-like" if e["k"] == 1
            else f'{e["k"]} chained lookups — the depth-gated part')
    khop_ex_html += (f'<div class="ex">query: [Q] {e["e"]} [H{_sub[e["k"]]}] → ?&nbsp;&nbsp; '
                     f'resolve: {steps} = answer '
                     f'<span class="ce">(k={e["k"]}: {need})</span></div>')

IND_Q = 39
IND_SRC = 13

html = f"""<title>Circuit Atlas — Bilinear Transformer Circuits</title>
<style>{css}</style>
<div class="col">

<div class="eyebrow">tensor_language · depth-differential program</div>
<h1>A Field Atlas of Bilinear Transformer Circuits</h1>
<p class="lede">What forms inside attention-only tensor networks as you add layers — found by
training depth ladders on natural text, tracking which validation tokens each extra layer
unlocks, and isolating every discovered circuit in a minimal toy task.</p>
<p class="meta">Bilinear attention (product of two dot products, no softmax) · RMSNorm · d=128, 4 heads,
ctx 256 · TinyStories (V=1024) &amp; OpenWebText (V=5120) · 2026-07-08</p>

<nav class="tabs" id="tabs">
<button data-t="method" class="on">Method</button>
<button data-t="induction">① Induction</button>
<button data-t="ngram">② N-gram</button>
<button data-t="khop">③ K-hop</button>
<button data-t="block3">④ Deep circuits</button>
<button data-t="axes">Two axes</button>
<button data-t="dyn">Formation dynamics</button>
</nav>

<section class="tab on" id="tab-method">
<h2>How circuits are found</h2>
<p>Every model in a depth ladder (1–4 attention layers, 3 seeds each, identical data order) is scored
on the same frozen validation stream, one cross-entropy value per token. A token is
<em>depth-d-gated</em> when the d-layer model learns it (median CE over seeds &lt; 0.5 nats) and the
(d−1)-layer model provably does not (median &gt; 1.5 nats — the hysteresis gap keeps threshold noise
out). Gated tokens are grouped by structure, the responsible heads are identified by ablation, and
each circuit is isolated in a synthetic task, following the training-dynamics methodology of
Singh et&nbsp;al. 2024. Seed medians matter: single-seed gates were 4× inflated by lottery noise.</p>
<div class="tablewrap"><table>
<tr><th>grid</th><th>data</th><th>configs</th><th>status</th></tr>
<tr><td>depth ladder</td><td>TinyStories, V=1024</td><td>attn1–4 × 3 seeds (+ dense replay, softmax archive)</td><td>complete</td></tr>
<tr><td>depth ladder v2</td><td>OpenWebText, V=5120</td><td>attn1–3, block2, d128/d256, 120k-step, mix03/mix10</td><td>active</td></tr>
<tr><td>Markov toy</td><td>order-k chains, V=64</td><td>attn1–4, block1–2 × k=1,2,3</td><td>complete</td></tr>
<tr><td>copy isolation</td><td>[u;u] iid copy</td><td>attn2 × V{{1024,5120}} × d{{128,256}}</td><td>complete</td></tr>
<tr><td>k-hop ladder</td><td>cycle bindings toy</td><td>attn2/3/4, attn·MLP·attn × 3 seeds</td><td>complete (prior)</td></tr>
</table></div>
</section>

<section class="tab" id="tab-induction">
<h2>Induction — the depth-2 circuit <span class="chip ok">recovered unsupervised</span></h2>
<p>136,740 TinyStories tokens (0.9% of the stream) are learnable by two attention layers and provably
out of reach for one; 64% complete a bigram seen earlier in the same window (11.7% base rate).
Below, the circuit on a real gated example — <em>“my j” must be completed with “e” because
“jewelry” appeared 27 tokens earlier</em>. Click heads to ablate them and watch the output.</p>

<div class="circuit flush wired" id="indcircuit">
<svg class="wire" id="wire"></svg>
<div class="stagelabel">Output distribution — top tokens (blue: full model; red: with the selected head ablated)</div>
<div class="bars" id="bars"><div class="barcol"><div class="barpair"><div class="bar" data-t="0" style="height:112px"></div><div class="bar abl" data-a="0" style="height:0px;display:none"></div></div><span class="barval" data-v="0">0.93</span><span class="barlab">e</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="1" style="height:2px"></div><div class="bar abl" data-a="1" style="height:0px;display:none"></div></div><span class="barval" data-v="1">0.02</span><span class="barlab">our</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="2" style="height:1px"></div><div class="bar abl" data-a="2" style="height:0px;display:none"></div></div><span class="barval" data-v="2">0.01</span><span class="barlab">ug</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="3" style="height:1px"></div><div class="bar abl" data-a="3" style="height:0px;display:none"></div></div><span class="barval" data-v="3">0.01</span><span class="barlab">ar</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="4" style="height:1px"></div><div class="bar abl" data-a="4" style="height:0px;display:none"></div></div><span class="barval" data-v="4">0.01</span><span class="barlab">ack</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="5" style="height:1px"></div><div class="bar abl" data-a="5" style="height:0px;display:none"></div></div><span class="barval" data-v="5">0.01</span><span class="barlab">u</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="6" style="height:1px"></div><div class="bar abl" data-a="6" style="height:0px;display:none"></div></div><span class="barval" data-v="6">0.01</span><span class="barlab">am</span></div><div class="barcol"><div class="barpair"><div class="bar" data-t="7" style="height:0px"></div><div class="bar abl" data-a="7" style="height:0px;display:none"></div></div><span class="barval" data-v="7">0.00</span><span class="barlab">ice</span></div></div>
<div class="delta" id="pdelta">full model — click a head (boxes or causal-map rows) to ablate it</div>
<div class="hrow">
<button class="hbox" id="hb-L1H2" data-h="L1H2">L1H2 — induction head
<span class="role">match at source, copy "e". causal map: acts at the LAST token (ΔP 0.80)</span></button>
<button class="hbox secondary" id="hb-L1H1" data-h="L1H1">L1H1 — secondary
<span class="role">parallel weaker path, also at the last token (ΔP 0.36)</span></button></div>
<div class="hrow"><button class="hbox" id="hb-L0H3" data-h="L0H3">L0H3 — previous-token head
<span class="role">acts at the SOURCE: writes " j" into "e"'s residual = L1H2's key (ΔP 0.15 there)</span></button></div>
<div class="stagelabel">Causal map — zero each head at each single token; color = drop in P(target).
Click a row to select that head.</div>
<canvas id="cmap" width="840" height="200" style="width:100%;border:1px solid var(--hair);border-radius:6px"></canvas>
<div class="stagelabel">Context — tokens underlined by the selected head's per-token causal effect</div>
<div class="ribbon" id="ribbon">{rib_html}</div>
<div class="legend"><span><span class="tok match" style="padding:0 3px">&nbsp;</span> earlier " j" (match)</span>
<span><span class="tok src" style="padding:0 3px">&nbsp;</span> source "e" (key+value)</span>
<span><span class="tok q" style="padding:0 3px">&nbsp;</span> prediction position " j"</span>
<span><span class="tok tgt" style="padding:0 3px">&nbsp;</span> target</span></div>
<div class="stagelabel" style="margin-top:14px">Attention pattern — combined bilinear score
(q₁·k₁)(q₂·k₂)/d², post-hadamard, causal-masked. Select a head above; click any context token to
re-anchor the query row (default: the head's working position). Blue = positive, red = negative.</div>
<div id="patrow" class="small" style="min-height:18px"></div>
<canvas id="patmap" width="840" height="960" style="width:100%;max-width:640px;border:1px solid var(--hair);border-radius:6px;display:none"></canvas>
<div class="stagelabel" style="margin-top:16px">Context heads — needed for the task, but not part of the match-and-copy wiring
(effects only when zeroed everywhere; zeroing them at src/q changes nothing)</div>
<div class="ctxheads">
<button class="headchip" data-h="L0H2">L0H2 <span class="hp">everywhere → 0.39</span></button>
<button class="headchip" data-h="L1H1">L1H1 <span class="hp">@q → 0.57</span></button>
<button class="headchip" data-h="L0H0">L0H0 <span class="hp">everywhere → 0.96</span></button>
<button class="headchip" data-h="L0H1">L0H1 <span class="hp">everywhere → 0.94</span></button>
<button class="headchip" data-h="L1H0">L1H0 <span class="hp">@q → 0.91</span></button>
<button class="headchip" data-h="L1H3">L1H3 <span class="hp">@q → 0.94</span></button>
</div>
</div>
<p class="small">The wiring is measured position-specifically: L0H3 matters only AT the source
(zeroing it there collapses L1H2's match weight −0.434 → −0.031 and P → 0.78; zeroing it at the
last token does nothing), and L1H2 matters only AT the last token (zero@q = zero-everywhere =
P 0.13). Solid arrows: reads; dashed: the write into the source's residual.</p>

<h3>Sign structure: pattern and OV agree (the XNOR view)</h3>
<p>Bilinear attention has no softmax, so pattern weights are signed — and a head's effect is the
<em>product</em> of its pattern weight and its OV-path contribution. A negative pattern is not
suppression if the OV path is also negative: the signs cancel. On this example L1H2 has pattern
−0.43 and OV −4.20 at the source — product <b>+1.83</b>, a strong vote <em>for</em> the correct
token. That agreement, not either sign alone, is the induction-head signature:</p>
<div class="tablewrap"><table>
<tr><th>head</th><th>pattern(q→src)</th><th>OV→logit(tgt)</th><th>product (this example)</th>
<th>mean product (2,898 positions)</th><th>sign-agree</th></tr>
{xnor_rows}
</table></div>
<p class="small">Aggregate columns: mean of pattern×OV over all induction positions in 100 val
windows. Only L1H2 carries consistent positive product — it is <em>the</em> induction head; both
of its factors happen to be negative, which single-sign readings misclassify.</p>


<h3>Weights → wiring: composition between layers, verified causally</h3>
<p>Multiplying the weight matrices directly (Elhage et al.'s composition, bilinear version —
here each L1 head reads through FOUR branches Q₁,K₁,Q₂,K₂ plus V). Raw Frobenius-norm
composition ‖W_K·OV_A‖/‖W_K‖‖OV_A‖ does <em>not</em> single out the causal edge; composed with
the matched token's embedding (what L0 actually writes when it reads " j"), both key branches of
L1H2 light up on L0H3 — and only that edge survives the causal test:</p>
<div class="tablewrap" style="display:flex;gap:22px;flex-wrap:wrap">
<div><div class="stagelabel">directional K₁ comp (input " j")</div><table class='comp'><tr><th></th><th>L1H0</th><th>L1H1</th><th>L1H2</th><th>L1H3</th></tr><tr><td>L0H0</td><td>0.186</td><td>0.082</td><td>0.112</td><td>0.141</td></tr><tr><td>L0H1</td><td>0.089</td><td>0.054</td><td>0.050</td><td>0.067</td></tr><tr><td>L0H2</td><td>0.057</td><td>0.074</td><td>0.072</td><td>0.057</td></tr><tr><td>L0H3</td><td>0.061</td><td>0.081</td><td class='big'>0.147</td><td>0.087</td></tr></table></div>
<div><div class="stagelabel">directional K₂ comp (input " j")</div><table class='comp'><tr><th></th><th>L1H0</th><th>L1H1</th><th>L1H2</th><th>L1H3</th></tr><tr><td>L0H0</td><td>0.052</td><td>0.130</td><td>0.044</td><td>0.044</td></tr><tr><td>L0H1</td><td>0.066</td><td>0.077</td><td>0.061</td><td>0.040</td></tr><tr><td>L0H2</td><td>0.058</td><td>0.062</td><td>0.045</td><td>0.057</td></tr><tr><td>L0H3</td><td>0.077</td><td>0.075</td><td class='big'>0.110</td><td>0.055</td></tr></table></div>
<div><div class="stagelabel">CAUSAL: L1 match weight q→src after zeroing L0 head at src (base -0.434 for L1H2)</div><table class='comp'><tr><th></th><th>L1H0</th><th>L1H1</th><th>L1H2</th><th>L1H3</th></tr><tr><td>L0H0</td><td>-0.009</td><td>+0.037</td><td>-0.410</td><td>-0.007</td></tr><tr><td>L0H1</td><td>-0.012</td><td>+0.030</td><td>-0.455</td><td>-0.003</td></tr><tr><td>L0H2</td><td>-0.011</td><td>+0.028</td><td>-0.439</td><td>-0.002</td></tr><tr><td>L0H3</td><td>-0.006</td><td>-0.004</td><td class='big'>-0.031</td><td>-0.006</td></tr></table></div>
</div>
<p class="small">Rows: L0 head zeroed / composed; columns: L1 head. The causal table is decisive:
only L0H3 carries L1H2's match (−0.434 → −0.031); zeroing any other L0 head leaves it intact.
Raw norm-composition (not shown) ranks L0H0→L1H2 higher — a caution: in bilinear attention,
weight products need the right input direction before they reflect the circuit.</p>
<h3>More gated examples — repeat visible in context</h3>
{ind_ex_html}

<h3>Head count is a sweet spot, not a redundancy dial</h3>
<p>Varying heads at fixed d=128 (3 seeds, median CE on the gated tokens): h1 <b>1.52</b> ·
h2 <b>0.64</b> · h4 <b>0.47</b> · h8 <b>1.56</b>. Too few heads and the complementary
circuit components can't form; too many and each head is too thin (d_head=16) for the bilinear
product match. Neither the softmax picture (more heads → redundant copies) nor "one big head
suffices."</p>
<h3>Formation is smooth; heads are not redundant</h3>
<p>On an exact dense-checkpoint replay, CE on the gated tokens falls 4.63 → 0.27 nats with no
plateau→cliff — the classic softmax phase change is absent. And unlike the additive, redundant
softmax induction heads of Singh et&nbsp;al., every knock-out here hurts and no solo head recovers
the task.</p>
<figure><img src="data:image/png;base64,{b64('induction_dynamics_attn2-seed0.png')}"
 alt="loss split, per-head scores, ablations">
<figcaption>Left: val CE on induction-predictable vs other tokens. Middle: per-head raw attention
score to the induction target (sign alone is not meaningful — see the XNOR table). Right:
knock-out vs all-but-one ablations.</figcaption></figure>
</section>

<section class="tab" id="tab-ngram">
<h2>Higher-order n-gram circuits — depth 3/4 <span class="chip ok">order-3 statistics</span></h2>
<p>The next rung is not more retrieval: depth-3-gated targets appear earlier in their window only at
base rate. They are word-internal completions whose answer lives in <em>corpus statistics over the
last 2–3 tokens</em>. Each example below shows the tokenized context, the train-corpus conditional
probabilities that prove its n-gram order, and the measured computation path (which heads gather
which offsets, and what ablating them does).</p>
{ng_html}
<div class="finding"><p><strong>The path, consistently:</strong> layer-0/1 heads attend to offsets
−1/−2 and pull those tokens' features into the current position; a layer-2 head reads the composed
local context (attending mostly to the current position itself) and emits the completion —
ablating that head alone sends P(target) to ≈0. So the circuit is “gather recent tokens → compose →
emit”: an n-gram table implemented across layers.</p></div>
<p>Note the second example (“ go”+“l” → “f”): its statistics are already order-2
(P=1.0 from two tokens), yet attn2 still fails it (P=0.01). Two bilinear attention layers don't
reliably <em>learn</em> even order-2 in-weights statistics — matching the Markov toy below — so
these gates reflect optimization, not just expressivity.</p>
<div class="tablewrap"><table>
<tr><th>gate set (tiny corpus)</th><th>P(tgt | 1 token)</th><th>P(tgt | 2 tokens)</th><th>P(tgt | 3 tokens)</th></tr>
<tr><td>random baseline</td><td>0.043</td><td>0.121</td><td>0.197</td></tr>
<tr><td>depth-2 gates (induction)</td><td>0.031</td><td>0.120</td><td>0.238</td></tr>
<tr><td>depth-3 gates</td><td>0.040</td><td class="hi">0.349</td><td class="hi">0.787</td></tr>
<tr><td>depth-4 gates</td><td>0.046</td><td>0.318</td><td class="hi">0.674</td></tr>
</table></div>
<p class="small">Median train-corpus conditional probability of gated targets (4,000-token samples).
Depth-2 gates are unpredictable at every order — pure context-readers; depth-3/4 gates are the
statistics family.</p>
</section>

<section class="tab" id="tab-khop">
<h2>Chained k-hop retrieval — pointer advance <span class="chip ok">reverse-engineered in toy</span></h2>
<p>From the toy program (results_hop.md): documents define a random cycle f over 24 entities via
binding pairs [e, f(e)]; queries ask for f<sup>k</sup>(e). Hop-2/3 are depth-gated at 3 attention
layers. The causal variables, bottom-up:</p>
{khop_ex_html}
<div class="circuit">
<div class="stagelabel">Output — answer accuracy (attn4 seed 0)</div>
<div class="pathrow"><span class="tok tgt">f³(e)</span><span class="small">top-1 accuracy <b>0.987</b> —
readable from the residual only after the last layer (output-head decode: 0.99 at L3, chance before)</span></div>
<div class="stagelabel">Layer 3 — advances the pointer twice more: f² (probe 0.77) and the answer f³ (0.98)</div>
<div class="stagelabel">Layer 2 — advances the pointer: f¹ = f(e) becomes linearly decodable (probe 0.62)</div>
<div class="stagelabel">Layer 1 — resolves the query entity: f⁰ = e (probe 1.00)</div>
<div class="stagelabel">Context</div>
<div class="pathrow"><span class="tok">e₁ f(e₁)</span><span class="tok">e₂ f(e₂)</span>
<span class="tok">…all 24 bindings…</span><span class="tok match">[Q]</span>
<span class="tok q">e</span><span class="tok match">[H₃]</span><span class="arrow">⟶ predict</span></div>
</div>
<div class="tablewrap"><table>
<tr><th>linear probe on residual (answer position)</th><th>f⁰ (query)</th><th>f¹</th><th>f²</th><th>f³ = answer</th></tr>
<tr><td>embedding</td><td>.04</td><td>.04</td><td>.04</td><td>.04</td></tr>
<tr><td>after layer 1</td><td class="hi">1.00</td><td>.05</td><td>.04</td><td>.04</td></tr>
<tr><td>after layer 2</td><td class="hi">1.00</td><td class="hi">0.62</td><td>.07</td><td>.05</td></tr>
<tr><td>after layer 3</td><td>.96</td><td>.49</td><td class="hi">0.77</td><td class="hi">0.98</td></tr>
</table></div>
<p>Each layer advances the entity pointer one hop, in a rotated basis the output head cannot read —
only f³ is assembled in the readable basis, at the last layer. The heaviest single head is a
layer-0 binding-substrate head (ablating it: hop-2 accuracy 0.93 → 0.09); layers 1–2 stack the
successive lookups. Bilinear MLPs cannot substitute for the attention layers (attn·MLP·attn fails
hop-2 with more parameters). Formation is a 1/3 seed lottery at any depth — failed seeds resolve
f⁰ but never advance the pointer — and an easy-first curriculum <em>backfires</em> (0/4 seeds).</p>
<p class="small">This circuit never surfaced as a gate on natural text at our scale: TinyStories and
OpenWebText don't demand chained retrieval. It remains the clean context-circuit exemplar above
induction.</p>
</section>


<section class="tab" id="tab-block3">
<h2>④ The block3 circuit family <span class="chip ok">5 circuit types, all adjudicated</span></h2>
<p>Protocol (one cycle, 2026-07-09): train block3 (6 layers, [attn+MLP]×3, 3 seeds) → STRICT
differential vs block2 (solved = p&gt;0.78 on seed medians; 17,422 gated tokens) → component-knockout
attribution fingerprints (12 heads + 3 MLPs) → k-means clusters → subagent labels + hypotheses →
targeted causal interventions. Verdicts:</p>
<div class="tablewrap"><table>
<tr><th>circuit</th><th>components</th><th>verdict / decisive measurement</th></tr>
<tr><td>MLP lexicon stack (C0–C3, 74%)</td><td>L1+L3+L5 MLP</td><td>one circuit split by k-means; order-2/3 statistics + BPE re-tokenization</td></tr>
<tr><td>specialized noun induction (C5)</td><td>L2H3</td><td>confirmed: signed-contribution offset median 32, 66% beyond 20 tokens, 28% source==target, 30% match+1 wiring</td></tr>
<tr><td>agreement / inflection (C6)</td><td>MLP stack jointly</td><td>confirmed: minimal pairs flip P("s") 0.85→0.02 on subject number (identical local suffix); zeroing any MLP destroys the gap; block2 unreliable</td></tr>
<tr><td>deep local read-off (C7)</td><td>L4H0 @ final token</td><td>confirmed: ΔP 0.78 zeroing at q only (=everywhere; control 0.10); contribution offset median 1 — reads the previous position's 2-block-deep residual. Retrieval hypothesis REFUTED (lemma overlap 11% ≈ control)</td></tr>
<tr><td>fuzzy name-context reader (C4)</td><td>L2H0</td><td>UNRESOLVED: long-range reads (35%&gt;20 tokens) without token anchoring (7% match+1) — the genuine feature-matching candidate</td></tr>
</table></div>
<h3>Examples per circuit (load-bearing component counts measured per token)</h3>
<div class="stagelabel">C5 — specialized induction (the story's introduced noun, re-completed later)</div>
<div class="ex">… soon he was at the top. He looked out over the world and it was so beautiful.⏎⏎The mouse was happy until suddenly he heard a noise! It was a harmless but ⟶ <b>ter</b> <span class="ce">(needs 5 components)</span></div><div class="ex">…ns and directions he had been given, and eventually he made it to the magical pie. He was so excited! He grabbed the p ⟶ <b>ie</b> <span class="ce">(needs 5 components)</span></div><div class="ex">… it to her. She says it is good for her skin. She decides to put some lotion on her doll.⏎⏎She opens the bottle and squeez ⟶ <b>es</b> <span class="ce">(needs 7 components)</span></div>
<div class="stagelabel">C6 — agreement/inflection</div>
<div class="ex">…lowly, Ricky began to move his feet and creep around the classroom on tip toe. He shuffled his feet ac ⟶ <b>ro</b> <span class="ce">(needs 5 components)</span></div><div class="ex">…in.⏎⏎One day, they sing a new song. It is about a frog and a fly. The frog wants to eat the fly, but the fly is too fast. The ⟶ <b> fly</b> <span class="ce">(needs 6 components)</span></div><div class="ex">…escaped from the bull.⏎⏎"Are you okay, Tom?" asked Sam. He was compassionate and kind. He hugged Tom and checked his bruis ⟶ <b>es</b> <span class="ce">(needs 6 components)</span></div>
<div class="stagelabel">C7 — deep local read-off</div>
<div class="ex">…'t sure what it was, so Mom told him a joke about it. "What did one wall say to the other wall?" she asked.⏎⏎Jim smiled but had no answer ⟶ <b>.</b> <span class="ce">(needs 5 components)</span></div><div class="ex">…ap for each of you," mom says. "But you have to be good and hold my hand."⏎⏎Anna and Ben are happy. They go to the shop with mom. They see many cap ⟶ <b>s</b> <span class="ce">(needs 6 components)</span></div><div class="ex">…'s apple. He says, "This is mine now. You are too small and weak." He laughs and walks away. Tom is sad and angry. He wants his a ⟶ <b>pp</b> <span class="ce">(needs 6 components)</span></div>
<div class="stagelabel">C4 — fuzzy name-context reader (unresolved)</div>
<div class="ex">… her forget her fear of heights.⏎⏎Susie flew so high that she eventually lost sight of her home. Suddenly, it started to rain heav ⟶ <b>ily</b> <span class="ce">(needs 5 components)</span></div><div class="ex">…. He asked his dad, "Can I roll the ball, please?" His dad said, "Sure, go ahead!" ⏎⏎Timmy rolled the ball all around the g ⟶ <b>y</b> <span class="ce">(needs 8 components)</span></div><div class="ex">… in the bathroom.Once upon a time, there was a little boy named Timmy. Timmy loved to play outside and look at the big billboard on the street. The b ⟶ <b>ill</b> <span class="ce">(needs 5 components)</span></div>
<p class="small">At depth, induction FRAGMENTS into token-type specialists while new families appear
(deep read-off; MLP-borne syntax). Full protocol + numbers: results_deeper.md session 2;
cluster_labels_block3.md has the subagent's full analysis with post-hoc verdicts.</p>
</section>

<section class="tab" id="tab-axes">
<h2>Two circuit families, two architectural axes <span class="chip ok">confirmed in toy + text</span></h2>
<p>Same corpus, three small models: <b>attn2</b> (two attention layers), <b>attn3</b> (three),
and <b>block1</b> (ONE attention layer + one bilinear MLP). Five real predictions:</p>
<div class="tablewrap"><table>
<tr><th>example (context ⟶ target)</th><th>family</th><th>attn2</th><th>attn3</th><th>block1</th></tr>
<tr><td class="mono">… makes bu ⟶ b (bubbles)</td><td>statistics</td><td class="lo">0.10</td><td class="hi">0.90</td><td class="hi">0.74</td></tr>
<tr><td class="mono">… loved playing gol ⟶ f</td><td>statistics</td><td class="lo">0.01</td><td class="hi">0.89</td><td class="hi">0.65</td></tr>
<tr><td class="mono">… a micro ⟶ s (microscope)</td><td>statistics</td><td class="lo">0.09</td><td class="hi">0.84</td><td class="hi">0.96</td></tr>
<tr><td class="mono">… set up his too ⟶ l</td><td>statistics</td><td class="lo">0.09</td><td class="hi">0.87</td><td>0.57</td></tr>
<tr><td class="mono">…jewelry… wear my j ⟶ e (copy)</td><td>context</td><td class="hi">0.93</td><td class="hi">0.84</td><td class="lo">0.45</td></tr>
</table></div>
<p>Read the columns: the four <em>statistics</em> examples (the answer is a common word-completion
in the corpus, NOT copied from the window) defeat two attention layers (P ≤ 0.10) but fall to a
single attention layer once a bilinear MLP sits on top of it — the MLP does the composition, so a
third attention layer was never the real requirement. The <em>context</em> example (the answer
must be copied from 27 tokens back) inverts the ranking: two attention layers excel, and block1's
single attention layer cannot complete the two-step match-and-copy (0.45). Two kinds of
datapoints, two different resources:</p>
<div class="axes">
<div class="cell"><div class="tag">context circuits</div>
<h4>copy · induction · k-hop</h4>
<p>Read the answer out of the context window. Need attention <em>layers</em> — cross-position
composition. Bilinear MLPs cannot substitute (attn·MLP·attn fails hop-2 with more parameters).
Form fast and smoothly in isolation; on rich data they lose to easier basins.</p></div>
<div class="cell"><div class="tag">statistics circuits</div>
<h4>n-gram · lexicon · word-form</h4>
<p>Compose the last few tokens' features against knowledge stored in weights. Need bilinear
<em>MLP</em> capacity — attention depth is a poor substitute, it only weakly emulates the MLP.
</p></div>
</div>
<p>The clean version of the same result on synthetic order-k Markov data (predict from a fixed
random table over the last k tokens — pure in-weights statistics, floor = exact entropy):</p>
<div class="tablewrap"><table>
<tr><th>model → gap to entropy floor</th><th>attn1</th><th>attn2</th><th>attn3</th><th>attn4</th><th>block1</th><th>block2</th></tr>
<tr><td>order-1 Markov</td><td>0.01</td><td>0.04</td><td>0.08</td><td>0.19</td><td>0.07</td><td>0.39</td></tr>
<tr><td>order-2 Markov</td><td class="lo">1.49</td><td>0.86</td><td>0.61</td><td>0.51</td><td class="hi">0.19</td><td>0.30</td></tr>
<tr><td>order-3 Markov</td><td>2.37</td><td>2.44</td><td>2.51</td><td>2.60</td><td>2.43</td><td>2.71</td></tr>
</table></div>
<p class="small">blockN = N × [bilinear attention + bilinear MLP]; gaps in nats at 12k steps.
One attn+MLP block beats four attention layers on order-2 by 2.7×. Deeper attention-only models
underfit even order-1, so read k=2 against each model's own k=1 baseline; the order-3 row is
capacity-bound for everything (262k-context table vs ~0.5M params) — it tests storage, not
composition. Overall val CE tells the same story: block1 1.97 &lt; attn3 2.07 &lt; attn2 2.14.</p>
</section>

<section class="tab" id="tab-dyn">
<h2>Basin competition decides which circuits form <span class="chip ok">lever confirmed</span></h2>
<p>On OpenWebText, plain d=128 bilinear models form <em>no</em> induction — 40k or 120k steps, the
attn1↔attn2 gap creeps to only 0.041 nats. Yet the circuit is provably available:</p>
<div class="tablewrap"><table>
<tr><th>pure copy task [u;u], floor = 0</th><th>V=1024, d=128</th><th>V=1024, d=256</th><th>V=5120, d=128</th><th>V=5120, d=256</th></tr>
<tr><td>bilinear attn2, final CE (nats)</td><td>0.0005</td><td>0.0000</td><td>0.0059</td><td>0.0444</td></tr>
</table></div>
<div class="finding"><p><strong>The failure is competition, not capability.</strong> Where
TinyStories' statistics saturate early (leaving gradient for the circuit), OpenWebText's statistics
keep paying — SGD never leaves that basin. Same phenomenon class as the k-hop seed lottery, now on
natural data.</p></div>
<h3>The lever: mix the circuit's pure form into the stream</h3>
<p>Retraining with 10% of batches replaced by copy-burst sequences (in-context structure, no
statistical shortcut) installs the circuit — and it <em>transfers</em> to natural text:</p>
<div class="tablewrap"><table>
<tr><th>natural induction tokens</th><th>step 10k</th><th>20k</th><th>30k</th><th>32.5k</th></tr>
<tr><td>mix10 ind-CE</td><td>3.857</td><td>3.459</td><td>3.233</td><td class="hi">3.214</td></tr>
<tr><td>plain ind-CE</td><td>3.711</td><td>3.530</td><td>3.449</td><td>3.438</td></tr>
<tr><td>mix10 induction score</td><td>0.027</td><td>0.086</td><td>0.137</td><td class="hi">0.143</td></tr>
<tr><td>plain induction score</td><td>0.015</td><td>0.012</td><td>0.012</td><td>0.013</td></tr>
</table></div>
<p>The circuit forms on the bursts at ~7–9k steps (5× slower than isolation — the basin's price),
then fires on natural induction tokens: a growing 0.22-nat advantage where the plain model stays
flat. Contrast with the k-hop curriculum, which backfired (0/4 seeds): ordering easy-first
entrenches a plateau; mixing in undiluted examples of the target computation installs it. Width is
a second dose knob: at d=256 the circuit forms slowly with no mixture at all (induction score
0.011 → 0.047 over 40k; 0.39-nat ind-token gap over its attn1), where d=128 shows nothing in 120k
steps. The dose-response is a steeply diverging <em>delay</em>: 10% mixture forms the circuit by
~9k steps (3/3 seeds), 5% by ~20–25k, and 3% not within 40k — formation time grows superlinearly
as the signal fraction drops, presenting as a threshold under any fixed budget.</p>
<h3>The scaffold can be temporary: formation is hysteretic</h3>
<p>Removing the mixture at 15k (after the circuit forms) collapses the <em>synthetic</em> copying
behavior within 2.5k steps — overshooting past chance, since pure OpenWebText anti-learns
iid-uniform copying — but the <em>natural</em> induction capability persists and keeps improving
without the scaffold: ind-CE 3.57 at the switch → 3.19 at 40k, better than the never-mixed model
(3.44) and equal to the always-mixed one. Natural data cannot create this circuit, but it
maintains it once created (7.4% of tokens exercise it). A temporary mixture phase installs a
permanent natural capability — and it clarifies scaffolding: it works when it installs machinery
the target distribution then rewards, and backfires (the k-hop curriculum, 0/4) when it entrenches
a competing basin.</p>
</section>

</div>
<script>
(function(){{
  var tabs = document.querySelectorAll('#tabs button');
  tabs.forEach(function(b){{ b.addEventListener('click', function(){{
    tabs.forEach(function(x){{ x.classList.remove('on'); }});
    document.querySelectorAll('section.tab').forEach(function(s){{ s.classList.remove('on'); }});
    b.classList.add('on');
    document.getElementById('tab-' + b.dataset.t).classList.add('on');
  }}); }});

  var IND = {{"base": 0.9328, "j": 12, "src": 13, "q": 39, "tokens": ["e", "our", "ug", "ar", "ack", "u", "am", "ice"], "dists": {{"full": [0.9328, 0.0158, 0.0096, 0.0092, 0.0083, 0.0067, 0.0065, 0.0023], "L0H0": [0.9567, 0.0061, 0.0089, 0.0127, 0.0042, 0.0014, 0.0003, 0.0007], "L0H1": [0.9419, 0.015, 0.0044, 0.0104, 0.0042, 0.0051, 0.0021, 0.0017], "L0H2": [0.3894, 0.1369, 0.107, 0.0825, 0.0799, 0.0207, 0.1087, 0.0018], "L0H3": [0.4447, 0.1204, 0.0551, 0.1029, 0.0363, 0.0514, 0.0047, 0.0293], "L1H0": [0.9103, 0.0214, 0.0121, 0.0099, 0.0134, 0.0073, 0.0085, 0.0037], "L1H1": [0.5743, 0.0161, 0.0657, 0.0192, 0.0059, 0.0648, 0.076, 0.0013], "L1H2": [0.1317, 0.2147, 0.0065, 0.0196, 0.0398, 0.0131, 0.0011, 0.0652], "L1H3": [0.9405, 0.0106, 0.0082, 0.0093, 0.0065, 0.0072, 0.0074, 0.0017]}}, "cmap": [[0.0014, -0.0013, 0.0019, 0.0001, 0.0008, 0.0, -0.0037, 0.003, -0.0002, 0.0033, 0.0005, -0.0027, 0.0005, -0.0098, 0.0012, 0.002, -0.0028, 0.0001, 0.0001, 0.0078, -0.0027, -0.0003, 0.0008, 0.0003, -0.0001, 0.0001, -0.0095, 0.002, -0.0173, 0.0002, -0.0005, -0.0119, -0.0051, 0.0003, 0.0062, -0.0062, 0.008, -0.012, 0.0028, 0.0234], [0.0001, -0.0, 0.0019, -0.0007, -0.002, 0.0022, 0.0009, -0.0001, -0.0016, 0.0001, -0.0001, -0.0015, 0.0, -0.0023, -0.0003, -0.0001, -0.0005, -0.0025, -0.0009, 0.0072, 0.0006, 0.0008, -0.0012, 0.0001, 0.0002, -0.0001, -0.0, 0.0003, 0.0, -0.0005, -0.0039, -0.0005, 0.0002, -0.0041, -0.0002, 0.0, 0.0026, -0.0022, -0.0038, 0.0045], [0.0, 0.0002, -0.0017, -0.0075, -0.0038, 0.0001, -0.0007, -0.0001, 0.0062, 0.0004, 0.0001, -0.0006, -0.0006, -0.0036, 0.0005, 0.0002, -0.0001, 0.0053, 0.0, -0.0003, -0.0, -0.0004, -0.0006, 0.0005, -0.0001, -0.008, 0.0001, 0.0017, -0.0018, -0.0054, 0.0001, -0.0016, 0.0001, -0.0039, 0.0009, 0.0031, -0.0005, 0.0015, 0.0003, -0.0051], [0.0002, -0.0001, 0.0003, 0.0014, -0.0026, -0.0019, 0.0002, -0.0003, -0.0, 0.0004, 0.0001, -0.0028, -0.0001, 0.1535, -0.0055, -0.0052, -0.0019, -0.0, -0.0013, -0.0059, 0.0001, 0.0001, -0.0069, 0.0003, 0.0008, 0.0005, 0.0016, 0.0005, 0.0002, 0.0008, -0.0038, 0.0, -0.0015, 0.0, 0.0003, 0.0003, 0.0012, -0.004, 0.0018, -0.0116], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0225], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3585], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8012], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0076]], "abl": {{"L1H2": 0.1317, "L1H1": 0.5743, "L0H3": 0.7794, "L0H2": 0.3894, "L0H0": 0.9567, "L0H1": 0.9419, "L1H0": 0.9103, "L1H3": 0.9405}}, "note": {{"L1H2": "L1H2 ablated everywhere \u2014 the primary match-and-copy is gone; bars show the full ablated output distribution", "L1H1": "L1H1 ablated everywhere \u2014 the secondary induction path (causal map: its effect is also at the last token)", "L0H3": "L0H3 ablated everywhere \u2014 without the previous-token write, L1H2's key never forms (match \u22120.434 \u2192 \u22120.031 when zeroed just at the source)", "L0H2": "L0H2 ablated everywhere \u2014 distributed context effect", "L0H0": "L0H0 minor", "L0H1": "L0H1 minor", "L1H0": "L1H0 minor", "L1H3": "L1H3 minor"}}}};
  var sel = null;
  var HEADS = ["L0H0","L0H1","L0H2","L0H3","L1H0","L1H1","L1H2","L1H3"];
  var boxes = document.querySelectorAll('#indcircuit .hbox, #indcircuit .ctxheads .headchip');
  function anchor(i){{ return document.querySelector('#indcircuit .ribbon .tok[data-i="' + i + '"]'); }}
  var toks = Array.prototype.slice.call(document.querySelectorAll('#indcircuit .ribbon .tok[data-i]'));
  function pos(el, rel){{ var r = el.getBoundingClientRect(), c = rel.getBoundingClientRect();
    return {{x: r.left - c.left + r.width/2, top: r.top - c.top, bot: r.top - c.top + r.height}}; }}
  function drawWires(){{
    var card = document.getElementById('indcircuit');
    var svg = document.getElementById('wire');
    svg.setAttribute('width', card.clientWidth); svg.setAttribute('height', card.clientHeight);
    svg.innerHTML = '';
    function path(x1,y1,x2,y2,dash,thin){{
      var m = (y1+y2)/2;
      var pp = document.createElementNS('http://www.w3.org/2000/svg','path');
      pp.setAttribute('d','M'+x1+','+y1+' C'+x1+','+m+' '+x2+','+m+' '+x2+','+y2);
      pp.setAttribute('stroke','#2f6fd0'); pp.setAttribute('fill','none');
      pp.setAttribute('stroke-width', thin ? '1' : '1.6'); pp.setAttribute('opacity','0.85');
      if(dash) pp.setAttribute('stroke-dasharray','5 4');
      svg.appendChild(pp);
    }}
    var jT = anchor(IND.j), sT = anchor(IND.src), qT = anchor(IND.q);
    var h0 = document.getElementById('hb-L0H3'), h1 = document.getElementById('hb-L1H2'),
        h1b = document.getElementById('hb-L1H1');
    var bar = document.getElementById('bars');
    if(!jT||!sT||!qT||!h0||!h1) return;
    var jc=pos(jT,card), sc=pos(sT,card), qc=pos(qT,card),
        h0c=pos(h0,card), h1c=pos(h1,card), h1bc=pos(h1b,card), bc=pos(bar,card);
    path(jc.x, jc.top, h0c.x-50, h0c.bot, false, false);
    path(h0c.x+50, h0c.bot, sc.x, sc.top, true, false);
    path(h0c.x, h0c.top, h1c.x+60, h1c.bot, false, false);
    path(qc.x, qc.top, h1c.x-60, h1c.bot, false, false);
    path(qc.x+8, qc.top, h1bc.x, h1bc.bot, false, true);
    path(h1c.x, h1c.top, bc.x-60, bc.bot, false, false);
    path(h1bc.x, h1bc.top, bc.x+60, bc.bot, false, true);
  }}
  var cm = document.getElementById('cmap');
  function drawCmap(){{
    var ctx = cm.getContext('2d');
    var n = IND.cmap[0].length, rows = 8;
    var L = 70, T = 6, W = cm.width - L - 8, H = cm.height - T - 26;
    var cw = W/n, ch = H/rows;
    var ink = getComputedStyle(document.body).color;
    ctx.clearRect(0,0,cm.width,cm.height);
    var m = 0; IND.cmap.forEach(function(r){{ r.forEach(function(v){{ m = Math.max(m, Math.abs(v)); }}); }});
    for (var r = 0; r < rows; r++){{
      for (var c = 0; c < n; c++){{
        var v = IND.cmap[r][c], a = Math.min(1, Math.abs(v)/m);
        ctx.fillStyle = v >= 0 ? 'rgba(179,55,47,' + a + ')' : 'rgba(47,111,208,' + a + ')';
        ctx.fillRect(L + c*cw, T + r*ch, Math.ceil(cw), Math.ceil(ch));
      }}
      ctx.fillStyle = (sel === HEADS[r]) ? '#e34948' : ink;
      ctx.font = '13px ui-monospace, Menlo, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(HEADS[r], L - 6, T + r*ch + ch/2);
    }}
    ctx.fillStyle = ink; ctx.textAlign = 'left'; ctx.font = '11px ui-monospace, Menlo, monospace';
    ctx.fillText('color: drop in P(target) when head zeroed at that token (max ' + m.toFixed(2) + '); columns = tokens in order', L, T + H + 14);
  }}
  cm.addEventListener('click', function(ev){{
    var r = cm.getBoundingClientRect();
    var y = (ev.clientY - r.top) * (cm.height / r.height) - 6;
    var row = Math.floor(y / ((cm.height - 32) / 8));
    if (row >= 0 && row < 8){{ sel = (sel === HEADS[row]) ? null : HEADS[row]; render(); }}
  }});
  function render(){{
    var d = document.getElementById('pdelta');
    d.textContent = sel ? IND.note[sel] : 'full model — click a head (boxes or causal-map rows) to ablate it';
    for (var i = 0; i < IND.tokens.length; i++){{
      var ab = document.querySelector('#bars .bar.abl[data-a="' + i + '"]');
      var bv = document.querySelector('#bars .barval[data-v="' + i + '"]');
      if (sel){{
        ab.style.display = 'block';
        ab.style.height = (IND.dists[sel][i]*120).toFixed(0) + 'px';
        bv.textContent = IND.dists.full[i].toFixed(2) + '→' + IND.dists[sel][i].toFixed(2);
      }} else {{ ab.style.display = 'none'; bv.textContent = IND.dists.full[i].toFixed(2); }}
    }}
    var ri = sel ? HEADS.indexOf(sel) : -1;
    var m = 0; if (ri >= 0) IND.cmap[ri].forEach(function(v){{ m = Math.max(m, Math.abs(v)); }});
    toks.forEach(function(t){{
      var i = +t.dataset.i;
      if (ri >= 0 && m > 0){{
        var v = IND.cmap[ri][i] || 0, a = Math.min(1, Math.abs(v)/m);
        t.style.borderBottom = '3px solid ' + (v >= 0 ? 'rgba(179,55,47,' + a + ')' : 'rgba(47,111,208,' + a + ')');
      }} else t.style.borderBottom = '';
    }});
    boxes.forEach(function(c){{ c.classList.toggle('sel', c.dataset.h === sel); }});
    drawCmap();
  }}
  boxes.forEach(function(c){{ c.addEventListener('click', function(){{
    sel = (sel === c.dataset.h) ? null : c.dataset.h; render();
  }}); }});
  window.addEventListener('resize', function(){{ drawWires(); drawCmap(); }});
  document.querySelector('#tabs button[data-t="induction"]').addEventListener('click', function(){{ setTimeout(function(){{ drawWires(); drawCmap(); }}, 60); }});
  setTimeout(function(){{ drawWires(); drawCmap(); }}, 60);

  render();
}})();
</script>
"""

PAT = json.loads((REPO / "runs_lm/circuit_demo_patterns.json").read_text())
pat_js = r"""
<script>
(function(){
  var PAT = __PATJSON__;
  var WORK = {"L1H2": IND_Q, "L1H1": IND_Q, "L1H0": IND_Q, "L1H3": IND_Q,
              "L0H3": IND_SRC, "L0H2": IND_SRC, "L0H0": IND_SRC, "L0H1": IND_SRC};
  var toks = Array.prototype.slice.call(document.querySelectorAll('#indcircuit .ribbon .tok[data-i]'));
  var canvas = document.getElementById('patmap');
  var rowinfo = document.getElementById('patrow');
  var curHead = null, curRow = null;
  function color(v, m){
    var a = Math.min(1, Math.abs(v) / (m || 1));
    return v >= 0 ? 'rgba(47,111,208,' + (a*0.9) + ')' : 'rgba(179,55,47,' + (a*0.9) + ')';
  }
  function maxAbs(M){ var m = 0; M.forEach(function(r){ r.forEach(function(v){ m = Math.max(m, Math.abs(v)); }); }); return m; }
  function paintRow(){
    toks.forEach(function(t){ t.style.background = ''; t.style.boxShadow = ''; });
    if (!curHead) { rowinfo.textContent = ''; canvas.style.display = 'none'; return; }
    var M = PAT[curHead], m = maxAbs(M), r = (curRow == null ? WORK[curHead] : curRow);
    toks.forEach(function(t){
      var i = +t.dataset.i;
      if (i <= r) t.style.background = color(M[r][i], m);
      if (i === r) t.style.boxShadow = '0 0 0 2px #555 inset';
    });
    rowinfo.textContent = curHead + ' attention FROM token #' + r +
      ' ("' + (toks[r] ? toks[r].textContent : '') + '") over everything before it — max |score| ' + m.toFixed(2);
    drawMap(M, m, r);
    canvas.style.display = 'block';
  }
  function lab(t){ return t.replace(/ /g, '\u00b7').replace(/\n/g, '\u23ce').slice(0, 8); }
  function drawMap(M, m, hi){
    var n = M.length, ctx = canvas.getContext('2d');
    var L = 120, T = 10, size = 690, cs = size / n;
    var mapBottom = T + size;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var ink = getComputedStyle(document.body).color || '#555';
    // cells
    for (var r = 0; r < n; r++) for (var c = 0; c <= r; c++){
      ctx.fillStyle = color(M[r][c], m);
      ctx.fillRect(L + c*cs, T + r*cs, Math.ceil(cs), Math.ceil(cs));
    }
    // query-row outline
    ctx.strokeStyle = ink; ctx.lineWidth = 2;
    ctx.strokeRect(L, T + hi*cs, size, cs);
    // y labels (query tokens), every row
    ctx.fillStyle = ink; ctx.textBaseline = 'middle';
    ctx.font = (cs * 0.72) + 'px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'right';
    for (var r = 0; r < n; r++)
      ctx.fillText(lab(toks[r].textContent), L - 6, T + r*cs + cs/2);
    // x labels (key tokens), rotated
    ctx.textAlign = 'right';
    for (var c = 0; c < n; c++){
      ctx.save();
      ctx.translate(L + c*cs + cs/2, mapBottom + 6);
      ctx.rotate(-Math.PI/3);
      ctx.fillText(lab(toks[c].textContent), 0, 0);
      ctx.restore();
    }
    // axis titles
    ctx.textAlign = 'left';
    ctx.font = '15px ui-monospace, Menlo, monospace';
    ctx.fillText('rows: query token (attends FROM)  \u00b7  cols: key token (attends TO)', L, mapBottom + 118);
    // colorbar
    var cbY = mapBottom + 140, cbH = 22, cbW = size;
    for (var i = 0; i < cbW; i++){
      var v = (i / cbW) * 2 - 1;
      ctx.fillStyle = color(v * m, m);
      ctx.fillRect(L + i, cbY, 1, cbH);
    }
    ctx.strokeStyle = ink; ctx.lineWidth = 1;
    ctx.strokeRect(L, cbY, cbW, cbH);
    ctx.textAlign = 'center'; ctx.font = '15px ui-monospace, Menlo, monospace';
    ctx.fillText('\u2212' + m.toFixed(2), L, cbY + cbH + 16);
    ctx.fillText('0', L + cbW/2, cbY + cbH + 16);
    ctx.fillText('+' + m.toFixed(2), L + cbW, cbY + cbH + 16);
    ctx.textAlign = 'left';
    ctx.fillText('combined bilinear score (q\u2081\u00b7k\u2081)(q\u2082\u00b7k\u2082)/d\u00b2', L, cbY - 8);
  }
  document.querySelectorAll('#indcircuit .hbox, #indcircuit .ctxheads .headchip').forEach(function(b){
    b.addEventListener('click', function(){
      curHead = (curHead === b.dataset.h) ? null : b.dataset.h; curRow = null; paintRow();
    });
  });
  toks.forEach(function(t){ t.addEventListener('click', function(){
    if (curHead){ curRow = +t.dataset.i; paintRow(); } }); });
})();
</script>
"""
pat_js = pat_js.replace("__PATJSON__", json.dumps(PAT)).replace("IND_Q", str(IND_Q)).replace("IND_SRC", str(IND_SRC))
html = html + pat_js

OUT.write_text(html)
print(f"wrote {OUT} ({len(html)/1024:.0f} KB)")
