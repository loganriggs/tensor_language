# make_loss_viewer: render optimal-ablation training curves (all components done so far)
# into loss_curves.html for the artifact. Re-run any time; embeds current results.
import json, time

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
r = json.load(open(PT + 'optimal_ablation_all_results.json'))
res = r['results']
clean = r.get('clean', 2.9455)
stamp = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())

rows = []
for name, v in res.items():
    if 'loss_curve' not in v:
        continue
    rows.append({'name': name, 'curve': v['loss_curve'],
                 'dm': v['delta_mean'], 'do': v['delta_opt'],
                 'ratio': v['opt_over_mean'], 'drift': v['rel_drift_from_mean']})
rows.sort(key=lambda x: x['ratio'])

data = json.dumps(rows)
html = """<title>Optimal-Ablation Curves</title>
<style>
:root { --bg:#faf9f6; --fg:#1c1e21; --muted:#6b7280; --card:#ffffff; --line:#2563eb;
  --accent:#b45309; --grid:#e5e7eb; --good:#047857; }
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#111418; --fg:#e5e7eb; --muted:#9ca3af; --card:#1a1f26; --line:#60a5fa;
  --accent:#f59e0b; --grid:#2a3038; --good:#34d399; } }
:root[data-theme="dark"] { --bg:#111418; --fg:#e5e7eb; --muted:#9ca3af; --card:#1a1f26;
  --line:#60a5fa; --accent:#f59e0b; --grid:#2a3038; --good:#34d399; }
body { background:var(--bg); color:var(--fg);
  font:14px/1.5 "IBM Plex Sans", system-ui, sans-serif; margin:0; padding:24px; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--muted); margin-bottom:6px; max-width:72ch; }
.budget { color:var(--muted); font-size:12.5px; margin-bottom:18px; max-width:78ch; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--grid); border-radius:6px;
  padding:10px 12px 6px; }
.hd { display:flex; justify-content:space-between; align-items:baseline; }
.nm { font-weight:600; font-size:13px; }
.ratio { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
.ratio.win { color:var(--good); font-weight:600; }
.meta { font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
svg { width:100%; height:64px; display:block; margin-top:4px; }
</style>
<h1>Optimal-Ablation Training Curves</h1>
<div class="sub">Per-component learned-constant ablation (Li &amp; Janson 2409.09951).
Each panel: per-step training CE (batch loss, Adam) from mean-init. Sorted by
&Delta;<sub>opt</sub>/&Delta;<sub>mean</sub> &mdash; biggest optimizer wins first.
Dashed line = mean-anchor held-out CE for that component.</div>
<div class="budget"><b>Data budget</b> &mdash; train: 480 FineWeb rows (T=256, skip=80),
batch 8 rows &times; 192 scored positions = 1,536 positions/step, 150 steps (MLPs, attn
layers) or 100 (heads) &asymp; 1.5&ndash;2 epochs over the pool. Validation: fully
held-out 960 rows (skip=7000), CE on positions &ge;64 = 184,320 paired positions for
both &Delta;<sub>mean</sub> and &Delta;<sub>opt</sub>. Clean CE = __CLEAN__.
Generated __STAMP__ &mdash; __N__ of 198 components; refreshed as the sweep runs.</div>
<div class="grid" id="g"></div>
<script>
const rows = __DATA__;
const clean = __CLEAN__;
const g = document.getElementById('g');
for (const r of rows) {
  const c = r.curve; const n = c.length;
  const meanCE = clean + r.dm;
  let lo = Math.min(...c, meanCE), hi = Math.max(...c, meanCE);
  if (hi - lo < 1e-6) { hi = lo + 1e-6; }
  const pad = 0.06 * (hi - lo); lo -= pad; hi += pad;
  const W = 220, Hh = 64;
  const x = i => (i / (n - 1)) * W;
  const y = v => Hh - ((v - lo) / (hi - lo)) * Hh;
  let pts = c.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const my = y(meanCE).toFixed(1);
  const win = r.ratio <= 0.9;
  const card = document.createElement('div'); card.className = 'card';
  card.innerHTML =
    `<div class="hd"><span class="nm">${r.name}</span>` +
    `<span class="ratio${win ? ' win' : ''}">opt/mean ${r.ratio.toFixed(3)}</span></div>` +
    `<div class="meta">&Delta;mean ${r.dm.toFixed(4)} &rarr; &Delta;opt ${r.do.toFixed(4)}` +
    ` &middot; drift ${r.drift.toFixed(3)}</div>` +
    `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="training loss curve ${r.name}">` +
    `<line x1="0" y1="${my}" x2="${W}" y2="${my}" stroke="var(--accent)"` +
    ` stroke-dasharray="4 3" stroke-width="1"/>` +
    `<polyline points="${pts}" fill="none" stroke="var(--line)" stroke-width="1.4"/></svg>`;
  g.appendChild(card);
}
</script>
"""
html = html.replace('__DATA__', data).replace('__CLEAN__', str(clean)) \
    .replace('__STAMP__', stamp).replace('__N__', str(len(rows)))
open(PT + 'loss_curves.html', 'w').write(html)
print(f"wrote loss_curves.html with {len(rows)} components")
