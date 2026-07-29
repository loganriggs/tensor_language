import json
P='/workspace/tensor_language/basis_aligned/qk_mdl/'
md=json.load(open(P+'tn_metric_dict_demo.json'));cb=json.load(open(P+'tn_circuit_bond_demo.json'))
fd=json.load(open(P+'tn_fold_demo.json'));rot=json.load(open(P+'tn_bond_rotation_demo.json'))
ga=json.load(open(P+'tn_gauge_demo.json'));toys=json.load(open(P+'tn_toys.json'));sc=json.load(open(P+'tn_sparsecode_demo.json'))

SEC=[('overview','Overview','the coverage map'),('arch','Architecture','bilinear layer &amp; folding'),
('l1','Lesson 1','the exact fold'),('l2','Lesson 2','the gauge trap'),('l3','Lesson 3','the right basis'),
('l4','Lesson 4','minimal circuits'),('l5','Lesson 5','bonds'),('l6','Lesson 6','what the bond carries'),
('l7','Lesson 7','two extremes')]
IDS=[s[0] for s in SEC]

CSS=open('/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/lesson3.html').read().split('</style>')[0].split('<style>')[1]
CSS='<style>'+CSS+'''
.layout{display:grid;grid-template-columns:236px 1fr;max-width:1240px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;border-right:1px solid var(--line);padding:20px 12px}
nav .brand{font-size:13px;font-weight:700;padding:0 10px 12px;color:var(--ink)}nav .brand span{display:block;font-size:10.5px;color:var(--dim);font-weight:500;letter-spacing:.1em;text-transform:uppercase;margin-top:3px}
nav a{display:block;padding:7px 10px;border-radius:8px;text-decoration:none;color:var(--dim);font-size:13.5px;cursor:pointer}
nav a b{color:var(--ink);font-weight:600;display:block}nav a small{font-size:11px;opacity:.85}
nav a.on{background:var(--grid)}nav a.on b{color:var(--acc)}nav a:hover{background:var(--grid)}
main{padding:clamp(20px,3.4vw,42px);max-width:1000px}
@media(max-width:820px){.layout{grid-template-columns:1fr}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:4px}nav .brand{width:100%}nav a small{display:none}}
.pager{display:flex;justify-content:space-between;margin:32px 0 8px;gap:10px}
.pager a{flex:1;padding:12px 15px;border:1px solid var(--line);border-radius:10px;text-decoration:none;color:var(--ink);font-size:13.5px;background:var(--panel);cursor:pointer}
.pager a small{display:block;font-size:11px;color:var(--dim)}.pager a.next{text-align:right}.pager a.dis{opacity:.3;pointer-events:none}
.hm{display:inline-block}.callout{background:var(--code);border:1px solid var(--warm);border-radius:11px;padding:15px 17px;margin:16px 0}.callout b{color:var(--warm)}.callout .q{font-size:15px;font-weight:600;color:var(--ink);margin-bottom:8px;display:block}
</style>'''

def chart(xs,series,w=430,h=170,xlab='k',vline=None,ymax=1.0):
    pl,pr,pt,pb=32,10,10,28;X=lambda i:pl+(i/(len(xs)-1))*(w-pl-pr);Y=lambda v:h-pb-(v/ymax)*(h-pt-pb);s=''
    for g in range(5):
        v=g/4*ymax;s+=f'<line x1="{pl}" y1="{Y(v):.1f}" x2="{w-pr}" y2="{Y(v):.1f}" stroke="var(--grid)"/><text x="{pl-4}" y="{Y(v)+3:.1f}" font-size="9" text-anchor="end" class="mono">{v:.1f}</text>'
    for i,x in enumerate(xs):
        if len(xs)<=13 or i%2==0:s+=f'<text x="{X(i):.1f}" y="{h-pb+13}" font-size="9" text-anchor="middle" class="mono">{x}</text>'
    s+=f'<text x="{(w+pl)/2:.0f}" y="{h-2}" font-size="10" text-anchor="middle">{xlab}</text>'
    if vline is not None and vline in xs:
        xi=xs.index(vline);s+=f'<line x1="{X(xi):.1f}" y1="{pt}" x2="{X(xi):.1f}" y2="{h-pb}" stroke="var(--warm)" stroke-dasharray="3 3"/>'
    for se in series:
        p=''.join((('L' if i else 'M')+f'{X(i):.1f} {Y(v):.1f} ') for i,v in enumerate(se['v']));s+=f'<path d="{p}" fill="none" stroke="{se["c"]}" stroke-width="2.2"/>'
        for i,v in enumerate(se['v']):s+=f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.3" fill="{se["c"]}"/>'
    return f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto">{s}</svg>'
def bars(pairs,w=430,h=64,mx=1.0):
    s=f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto">'
    for i,(lab,v,col) in enumerate(pairs):
        y=6+i*30;bw=v/mx*(w-190);s+=f'<text x="0" y="{y+13}" font-size="12.5">{lab}</text><rect x="150" y="{y+2}" width="{max(bw,2):.0f}" height="14" rx="3" fill="{col}"/><text x="{150+max(bw,2)+6:.0f}" y="{y+13}" font-size="12" class="mono" style="fill:var(--dim)">{v:.2f}</text>'
    return s+'</svg>'
def heat(Mat,cell=13):
    R=len(Mat);C=len(Mat[0]);mx=max(abs(v) for row in Mat for v in row) or 1
    s=f'<svg viewBox="0 0 {C*cell} {R*cell}" style="width:{C*cell}px;max-width:100%;height:auto;border-radius:3px">'
    for i in range(R):
        for j in range(C):
            v=Mat[i][j]/mx;col='var(--acc)' if v>=0 else 'var(--bad)';s+=f'<rect x="{j*cell}" y="{i*cell}" width="{cell}" height="{cell}" fill="{col}" opacity="{abs(v):.2f}"/>'
    return s+'</svg>'
def spec(sv,col='var(--acc)',w=200,h=56):
    n=len(sv);mx=max(sv) or 1;s=f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;height:auto">'
    for i,v in enumerate(sv):
        bh=v/mx*(h-8);s+=f'<rect x="{4+i*(w-8)/n:.1f}" y="{h-4-bh:.1f}" width="{(w-8)/n-1.5:.1f}" height="{max(bh,0.5):.1f}" fill="{col}" rx="1"/>'
    return s+'</svg>'
def act(vec,w=420,h=76):
    n=len(vec);mx=max(abs(x) for x in vec) or 1;s=f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto"><line x1="0" y1="{h/2}" x2="{w}" y2="{h/2}" stroke="var(--grid)"/>'
    for i,v in enumerate(vec):
        bw=(w-8)/n;bh=v/mx*(h/2-6);s+=f'<rect x="{4+i*bw:.1f}" y="{h/2-bh if v>=0 else h/2:.1f}" width="{max(bw-2,1):.1f}" height="{abs(bh):.1f}" rx="1" fill="{"var(--acc)" if v>=0 else "var(--bad)"}"/>'
    return s+'</svg>'
def pager(id):
    i=IDS.index(id);prev=SEC[i-1] if i>0 else None;nxt=SEC[i+1] if i<len(SEC)-1 else None
    p=f'<a class="prev{" dis" if not prev else ""}" onclick="go(\'{prev[0] if prev else ""}\')"><small>← previous</small>{prev[1] if prev else ""}</a>'
    n=f'<a class="next{" dis" if not nxt else ""}" onclick="go(\'{nxt[0] if nxt else ""}\')"><small>next →</small>{nxt[1] if nxt else ""}</a>'
    return f'<div class="pager">{p}{n}</div>'
def hdr(k,h,l):return f'<div class="kicker">{k}</div><h1>{h}</h1><p class="lede">{l}</p>'
def S(id,inner):return f'<section id="s-{id}" class="lesson" style="display:none">{inner}{pager(id)}</section>'

# ===== OVERVIEW =====
strip='<svg viewBox="0 0 900 120" style="width:100%;height:auto">'
for i in range(18):
    x=20+i*48;strip+=f'<rect x="{x}" y="50" width="42" height="26" rx="3" fill="var(--grid)"/><text x="{x+21}" y="67" font-size="9" text-anchor="middle" class="mono">{i}</text>'
for a,b,c,lab,ty in [(0,1,'var(--good)','L0–1 taken apart',38),(3,8,'var(--acc)','L3–8 memory routing',95),(5,5,'var(--node)','L5 induction',22),(13,17,'var(--warm)','L13–17 top MLPs',108)]:
    x0=20+a*48;x1=20+b*48+42;strip+=f'<rect x="{x0}" y="50" width="{x1-x0}" height="26" rx="3" fill="{c}" opacity="0.45"/><text x="{(x0+x1)/2:.0f}" y="{ty}" font-size="10" text-anchor="middle" style="fill:{c};font-weight:600">{lab}</text>'
strip+='</svg>'
OV=S('overview',hdr('A course in tensor-network interpretability','What we took apart, and how',
'A decomposition of <b>bilin18</b> — a 546-million-parameter transformer with <b>no softmax</b>: attention is (q1·k1)(q2·k2), MLP is Down(L·x ⊙ R·x), so every layer is exactly multilinear. That lets us fold each layer into a tensor and take it apart. Here is the map; the lessons are the method. Use the sidebar, or ← → to move.')+f'''
<div class="card"><h2>The coverage map</h2><div class="sub">Understood deeply at the front, partially deeper in.</div>{strip}
<table class="nums" style="margin-top:10px"><tr><th>region</th><th>what we found</th><th>headline</th></tr>
<tr><td><b>Layer 0</b></td><td>QK folds to a sparse dictionary of named atoms; block-0 MLP dense but its output bond thin</td><td class="mono">+0.006 @ 6.1% bits; bond rank ~10</td></tr>
<tr><td><b>Layer 1</b></td><td>whole attention pattern ~99% token-identity; 9/9 heads validated</td><td class="mono">static tables +0.027; 27× L0's load</td></tr>
<tr><td><b>Layers 3–8</b></td><td>query-side routing of a first-mention entity (memory pipeline)</td><td class="mono">band restore 0.90; no layer &gt;0.26</td></tr>
<tr><td><b>Layer 5</b></td><td>a textbook induction head (match+copy) + a rank-1 gain head</td><td class="mono">named, causally traced</td></tr>
<tr><td><b>Layers 13–17</b></td><td>top MLPs pull context in; mlp16 = token mean + ~16 register gains</td><td class="mono">rank-16 sufficient, +0.024</td></tr>
<tr><td><b>Whole stack</b></td><td>every read ≈ embedding + token tables + a live 6-layer window</td><td class="mono">all reads windowed +0.059</td></tr></table>
<div class="verdict">Honest scope: layers 0–1 in full; deeper layers <em>to a degree</em> (routing, induction/gain heads, top-MLP register), not exhaustively; model competent to ~512 tokens (audits frozen there).</div></div>
<div class="card"><h2>The method, in two questions</h2><div class="sub">A network is a tensor network: <b>nodes</b> compute, <b>bonds</b> carry information.</div>
<div class="two"><div class="callout" style="border-color:var(--node)"><b style="color:var(--node)">Is the node decomposable?</b><br>Rewrite the tensor as a few real features, in a basis that isn't arbitrary? (Lessons 1–4)</div>
<div class="callout"><b>Is the bond sparse?</b><br>How much flows on the wire — few dimensions, few symbols, a typed blob? (Lessons 5–6)</div></div></div>''')

# ===== ARCHITECTURE =====
AR=S('arch',hdr('Architecture','The bilinear layer, and what "folding" means',
'Before the method, the object. Everything rests on one choice: the model multiplies instead of thresholding. Start with a single unit, then the whole layer, then why that makes it foldable.')+'''
<div class="card"><h2>Start simple: one bilinear unit is a product of two knobs</h2>
<div class="sub">A normal neuron takes one weighted sum and thresholds it. A <b>bilinear</b> unit takes <b>two</b> weighted sums of the same input and <b>multiplies</b> them — no threshold. That product is a soft AND: large only when both reads are large, and it can flip sign.</div>
<pre class="code"><span class="c"># one bilinear unit: two linear reads, multiplied</span>
score_L = l · x          <span class="c"># "is it about music?"</span>
score_R = r · x          <span class="c"># "is it a plural noun?"</span>
out     = score_L * score_R    <span class="c"># fires on music-AND-plural — a soft conjunction</span></pre>
<div class="verdict">The whole trick: a product of two linear reads is a <b>degree-2 polynomial</b> in the input. Polynomials, unlike thresholds, can be rewritten in closed form. Hold onto that.</div></div>
<div class="card"><h2>The two real layers of bilin18</h2><div class="two">
<div><div class="sub" style="margin:0 0 6px"><b>Bilinear attention</b> — how tokens talk</div>
<pre class="code"><span class="c"># two score circuits, multiplied. no softmax.</span>
pattern = (q1·k1)(q2·k2) / d²   <span class="c"># attend when BOTH agree</span>
out_i   = Σ_j pattern[i,j]·(W_o W_v·x_j)</pre>
<div class="sub">Two query/key circuits multiplied (a conjunction), then the value/output path delivers the payload. No exponential, no normalization.</div></div>
<div><div class="sub" style="margin:0 0 6px"><b>Bilinear MLP</b> — how a token thinks</div>
<pre class="code"><span class="c"># two linear reads of the residual, multiplied</span>
h   = (L·x) ⊙ (R·x)    <span class="c"># elementwise, 4608 units</span>
out = Down · h</pre>
<div class="sub">Left and Right read the residual, multiply elementwise, Down writes it back — the single unit, stacked 4,608 wide.</div></div></div>
<div class="verdict"><b>Every layer is multilinear</b> (degree 2–3), except RMS-norm, which at layer 0 is a fixed per-token rescale. That property is what the whole course exploits.</div></div>
<div class="card"><h2>What "folding" and "decomposition" mean</h2>
<div class="sub"><b>Folding</b> = collapsing a layer's weight matrices into the one interaction tensor they secretly are, exactly, by algebra (Lesson 1). <b>Decomposition</b> = rewriting that tensor in a basis where it becomes a few interpretable pieces, if it can (Lessons 2–4); reading the thin channels between layers when it can't (Lessons 5–6).</div>
<pre class="code">weights (L, R, Down)  ──fold──▶  one tensor T[o,i,j]  ──decompose──▶  a few named features
                                  (exact · Lesson 1)     (metric dictionary · Lesson 3)</pre>
<div class="verdict">The plan: <b>fold</b> every layer to a tensor, <b>decompose</b> the ones that will, <b>measure the bonds</b> of the ones that won't. That is the rest of the course.</div></div>''')
print('overview+arch ok')
open(P+'_cp.json','w').write(json.dumps({'OV':OV,'AR':AR}))

SC='/tmp/claude-0/-workspace-tensor-language/a6c5fb86-7bce-48e6-bb32-8679e85cbf66/scratchpad/'
def bodyof(f):
    h=open(SC+f).read();return h.split('<div class="wrap">',1)[1].rsplit('</div>',1)[0]
L1=S('l1',bodyof('lesson1.html'));L3=S('l3',bodyof('lesson3.html'));L4=S('l4',bodyof('lesson4.html'));L5=S('l5',bodyof('lesson5.html'))

# ===== L2 gauge (revised) =====
g=ga
L2=S('l2',hdr('Lesson 2','The gauge trap',
'The layer is folded (Lesson 1). It is tempting to read its <b>neurons</b> — "neuron 5 fires for X". But the neuron basis is a <b>choice of coordinates</b>, not a fact: you can change every neuron\'s activation without changing what the layer does. Only the folded tensor is real — and even part of that is invisible.')+f'''
<div class="card"><h2>Same function, totally different neurons</h2>
<div class="sub">One bilinear layer, one input, drawn two ways. On the right every neuron is rescaled by an arbitrary amount (compensated downstream). Activations move — one swings from about {g['sample_activations_original'][4]} to {g['sample_activations_regauged'][4]} — but the output is <b>identical to {g['gauge_output_maxerr']:.0e}</b>.</div>
<div class="two"><div><div class="sub" style="margin:0 0 5px">a neuron reading</div>{act(g['sample_activations_original'])}</div>
<div><div class="sub" style="margin:0 0 5px"><b style="color:var(--warm)">re-gauged</b> — same layer</div>{act(g['sample_activations_regauged'])}</div></div>
<div class="verdict">Permuting or rescaling neurons leaves the folded tensor unchanged to machine precision. So <b>"neuron 5 does X" reads the coordinates, not the layer.</b></div></div>
<div class="card"><h2>Even the tensor has a hidden half</h2>
<div class="sub">The tensor takes the input on <em>both</em> legs, so only its symmetric part can affect the output; the antisymmetric part cancels. This layer's tensor is <b>{int(g['antisymmetric_fraction']*100)}% antisymmetric</b>, yet computing from the symmetric part alone changes the output by {g['symmetric_only_output_maxerr']:.0e} — a whole chunk that is behaviorally invisible.</div></div>
<div class="card real"><h2>The real thing — bilin18</h2>
<div class="sub">Permuting the neurons of a real bilinear MLP leaves the folded tensor identical to <b>2.6×10⁻⁷</b>, and <b>~65% of the raw tensor is antisymmetric gauge</b> — the same phenomenon the toy shows. That is why per-neuron "circuit" stories can be reading noise. There is also a specific attention gauge: the two query/key branches can be <b>swapped and reciprocally rescaled</b> (q1·k1)(q2·k2) is symmetric in them — so we canonicalize (‖W1‖=‖W2‖, fixed order) before any analysis. Findings live only in gauge-invariant quantities — which is why the next lesson goes looking for a basis that is real.</div></div>''')

# ===== L6 what the bond carries (revised) =====
L6=S('l6',hdr('Lesson 6','What the bond carries',
'Lesson 5 counted how many dimensions flow. But a bond can be <b>wide and still sparse</b> — width and sparsity are different axes. A thin channel takes one of three shapes, and each is legible even when the node isn\'t.')+f'''
<div class="card"><h2>1 · A sparse code — few active symbols of a large dictionary</h2>
<div class="sub">The wire holds a dictionary of {sc['M_symbols']} named symbols, but each input lights only a handful. Behavior saturates once ~{sc['knee']} are allowed. This is <em>not</em> the same as narrow: the wire is {sc['M_symbols']} wide but {sc['knee']}-sparse.</div>
{chart(sc['ks_active'],[{'v':sc['r2'],'c':'var(--good)'}],440,170,'symbols allowed active',vline=sc['knee'])}
<div class="verdict">You read the channel as "which few symbols fired" — the way we read attention as a small dictionary of topics and morphology.</div></div>
<div class="card"><h2>2 · A typed blob — an opaque payload, routed unopened</h2>
<div class="sub">Sometimes the payload is dense and illegible, but the channel is still understood because the payload is <b>typed and routed</b>: carried through intermediate nodes that address it without reading it, opened only at the end. You understand the wire by its routing protocol, not its bytes.</div></div>
<div class="card real"><h2>The real thing — bilin18</h2>
<div class="sub"><b>Sparse code beats narrow at equal bits.</b> Layer-0 attention: a dictionary of n=1024 symbols, k=8 active, costs <b>+0.006 nats</b> at 6.1% of the raw bits — versus SVD rank-16 (a narrow dense bond) at <b>+0.035</b>, 6× worse. Wide-and-sparse wins. The symbols are nameable (music, film, plural-suffix, {{the}}), and per head the inventory spans 32 to 4,096 symbols — channel vocabulary is a per-component property.</div>
<div class="sub" style="margin-top:12px"><b>The typed blob — the memory pipeline.</b> A first-mention entity ("Lindsay → Lohan") is carried as a payload through four verified stages: front MLPs <b>write</b> it into the key's residual (block 1 largest, recovery 0.50; context-bound — transplant fails 0.04 vs 1.00); shallow attention layers 3–8 <b>route</b> it (band 0.90, no single layer &gt;0.26 — moved, unopened); it is <b>fetched late</b> (corrupting layers 13–17 of the key destroys 0.99 of the prediction); then read out. The channel is legible (four causal stages); the payload contents are not.</div>
<div class="verdict">Honest limit: one shared code for <em>every</em> wire fails — a single 512-symbol dictionary coding all bonds costs +0.59 nats. Channels have per-bond vocabularies; there is no universal code.</div></div>''')

# ===== L7 two extremes (revised) =====
tA=toys['toyA_decomposable'];tB=toys['toyB_dense_thinbond']
L7=S('l7',hdr('Lesson 7','Two ways to understand a computation',
'The thesis. For every piece you ask two questions — <b>is the node decomposable?</b> and <b>is the bond sparse?</b> — and a computation is legible if <em>either</em> answer is favorable. A decomposable node you understand from the inside; a dense node with thin bonds you understand entirely from its interface.')+f'''
<div class="card"><div class="two">
<div><h2 style="color:var(--acc);font-size:16px">Extreme A · decomposable node</h2><div class="sub" style="margin:2px 0 8px">toy: (a+b) mod 23, its {tA['hidden_units']} units are ~{tA['effective_num_frequencies']} circular features</div>
{chart(tA['rank_k_ks'],[{'v':tA['ranked_acc'],'c':'var(--good)'},{'v':tA['random_acc'],'c':'var(--bad)'}],400,150,'features kept')}
<div class="verdict">chosen ≫ random, plateaus fast → a few privileged features → you name them.</div></div>
<div><h2 style="color:var(--warm);font-size:16px">Extreme B · dense node, thin bonds</h2><div class="sub" style="margin:2px 0 8px">toy: one node (eff-rank {tB['node_effective_rank']}/{tB['node_units']}) feeding {tB['K_consumers']} rank-2 consumers</div>
{bars([('node eff-rank',tB['node_effective_rank'],'var(--warm)'),('per-bond eff-rank',tB['per_consumer_bond_eff_rank'],'var(--good)')],400,70,tB['node_units'])}
<div class="verdict">no few units suffice (~48 of 64 needed), but every channel out is rank ~2 → understand it by the sparse communication, not the node.</div></div>
</div><div class="legend"><span><span class="sw" style="background:var(--good)"></span>chosen / bond</span><span><span class="sw" style="background:var(--bad)"></span>random / node</span></div></div>
<div class="card real"><h2>The real thing — two MLPs of bilin18</h2><div class="two">
<div><div class="sub" style="margin:0 0 6px"><b style="color:var(--acc)">mlp16 — Extreme A.</b> token mean + a few live gains:</div>
<table class="nums"><tr><th>kept</th><th>ΔCE</th></tr><tr><td>mean only</td><td class="mono">+0.141</td></tr><tr><td>+ rank-4</td><td class="mono">+0.040</td></tr><tr><td><b>+ rank-16</b></td><td class="mono" style="color:var(--good)">+0.024</td></tr></table>
<div class="sub" style="margin-top:6px">the ~16 directions decode to register: legal (40%), prose (18%), markup (5%); each an exact weight quadratic (gate 8e-7); causally sufficient at 64 whitened features (R² 0.954).</div></div>
<div><div class="sub" style="margin:0 0 6px"><b style="color:var(--warm)">block-0 MLP — Extreme B.</b> dense inside, thin bond out:</div>
<table class="nums"><tr><th>channel rank</th><th></th></tr><tr><td>in weights</td><td class="mono">68 of 128</td></tr><tr><td>realized on text</td><td class="mono" style="color:var(--good)">10</td></tr><tr><td>priced at rank-16</td><td class="mono">+0.0113 (78%)</td></tr></table>
<div class="sub" style="margin-top:6px">removing it costs +2.50 nats; its own decomposition ties the random null (0.483 vs 0.485) — no privileged basis. Read the bond, not the node.</div></div></div>
<div class="verdict"><b>The scorecard.</b> mlp16 → node <b>decomposable</b> (rank-16, named). block-0 → node <b>not</b> (distributed), bond <b>thin</b> (rank ~10, 78% at rank-16). Both are understood — by different answers to the same two questions. That is the thesis the whole course builds to.</div></div>''')

# ===== ASSEMBLE =====
navlinks=''.join(f'<a id="nav-{i}" onclick="go(\'{i}\')"><b>{t}</b><small>{d}</small></a>' for i,t,d in SEC)
allsec=OV+AR+L1+L2+L3+L4+L5+L6+L7
page=f'''<title>Tensor-network interpretability — a course</title>{CSS}<div class="layout">
<nav><div class="brand">Tensor networks<span>interpretability course</span></div>{navlinks}</nav>
<main>{allsec}</main></div>
<script>
const IDS={json.dumps(IDS)};
function go(id){{if(!id)return;for(const x of IDS){{document.getElementById('s-'+x).style.display=(x===id)?'block':'none';document.getElementById('nav-'+x).classList.toggle('on',x===id);}}
history.replaceState(null,'','#'+id);window.scrollTo(0,0);}}
document.addEventListener('keydown',e=>{{const i=IDS.indexOf(location.hash.slice(1)||'overview');if(e.key==='ArrowRight'&&i<IDS.length-1)go(IDS[i+1]);if(e.key==='ArrowLeft'&&i>0)go(IDS[i-1]);}});
go((location.hash.slice(1)&&IDS.includes(location.hash.slice(1)))?location.hash.slice(1):'overview');
</script>'''
open(SC+'tn_course.html','w').write(page)
print('COURSE BUILT',len(page),'bytes')
