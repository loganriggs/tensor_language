"""Regenerate circuits.html from circuits/registry.json + records.
Static, self-contained; run after any batch of write_circuit calls."""
import json, os, html
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
reg=json.load(open(PT+'circuits/registry.json'))
cards=[]
rows=[]
for tag,meta in sorted(reg['circuits'].items()):
    c=json.load(open(PT+'circuits/'+meta['file']))
    ca=c.get('causal',{}); st=c.get('story',{})
    conc=(ca.get('abs_dce_members',0)/max(ca.get('abs_dce_offslice',1e-4),1e-4))
    rows.append(f"<tr><td><a href='#{tag}'>{tag}</a></td>"
        f"<td>{c.get('members',{}).get('n','')}</td>"
        f"<td class='n'>{conc:.1f}×</td>"
        f"<td class='n'>{ca.get('minority_share','')}</td>"
        f"<td>{html.escape(st.get('blind_name',''))}</td>"
        f"<td>{st.get('mechanism_level','')}</td>"
        f"<td class='n'>{st.get('program_bacc','')}</td></tr>")
    exs=c.get('examples',{})
    def exrow(e,k):
        ctx=html.escape(e.get('context','')).replace('\n','<span class=nl>⏎</span>')
        return (f"<tr><td class=k>{k}</td><td class=ctx>{ctx}</td>"
                f"<td class=ctx>{html.escape(repr(e.get('target','')))}</td>"
                f"<td class=n>{e.get('base_ce','')}</td><td class=n>{e.get('dce','')}</td></tr>")
    ex=''.join(exrow(e,'top') for e in exs.get('top',[]))+''.join(
        exrow(e,'rnd') for e in exs.get('random',[]))
    certs=''.join(f"<li>{html.escape(t.get('test',''))}: <b class={'ok' if t.get('verdict')=='HELD' else 'bad'}>"
                  f"{t.get('verdict')}</b> ({t.get('value')}) <span class=src>{t.get('source','')}</span></li>"
                  for t in c.get('certification',[]))
    tens=', '.join(f"{e['tag']} ({e['value']})" for e in c.get('relations',{}).get('tension',[])) or '—'
    cards.append(f"""<section id="{tag}"><h3>{tag} <span class=nm>{html.escape(st.get('blind_name',''))}</span></h3>
<p class=meta>components: <code>{html.escape('; '.join(str(x) for x in c.get('components',[])))}</code><br>
members {c.get('members',{}).get('n','?')} · |ΔCE| {ca.get('abs_dce_members','?')} vs {ca.get('abs_dce_offslice','?')} background ·
sign split +{ca.get('n_pos','?')}/−{ca.get('n_neg','?')} ({ca.get('dce_pos','?')}/{ca.get('dce_neg','?')}) ·
program <code>{html.escape(str(st.get('program','')))}</code> bacc {st.get('program_bacc','?')} null {st.get('program_null','?')} ·
tension: {tens}</p>
<ul class=cert>{certs}</ul>
<table><tr><th></th><th>context</th><th>next</th><th>base CE</th><th>ΔCE</th></tr>{ex}</table></section>""")
page=f"""<title>bilin18 Circuit Registry</title>
<style>
:root{{--ink:#0b0b0b;--sec:#52514e;--grid:#e1e0d9;--surface:#fcfcfb;--card:#f5f4f0;
--blue:#3987e5;--dblue:#104281;--red:#e34948;}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--ink:#e8e6e1;--sec:#b3b0a8;
--grid:#2b2a27;--surface:#161512;--card:#1e1d19;--blue:#5b9ce8;--dblue:#8db8ef;--red:#e8695f;}}}}
:root[data-theme="dark"]{{--ink:#e8e6e1;--sec:#b3b0a8;--grid:#2b2a27;--surface:#161512;
--card:#1e1d19;--blue:#5b9ce8;--dblue:#8db8ef;--red:#e8695f;}}
body{{background:var(--surface);color:var(--ink);font:15px/1.5 Georgia,serif;margin:0;padding:2rem 1rem 4rem;}}
main{{max-width:52rem;margin:0 auto;}}
h1{{font-size:1.7rem;margin:0 0 .2rem;}}
.sub{{color:var(--sec);margin:0 0 1.4rem;}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;}}
th{{text-align:left;color:var(--sec);border-bottom:1px solid var(--grid);padding:.35rem .5rem;
font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;}}
td{{padding:.35rem .5rem;border-bottom:1px solid var(--grid);vertical-align:top;}}
.n{{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Menlo,monospace;}}
.ctx,code{{font:12px ui-monospace,Menlo,monospace;}}
.nl{{color:var(--blue);}}
.k,.src,.meta{{color:var(--sec);font-size:.78rem;}}
.nm{{color:var(--sec);font-weight:normal;font-size:1rem;}}
.ok{{color:var(--dblue);}}.bad{{color:var(--red);}}
section{{background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:1rem;margin:1.2rem 0;}}
.tw{{overflow-x:auto;border:1px solid var(--grid);border-radius:6px;}}
ul.cert{{font-size:.82rem;padding-left:1.2rem;}}
a{{color:var(--blue);}}
</style>
<main><h1>bilin18 Circuit Registry</h1>
<p class=sub>{len(reg['circuits'])} recorded circuits · schema v1 · every number carries provenance;
examples are mechanically selected; FAILED verdicts stay on the record.</p>
<div class=tw><table><tr><th>tag</th><th>n</th><th>conc</th><th>minority</th><th>name</th><th>mech</th><th>prog bacc</th></tr>
{''.join(rows)}</table></div>
{''.join(cards)}</main>"""
open(PT+'circuits.html','w').write(page)
print(f'circuits.html: {len(reg["circuits"])} circuits')
