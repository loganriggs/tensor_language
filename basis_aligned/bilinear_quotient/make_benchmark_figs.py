"""Benchmark figures for the compression program (CPU, matplotlib).
Figure 1: whole-model compression-fidelity frontier (components replaced
vs nats lost), all assembly versions, Pareto front highlighted, ceiling =
wte-only model. Figure 2: per-module CE relevance (mean-ablation cost),
ranked, with the in-assembly stand-in marginal overlaid where measured."""
import json, sys
sys.path.insert(0,'/workspace/tensor_language')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from palette import INK, SECONDARY, MUTED, GRID, SURFACE
BLUE='#3987e5'; DBLUE='#104281'; RED='#e34948'
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
mr=json.load(open(PT+'module_relevance_results.json'))
base=mr['base']; ceiling=mr['ceiling']; span=ceiling-base

# ---- Figure 1: whole-model frontier ----
# (n components replaced, nats lost, label, frontier?)
versions=[
 (13,1.85,'v1 naive stack',0),(13,2.69,'v2',0),(13,1.97,'',0),
 (13,1.39,'v4 front tables\n+absorbers',1),
 (19,2.10,'v5',0),(19,1.99,'',0),(19,1.90,'',0),
 (19,1.68,'v7 weights-only\nmiddle',1),
 (27,2.55,'v6',0),(27,2.21,'v8 merged',1),
 (35,3.27,'v9 all rungs',0),(34,2.87,'v9-best',0),
 (34,2.85,'',0),(34,2.78,'v11: greedy mean-swaps\n(5 rungs -> one vector)',1),
]
fig,ax=plt.subplots(figsize=(8.6,5.4),facecolor=SURFACE)
ax.set_facecolor(SURFACE)
front=sorted([(n,c,l) for n,c,l,f in versions if f])
fx=[0]+[n for n,_,_ in front]; fy=[0]+[c for _,c,_ in front]
ax.plot(fx,fy,'--',color=MUTED,lw=1.6,zorder=2,
        label='standard eval window (optimistic, ledger 22)')
FRESH=[(0,0),(13,1.98),(20,1.46),(24,2.29),(28,2.84),(34,2.93)]
FRESH=sorted(FRESH)
ax.plot([p_[0] for p_ in FRESH],[p_[1] for p_ in FRESH],'-',color=BLUE,
        lw=2.2,zorder=3,label='fresh never-seen documents (honest)')
for n,c in FRESH[1:]:
    ax.scatter([n],[c],s=50,color=DBLUE,zorder=5)
ax.legend(fontsize=8.5,frameon=False,loc='center right')
for n,c,l,f in versions:
    if f:
        ax.scatter([n],[c],s=54,color=DBLUE,zorder=4)
        ax.annotate(l,(n,c),textcoords='offset points',xytext=(8,-14),
                    fontsize=8.5,color=INK)
    else:
        ax.scatter([n],[c],s=26,color=MUTED,zorder=3)
        ax.annotate(l,(n,c),textcoords='offset points',xytext=(6,3),
                    fontsize=7.5,color=SECONDARY)
ax.scatter([0],[0],s=54,color=DBLUE,zorder=4)
ax.annotate('the real model\n(nothing replaced)',(0,0),
            textcoords='offset points',xytext=(8,4),fontsize=8.5,color=INK)
ax.axhline(span,color=RED,lw=1.2,ls='--',alpha=.8)
ax.text(0.4,span-0.14,
        f'replace everything with nothing (embeddings only): '
        f'+{span:.2f} nats = 0% of the model’s work',
        fontsize=8.5,color=RED,va='top')
ax.axhline(0,color=GRID,lw=1)
sec=ax.secondary_yaxis('right',
    functions=(lambda y:100*(1-y/span),lambda p:span*(1-p/100)))
sec.set_ylabel('% of the model’s work retained',color=SECONDARY,
               fontsize=9)
sec.tick_params(colors=SECONDARY,labelsize=8)
ax.set_xlim(-1.2,37); ax.set_ylim(-0.15,span+0.35)
ax.set_xticks([0,6,12,18,24,30,36])
ax.set_xlabel('components replaced by interpretable stand-ins (of 36)',
              fontsize=10,color=INK)
ax.set_ylabel('fidelity cost (nats of cross-entropy lost)',fontsize=10,
              color=INK)
ax.set_title('bilin18 compression benchmark: how much of the model can '
             'be replaced, at what cost',fontsize=11,color=INK,pad=12)
ax.grid(color=GRID,lw=.7,alpha=.8); ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=SECONDARY,labelsize=8.5)
fig.tight_layout()
fig.savefig(PT+'bilin18_frontier.png',dpi=200,facecolor=SURFACE)
print('wrote bilin18_frontier.png')

# ---- Figure 2: per-module relevance, ranked ----
marg={  # in-assembly LOO marginal, from the version that introduced it
 'attn0':.000,'attn1':.007,'mlp0':.010,'mlp1':.006,'mlp2':.014,
 'mlp3':.004,                                                         # 274
 'mlp10':.013,'mlp11':.016,'mlp12':.015,'mlp13':.019,'mlp14':.014,
 'mlp15':.011,'mlp16':.031,'mlp17':.082,                              # 274
 'mlp4':.19,'mlp5':.02,'mlp6':.11,'mlp7':.14,'mlp8':.12,'mlp9':.13,  # v5
 'attn2':.18,'attn3':.11,'attn4':.14,'attn5':-.38,'attn6':.23,
 'attn7':.28,'attn8':.40,'attn9':.09,                                 # v9
 'attn10':.05,'attn11':.06,'attn12':.05,'attn13':.10,'attn14':.12,
 'attn15':.04,'attn16':.13,'attn17':.06}                              # v6
items=sorted(mr['mean_abl'].items(),key=lambda kv:-kv[1])
fig,ax=plt.subplots(figsize=(9.6,5.6),facecolor=SURFACE)
ax.set_facecolor(SURFACE)
xs=range(len(items))
cols=[DBLUE if k.startswith('mlp') else BLUE for k,_ in items]
ax.bar(xs,[v for _,v in items],color=cols,width=.72,zorder=3)
mx=[i for i,(k,_) in enumerate(items) if k in marg]
my=[marg[items[i][0]] for i in mx]
ax.scatter(mx,my,s=22,color=RED,zorder=4,
           label='cost our best stand-in still pays in the assembly')
ax.axhline(0,color=GRID,lw=1)
ax.set_xticks(list(xs))
ax.set_xticklabels([k.replace('attn','a').replace('mlp','m')
                    for k,_ in items],fontsize=7,rotation=90,
                   color=SECONDARY)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=DBLUE,label='MLP'),
                   mp.Patch(color=BLUE,label='attention'),
                   plt.Line2D([],[],marker='o',ls='',color=RED,
                              label='in-assembly stand-in marginal '
                                    '(where measured)')],
          fontsize=8.5,frameon=False,loc='upper right')
ax.set_ylabel('CE cost of replacing the component with its mean output '
              '(nats)',fontsize=9.5,color=INK)
ax.set_title('bilin18 module relevance, ranked -- and what the current '
             'stand-ins leave on the table',fontsize=11,color=INK,pad=12)
ax.grid(axis='y',color=GRID,lw=.7,alpha=.8); ax.set_axisbelow(True)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=SECONDARY,labelsize=8.5)
fig.tight_layout()
fig.savefig(PT+'bilin18_module_relevance.png',dpi=200,facecolor=SURFACE)
print('wrote bilin18_module_relevance.png')
