"""Two-panel explainer for the composition-economics arc (sections
303-312): (A) the coverage-vs-fidelity trade-off curve, before vs after
tonight, points labeled by what the stand-ins ARE; (B) the pricing law:
the same 38 head-replacements cost 2.4x more on the substrate whose
internal signals are less accurate where those heads read."""
import sys
sys.path.insert(0,'/workspace/tensor_language')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from palette import INK, SECONDARY, MUTED, GRID, SURFACE
BLUE='#3987e5'; DBLUE='#104281'; RED='#e34948'
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'

fig,(ax,bx)=plt.subplots(1,2,figsize=(12.6,5.2),facecolor=SURFACE,
                         gridspec_kw={'width_ratios':[1.5,1]})
for a in (ax,bx): a.set_facecolor(SURFACE)

# ---- Panel A: the trade-off curve ----
OLD=[(0,0),(13,1.98),(20,2.02),(28,2.49),(34,2.93)]
NEW=[(0,0),(20,1.46),(24,2.29),(32,2.67),(34,2.93)]
ax.plot([p[0] for p in OLD],[p[1] for p in OLD],'--o',color=MUTED,
        lw=1.6,ms=5,label='before tonight')
ax.plot([p[0] for p in NEW],[p[1] for p in NEW],'-o',color=BLUE,
        lw=2.4,ms=7,mfc=DBLUE,label='after (all points on never-seen text)')
notes=[(20,1.46,'20 layers replaced:\nword-lookup tables built\nfrom weights alone + simple\nmath pieces; attention real'),
       (24,2.29,'+ 38 attention heads,\neach = "look at previous\nword" (or self) x one gain'),
       (32,2.67,'+ 8 more attention layers\nas per-class rules'),
       (34,2.93,'old best: 34 layers\n(more replaced, but\npricier recipe)')]
for x,y,t in notes:
    ax.annotate(t,(x,y),textcoords='offset points',
                xytext=(-2,14) if x<30 else (-96,-46),fontsize=7.8,
                color=INK)
ax.set_xlabel('how much of the model is replaced '
              '(layers of 36; heads counted fractionally)',fontsize=10,
              color=INK)
ax.set_ylabel('prediction quality lost (nats, fresh text)',
              fontsize=10,color=INK)
ax.set_title('A. The trade-off curve: replace more, lose more — '
             'the goal is pushing the right side down',fontsize=10.5,
             color=INK,pad=10)
ax.legend(fontsize=8.5,frameon=False,loc='upper left')
ax.grid(color=GRID,lw=.7,alpha=.8); ax.set_axisbelow(True)
ax.set_xlim(-1,36.5); ax.set_ylim(-0.1,3.3)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=SECONDARY,labelsize=8.5)

# ---- Panel B: the pricing law ----
labels=['substrate 1:\nfitted tables\n(accurate mid-model\nsignals)',
        'substrate 2:\nweights-built tables\n(mid-model signals\n18% worse)']
cost=[0.43,1.04]
bars=bx.bar([0,1],cost,width=.52,color=[DBLUE,RED],zorder=3)
for i,c in enumerate(cost):
    bx.text(i,c+0.03,f'+{c:.2f} nats',ha='center',fontsize=10,
            color=INK,fontweight='bold')
bx.set_xticks([0,1]); bx.set_xticklabels(labels,fontsize=8.2,
                                         color=SECONDARY)
bx.set_ylabel('cost of adding the SAME 38 head\nreplacements (nats, fresh)',
              fontsize=9.5,color=INK)
bx.set_title('B. The pricing law: identical replacements cost 2.4x\n'
             'more when the signals they read are less accurate',
             fontsize=10.5,color=INK,pad=10)
bx.set_ylim(0,1.3)
bx.grid(axis='y',color=GRID,lw=.7,alpha=.8); bx.set_axisbelow(True)
for s in ('top','right'): bx.spines[s].set_visible(False)
for s in ('left','bottom'): bx.spines[s].set_color(GRID)
bx.tick_params(colors=SECONDARY,labelsize=8.5)
fig.tight_layout()
fig.savefig(PT+'bilin18_arc_explainer.png',dpi=200,facecolor=SURFACE)
print('wrote bilin18_arc_explainer.png')
