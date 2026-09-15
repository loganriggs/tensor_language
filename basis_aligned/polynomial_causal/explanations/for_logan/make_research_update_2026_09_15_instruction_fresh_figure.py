#!/usr/bin/env python3
"""Plot fresh instruction-cross fidelity and the selectivity failure."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_instruction_cross_fresh_v1_result.json"
OUT=HERE/"assets/research_update_2026-09-15_instruction_cross_fresh.png"
d=json.loads(RESULT.read_text())["family_reports"]
labels=("Head-vector\nerror","Recursive-logit\nerror","Largest control\nratio")
x=np.arange(3); width=.36
fig,ax=plt.subplots(figsize=(8.2,4.6),layout="constrained")
for j,(family,color) in enumerate((("0","#2b6cb0"),("1","#dd6b20"))):
    r=d[family]; vals=(r["instruction_head_replay_relative_error"],r["logit"]["instruction_cross_replay_relative_error"],max(r["logit"]["unrelated_reader_ratios"].values()))
    bars=ax.bar(x+(j-.5)*width,vals,width,color=color,label=f"Fresh template {int(family)+1}")
    ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
ax.plot([-.5,.5],[.30,.30],color="#555",ls="--",lw=1)
ax.plot([.5,1.5],[.35,.35],color="#555",ls="--",lw=1)
ax.plot([1.5,2.5],[.75,.75],color="#555",ls="--",lw=1,label="Registered gates")
ax.set_xticks(x,labels); ax.set_ylabel("Error or control/target ratio (lower is better)")
ax.set_title("Instruction-source compression transfers, but one control fails")
ax.set_ylim(0,1.3); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT,dpi=180); print(OUT)
