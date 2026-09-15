#!/usr/bin/env python3
"""Plot source-token compression and role errors for the head9.8 cross term."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_head_cross_source_census_v2_result.json"
OUT=HERE/"assets/research_update_2026-09-15_head_cross_sources.png"
d=json.loads(RESULT.read_text())["family_reports"]
labels=("Top 4\nwithin family","Top 4\ncross-family","Instruction\nonly","Description\nonly")
x=np.arange(4); width=.36
fig,ax=plt.subplots(figsize=(8.2,4.6),layout="constrained")
for j,(family,color) in enumerate((("0","#2b6cb0"),("1","#dd6b20"))):
    r=d[family]; vals=(r["top4_vector_replay_relative_error"],r["top4_transfer_to_family"]["vector_replay_relative_error"],r["roles"]["instruction"]["to_total_relative_error"],r["roles"]["description"]["to_total_relative_error"])
    bars=ax.bar(x+(j-.5)*width,vals,width,color=color,label=f"Template {int(family)+1}")
    ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
ax.axhline(.35,color="#555",ls="--",lw=1,label="Within-family gate (.35)")
ax.set_xticks(x,labels); ax.set_ylabel("Cross-vector relative error (lower is better)")
ax.set_title("Four source positions capture the mostly instruction-side cross term")
ax.set_ylim(0,1); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,ncol=2)
OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT,dpi=180); print(OUT)
