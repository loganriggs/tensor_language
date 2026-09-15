#!/usr/bin/env python3
"""Plot the fresh regional routing/value factorial from its frozen receipt."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_value_composition_fresh_v1_result.json"
OUT=HERE/"assets/research_update_2026-09-15_routing_value_composition.png"

data=json.loads(RESULT.read_text())["family_reports"]
labels=("Routing", "Value", "Joint", "Interaction")
keys=("routing_target_rms", "value_target_rms", "joint_target_rms", "interaction_target_rms")
x=np.arange(len(labels)); width=.36
fig,ax=plt.subplots(figsize=(8.2,4.6),layout="constrained")
for offset,(family,color) in enumerate((("0","#2b6cb0"),("1","#dd6b20"))):
    values=[data[family]["logit"][key] for key in keys]
    bars=ax.bar(x+(offset-.5)*width,values,width,label=f"Fresh template {int(family)+1}",color=color)
    ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
ax.set_xticks(x,labels); ax.set_ylabel("Paired target-effect RMS (logits)")
ax.set_title("Head9.8 routing and current-value branches overlap strongly")
ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT,dpi=180); print(OUT)
