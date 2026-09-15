#!/usr/bin/env python3
"""Plot target and failed-reader magnitudes for additive branch attribution."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_additive_branch_collateral_v1_result.json"
OUT=HERE/"assets/research_update_2026-09-15_additive_branches.png"
r=json.loads(RESULT.read_text())["family_reports"]["1"]["logit"]
labels=("Routing", "Value", "Additive"); x=np.arange(3); width=.36
target=(r["routing_target_rms"],r["value_target_rms"],r["additive_target_rms"]); c=r["controls"]["work_jobs"]; control=(c["routing_rms"],c["value_rms"],c["additive_rms"])
fig,ax=plt.subplots(figsize=(8.2,4.6),layout="constrained")
for j,(vals,color,name) in enumerate(((target,"#2b6cb0","Regional target"),(control,"#dd6b20","Work/jobs control"))):
    bars=ax.bar(x+(j-.5)*width,vals,width,color=color,label=name);ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
ax.set_xticks(x,labels);ax.set_ylabel("Paired target or row-wise control RMS (logits)")
ax.set_title("QK1 routing causes the failed control; both singles carry the target")
ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False)
OUT.parent.mkdir(exist_ok=True);fig.savefig(OUT,dpi=180);print(OUT)
