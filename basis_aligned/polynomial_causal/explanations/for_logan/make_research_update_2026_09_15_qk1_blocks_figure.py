#!/usr/bin/env python3
"""Plot QK1 ordered-block target and work/jobs effects in the failed family."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
RESULT=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_qk1_block_collateral_v1_result.json"
OUT=HERE/"assets/research_update_2026-09-15_qk1_blocks.png"
r=json.loads(RESULT.read_text())["family_reports"]["1"]["logit"]; labels=("DD","DR","RD","All three");x=np.arange(4);width=.36
target=[r["block_effects"][k]["target_rms"] for k in ("DD","DR","RD")]+[r["full_target_rms"]];control=[r["block_effects"][k]["work_jobs_rms"] for k in ("DD","DR","RD")]+[r["full_work_jobs_rms"]]
fig,ax=plt.subplots(figsize=(8.2,4.6),layout="constrained")
for j,(vals,color,name) in enumerate(((target,"#2b6cb0","Regional target"),(control,"#dd6b20","Work/jobs control"))):
    bars=ax.bar(x+(j-.5)*width,vals,width,color=color,label=name);ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
ax.set_xticks(x,labels);ax.set_ylabel("Effect RMS (logits)");ax.set_title("QK1 collateral is distributed across ordered interaction blocks");ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False)
OUT.parent.mkdir(exist_ok=True);fig.savefig(OUT,dpi=180);print(OUT)
