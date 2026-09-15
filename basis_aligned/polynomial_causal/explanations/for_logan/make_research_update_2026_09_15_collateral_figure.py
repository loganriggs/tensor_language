#!/usr/bin/env python3
"""Contrast full-package and cross-specific work/jobs collateral."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
FRESH=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_instruction_cross_fresh_v1_result.json"
ATTR=HERE.parents[2]/"bilinear_quotient/circuits/fast_screens/setting2_regional_instruction_cross_collateral_attribution_v1_result.json"
OUT=HERE/"assets/research_update_2026-09-15_cross_collateral.png"
fresh=json.loads(FRESH.read_text())["family_reports"]; attr=json.loads(ATTR.read_text())["family_reports"]
labels=("Full package", "Instruction cross", "Full cross")
x=np.arange(3); width=.36
fig,ax=plt.subplots(figsize=(8.2,4.6),layout="constrained")
for j,(family,color) in enumerate((("0","#2b6cb0"),("1","#dd6b20"))):
    vals=(fresh[family]["logit"]["unrelated_reader_ratios"]["work_jobs"],attr[family]["logit"]["instruction_cross_control_ratios"]["work_jobs"],attr[family]["logit"]["full_cross_control_ratios"]["work_jobs"])
    bars=ax.bar(x+(j-.5)*width,vals,width,color=color,label=f"Fresh template {int(family)+1}")
    ax.bar_label(bars,fmt="%.2f",padding=2,fontsize=9)
ax.axhline(.75,color="#555",ls="--",lw=1,label="Selectivity gate (.75)")
ax.set_xticks(x,labels); ax.set_ylabel("Work/jobs control-to-target RMS ratio")
ax.set_title("The failed control comes from additive branches, not the cross term")
ax.set_ylim(0,1.3); ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False)
OUT.parent.mkdir(exist_ok=True);fig.savefig(OUT,dpi=180);print(OUT)
