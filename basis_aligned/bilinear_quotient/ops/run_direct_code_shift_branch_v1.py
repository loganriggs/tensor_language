#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_preservation pred_c_effect
"""Frozen code-shift repeat of run_direct_native_quartic_branch_v1; unchanged registered bars.
CODE_SHIFT_INTERVENTION_PLAN_V1.md: no fitting, 16 files/context128.
Null: FineWeb results fail domain transfer. Original candidate price retained.
"""
import run_direct_native_quartic_branch_v1 as run
run.PANEL_PATH=run.P/'CODE_SHIFT_PANEL_V1.pt'
run.OUTPUT_STEM='CODE_SHIFT_BRANCH_V1'
run.PLAN=dict(run.PLAN,documents=list(range(16)),panel='CODE_SHIFT_PANEL_V1.pt')
run.PLAN['registered_predictions']={'pred_a_instrument': 'replay and solve <1e-5', 'pred_b_preservation': 'CE<.02 and KL<.02', 'pred_c_effect': 'logit norm ratio<.5'}
if __name__=='__main__':run.main()
