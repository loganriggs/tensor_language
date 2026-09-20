#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_major pred_c_all
"""Frozen code-shift repeat of run_direct_native_mode_intervention_v1; unchanged registered bars.
CODE_SHIFT_INTERVENTION_PLAN_V1.md: no fitting, 16 files/context128.
Null: FineWeb results fail domain transfer. Original candidate price retained.
"""
import run_direct_native_mode_intervention_v1 as run
run.PANEL_PATH=run.P/'CODE_SHIFT_PANEL_V1.pt'
run.OUTPUT_STEM='CODE_SHIFT_MODES_V1'
run.PLAN=dict(run.PLAN,documents=list(range(16)),panel='CODE_SHIFT_PANEL_V1.pt')
run.PLAN['registered_predictions']={'pred_a_replay': 'relative replay<1e-5', 'pred_b_major': 'modes0/1 cosine>.9 and error<.4', 'pred_c_all': 'all4 cosine>.8 and error<.65'}
if __name__=='__main__':run.main()
