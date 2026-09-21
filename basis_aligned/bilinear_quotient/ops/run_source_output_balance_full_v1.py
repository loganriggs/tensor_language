#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_sensitivity pred_c_values
"""Full output whitening, power1, same random graph capacity.
pred_a_instrument implicit/dense objective agreement<1e-8.
pred_b_sensitivity both component Jacobians improve>=10% vsparent.
pred_c_values eachscalar<=.15 and<=1.10separatebaseline.
Additional all4originalcanonical-sourcecosines>=.99 required for grouping claim.
4Adamfits rates.02/.05 x seeds816/1816,2000cosine steps; output-map undone
before original metric/value/derivative checks.512products897804floats.
Null: balancing weak contrast harms other features or native fidelity.
"""
from run_source_sobolev_refit_v1 import main
if __name__=='__main__':
 predictions=dict(pred_a_instrument=None,pred_b_sensitivity=None,pred_c_values=None)
 main('SOURCE_OUTPUT_BALANCE_FULL_PLAN_V1.json','SOURCE_OUTPUT_BALANCE_FULL_V1.json','SOURCE_OUTPUT_BALANCE_FULL_PROGRAMS_V1.pt')
