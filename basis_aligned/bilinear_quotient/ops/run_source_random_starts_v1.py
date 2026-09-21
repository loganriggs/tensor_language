#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_sensitivity pred_c_values
"""Random-start source tensor decomposition, same partial graph capacity.
8Adamfits: lambda0/1 x rates.02/.05 x seeds816/1816,2000steps each.
pred_a_instrument implicit/dense replay<1e-8; finite;512products897804floats.
pred_b_sensitivity primarylambda1 BOTH conditional Jacobianerrors<=.9parent.
pred_c_values allthree scalarerrors<=.15 and<=1.10separatebaseline.
Null: parent-decomposition initialization is important at this bounded budget.
No claim of equal lifetime optimization budget with the warm-start parent.
Only shared directions random; analytic readout and fixed private3. No nativeforwards.
"""
from run_source_sobolev_refit_v1 import main
if __name__=='__main__':
 predictions=dict(pred_a_instrument=None,pred_b_sensitivity=None,pred_c_values=None)
 main('SOURCE_RANDOM_START_PLAN_V1.json','SOURCE_RANDOM_START_FIT_V1.json','SOURCE_RANDOM_START_PROGRAMS_V1.pt')
