"""A zero-curvature limit must recover the independently solved LP optimum."""
import json
from pathlib import Path
import numpy as np
from budgeted_modal_direction import choose as linear_choose
from quadratic_budgeted_direction import choose

def main():
    rng=np.random.default_rng(613);errors=[]
    for _ in range(12):
        g=rng.normal(size=(4,5));h=np.zeros((4,5,5));reference,_=linear_choose(g,.08);a,info=choose(g,h,np.zeros(5));sign=np.sign((-g[0,2:]).sum()) or 1.;error=abs(sign*g[0]@(a-reference));errors.append(float(error));assert error<1e-8
    out=dict(zero_curvature_lp_objective_error=max(errors),cases=12)
    Path(__file__).with_name('QUADRATIC_BUDGETED_DIRECTION_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
