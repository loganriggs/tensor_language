"""Conditional bound: output-only changes with <=.001 added error per fit.

For projection F of T onto a fixed product span, ||T-(F+d)||^2=
||T-F||^2+||d||^2 for any output-only d. Normalization is Lipschitz with
||normalize(F+d)-normalize(F)|| <=2||d||/||F||.
Triangle inequality bounds the smallest attainable unit-vector separation.
This does not constrain changes to the readers or their span.
"""
import json
import math
from pathlib import Path


def main():
    p=Path(__file__).resolve().parent
    source=json.loads((p/'INTERMEDIATE_FUNCTION_STABILITY_V1_AUDIT.json').read_text())
    assert source['predictions']['pred_a_instrument']
    energy=source['fitted_energies'];cross=source['native_crosses']
    orthogonality=max(abs(e-c) for e,c in zip(energy,cross))
    budget=.001
    assert min(energy)>budget and orthogonality<=1e-8
    separation=math.sqrt(2*(1-source['function_cosine']))
    displacements=[2*math.sqrt(budget/e) for e in energy]
    minimum_separation=max(0.,separation-sum(displacements))
    upper=1-minimum_separation**2/2
    # Conservative equal per-fit budget below which .9 cosine is impossible.
    required_unit_motion=separation-math.sqrt(2*(1-.9))
    threshold=(max(0.,required_unit_motion)/(2*sum(1/math.sqrt(e) for e in energy)))**2
    result=dict(predictions=dict(pred_a_projection=orthogonality<=1e-8,
        pred_b_output_repair_excluded=upper<.9),
        per_fit_capture_loss_budget=budget,projection_identity_error=orthogonality,
        original_function_cosine=source['function_cosine'],original_unit_separation=separation,
        maximum_unit_displacements=displacements,minimum_unit_separation=minimum_separation,
        attainable_cosine_upper_bound=upper,necessary_equal_budget_for_point9=threshold,
        assumption='Exact conditional output projections. Numerical projection residuals are small, not interval-certified.',
        scope='Frozen intermediate cross-start readers; no bound for reader changes, nonlinear optimization or native circuit recoverability.',
        source='INTERMEDIATE_FUNCTION_STABILITY_V1_AUDIT.json',sources=source['sources'])
    with (p/'OUTPUT_ONLY_STABILITY_BOUND_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
