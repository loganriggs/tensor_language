"""Native check: each source-read block is conditionally quadratic in amplitudes."""
import json,torch
from pathlib import Path
from conditional_source_constraints import ConditionalSourceConstraints
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];parent=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)['calibration_shaped_inherited'];bundle=expand(parent);H=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metric=ConditionalSourceConstraints(d,H,base);rows=[];rng=torch.Generator().manual_seed(54802)
for parity in (0,1):
 direction=torch.zeros_like(H)
 for j in range(3):
  v=torch.randn(H.shape[-1],generator=rng,dtype=H.dtype);v/=v.norm();direction[2*j+parity]=torch.outer(v,v)*H[2*j+parity].norm()*.01
 q0=metric.ratios_full(H);qp=metric.ratios_full(H+direction);qm=metric.ratios_full(H-direction);linear=(qp-qm)/2;quadratic=(qp+qm)/2-q0
 for alpha in (-.7,.3,1.2):
  predicted=q0+alpha*linear+alpha*alpha*quadratic;actual=metric.ratios_full(H+alpha*direction);error=float((predicted-actual).abs().max()/actual.abs().max().clamp_min(1));assert error<1e-8
  rows.append(dict(parity=parity,alpha=alpha,relative_quadratic_replay=error))
(P/'CONDITIONAL_TWO_READ_CONTROLS_V1.json').write_text(json.dumps(dict(records=rows,pass_all=True,scope='Both read blocks separately, other block and native h fixed. This validates conditional quadratic dependence, not joint convexity or fidelity recovery. Existing correction products already store two readout coefficients.'),indent=2)+'\n');print('Six native conditional-quadratic controls pass')
