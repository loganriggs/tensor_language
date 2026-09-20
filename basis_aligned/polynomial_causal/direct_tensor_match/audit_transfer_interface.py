"""Constructive controls for interpreting row-dependent amplitude-coordinate transfer."""
import json
from pathlib import Path
import torch
from core import metric,inner,multiply,terms
from coordinates import transform_matrix
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1606)
d=5;o=3;n=len(terms(d,2));q=torch.randn(1,n);r=torch.randn(o,n);global_coeff=multiply(q,r,d,2,2);M=metric(d,4)
# Same global polynomial circuit under differing invertible input frames and readout maps.
K0=torch.linalg.qr(torch.randn(d,d)).Q@torch.diag(torch.linspace(.5,1.5,d));K1=torch.linalg.qr(torch.randn(d,d)).Q@torch.diag(torch.linspace(.7,1.3,d));O0=torch.linalg.qr(torch.randn(o,o)).Q;O1=torch.linalg.qr(torch.randn(o,o)).Q
c0=O0@global_coeff@transform_matrix(K0,4);c1=O1@global_coeff@transform_matrix(K1,4)
naive=c0-c1;naive_error=float((inner(naive,naive,M)/inner(c1,c1,M)).sqrt())
A=torch.linalg.solve(K0,K1);output_transport=O1@torch.linalg.inv(O0);transported=output_transport@c0@transform_matrix(A,4);delta=transported-c1;transport_error=float((inner(delta,delta,M)/inner(c1,c1,M)).sqrt());assert transport_error<1e-10 and naive_error>.1
# Rectangular ports leave unseen directions: two global quartics agree on one port, differ elsewhere.
# Ambient x in R6. P0 embeds a in first5 coordinates, x5=0. P1 uses x5=a0.
# F=(sum_{i<5} x_i^2)^2, G=F+x5^4. Their restrictions to P0 are identical.
zero=torch.zeros(1,len(terms(5,4)));difference1=zero.clone();difference1[0,terms(5,4).index((0,0,0,0))]=1.
ambiguity=float(inner(difference1,difference1,M).sqrt());assert ambiguity>0
result=dict(square_frames=dict(naive_frozen_program_error=naive_error,exact_transport_error=transport_error,global_circuit_identical=True),rectangular_frames=dict(calibration_restriction_difference_norm=0.,other_port_difference_norm=ambiguity,global_polynomials='F=(sum_{i=0}^4 x_i^2)^2; G=F+x_5^4',calibration_port='x=(a0,a1,a2,a3,a4,0)',other_port='x=(a0,a1,a2,a3,a4,a0)'),native_mapping=dict(source='ops/run_native_two_mlp_quartic_ht_v1r1.py',input_port='K_b=ports[mlp_input_directions][0][batch,pos], 1152x5',output_port='q_b=ports[mlp_readers][1][batch,pos], 4x1152',shared_native_layers=[11,12],implication='Frozen amplitude-coordinate program failure is not a lower bound on existence of reusable ambient-residual circuits. It remains a valid failure of the specified fixed amplitude interface.'),scope='Constructive mathematical controls, not a demonstration that native row differences admit invertible coordinate transport.')
(P/'TRANSFER_INTERFACE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
