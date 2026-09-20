"""CPU-only cross-receipt and fixed-frame audit; no model checkpoint required."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'basis_aligned/bilinear_quotient/circuits/followups'
OUT=Path(__file__).with_name('SUBJECT_INTERFACE_AUDIT_2026-09-20.json')

def main():
    records={v:json.loads((ART/f'subject_attention_freeze_v{v}_result.json').read_text()) for v in range(680,686)}
    comparisons=[]
    for v in range(681,686):
        for source,cells in records[680]['reuse_reports'].items():
            for name,cell in cells.items():
                other=records[v]['reuse_reports'][source][name]
                for key in ['target','prediction','native_modal','predicted_modal']:
                    comparisons.append(float(np.max(np.abs(np.asarray(cell[key])-other[key]))))
    assert max(comparisons)==0
    for source,cells in records[684]['initial_reports'].items():
        for name,cell in cells.items():
            for key in ['prediction','predicted_modal']:
                assert np.array_equal(cell[key],records[685]['initial_reports'][source][name][key])
    package=torch.load(ART/'subject_response_v665_program.pt',map_location='cpu',weights_only=True)
    P=package['producer']['programs'][0]['input_basis'].double()
    W=package['producer']['initial_encoder'].double()
    identity_error=float((W.T@P-torch.eye(P.shape[1],dtype=P.dtype)).abs().max())
    torch.manual_seed(686)
    delta=torch.randn(16,P.shape[0],dtype=P.dtype)
    decoded=(delta@W)@P.T
    kernel_error=float(((delta-decoded)@W).abs().max())
    torch.testing.assert_close(W.T@P,torch.eye(P.shape[1],dtype=P.dtype),atol=1e-10,rtol=1e-10)
    summary={}
    for name,cells in records[685]['initial_reports'].items():
        summary[name]=dict(max_target_error=max(c['unprojected_relative_error'] for c in cells.values()),
                          max_modal_error=max(max(c['unprojected_modal_error']) for c in cells.values()))
    result=dict(cross_receipt_max_difference=max(comparisons),initial_final_projection_rerun_identical=True,
                encoder_decoder_identity_max_error=identity_error,random_discarded_coordinate_max_error=kernel_error,
                exact_native_after_initial_decode=summary,
                algebra='For column delta, z=W.T delta, decoded=P z; W.T P=I implies W.T(delta-decoded)=0.',
                scope='Fixed linear reconstruction is not faithful for all tested source effects. Original versus decoded states constitute a kernel-direction diagnostic on the continuous interface; decoded states need not lie on the natural/source-edit manifold. No lower bound against all width8 or nonlinear predictors.')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
