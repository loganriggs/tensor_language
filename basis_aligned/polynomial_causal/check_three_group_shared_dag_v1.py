"""Cached real-port fidelity, numerical stress and timing for retained DAG rewrites."""
from pathlib import Path
import json,time,statistics
import torch
from joint_attention_three_group_v1 import execute as original
from three_group_shared_dag_v1 import execute
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(356)
    saved=torch.load(P/'ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    ports=[tuple(x[k] for k in (0,1,3,4)) for x in saved['ports']]
    assert len(ports)==120
    W=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].double()
    methods={'original':original,'horner':lambda *p:execute(*p),
             'shared_product':lambda *p:execute(*p,shared_product=True)}
    results={}
    for name,fn in methods.items():
        errors=[];error_sq=ref_sq=0.;identical=0
        for p in ports:
            expected=original(*p)@W.T;actual=fn(*p)@W.T
            errors.append(float((actual-expected).norm()/expected.norm().clamp_min(1e-30)))
            error_sq+=float((actual-expected).square().sum());ref_sq+=float(expected.square().sum())
            identical+=int(torch.equal(actual.float(),expected.float()))
        fn(*ports[0]);timings=[]
        for _ in range(7):
            tic=time.perf_counter()
            for p in ports:fn(*p)
            timings.append(time.perf_counter()-tic)
        results[name]=dict(max_relative_write_error=max(errors),aggregate_relative_write_error=(error_sq/ref_sq)**.5,
                           identical_fp32_residual_writes=identical,median_120_core_calls_seconds=statistics.median(timings))
    # Use actual quantized FP32 port inputs for both reference and rewrites.
    stress=[]
    native=(torch.randn(4,32),torch.randn(4,32),torch.randn(4,32,128))
    for strength in (1.,.01,.0001,.000001):
        child=tuple(x+strength*torch.randn_like(x) for x in native)
        remainder=tuple(x+strength*torch.randn_like(x) for x in native)
        additive=tuple(c+r-n for c,r,n in zip(child,remainder,native))
        p=(native,child,remainder,additive)
        reference=original(*(tuple(x.double() for x in corner) for corner in p))
        stress.append(dict(strength=strength,relative_errors={name:float((fn(*p).double()-reference).norm()/reference.norm().clamp_min(1e-30)) for name,fn in methods.items()}))
    assert results['horner']['max_relative_write_error']<1e-8
    result=dict(native_ports=results,fp32_stress=stress,gate_multiplications={'original':8,'horner':6,'shared_product':5},
                total_core_multiplications_per_source_at_width128={'original':392,'horner':390,'shared_product':389},
                scope='Exact real-arithmetic rewrite of the retained approximation, not of full attention. '
                '120cached native FP64 port contexts; scalar graph savings only, three value contractions '
                'and all input generators unchanged. Timings CPU eager core only, not GPU or full model. '
                'FP32 stress is synthetic on quantized inputs; recorded failures are not filtered.')
    (P/'THREE_GROUP_SHARED_DAG_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
