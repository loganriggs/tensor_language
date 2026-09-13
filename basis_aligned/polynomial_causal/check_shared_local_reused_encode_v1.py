"""Compare independent joint-span and shared-complement LS implementations."""
import json,time
from pathlib import Path
import torch
from shared_local_subspaces_v1 import encode,parts
from shared_local_reused_encode_v1 import reused_encode


def main():
    torch.set_num_threads(2);torch.manual_seed(9531)
    n,d,g,k,r=1024,1152,64,32,8
    x=torch.randn(n,d,dtype=torch.float64)
    p=torch.linalg.qr(torch.randn(d,g,dtype=x.dtype))[0].T
    banks=[torch.linalg.qr(torch.randn(d,r,dtype=x.dtype))[0].T for _ in range(k)]
    tic=time.perf_counter();a=encode(x,p,banks);old=time.perf_counter()-tic
    tic=time.perf_counter();b=reused_encode(x,p,banks);new=time.perf_counter()-tic
    err=float((sum(parts(a))-sum(parts(b))).norm()/x.norm())
    codes=max(float((a[key]-b[key]).norm()/a[key].norm()) for key in ['global_codes','local_codes'])
    assert torch.equal(a['labels'],b['labels'])
    assert max(err,codes)<1e-10
    # Exactly shared rows exercise the coefficient-gauge-preserving fallback.
    degenerate=[p[:r]]+banks[1:]
    c=encode(x,p,degenerate);e=reused_encode(x,p,degenerate)
    fallback_error=float((c['global_codes']-e['global_codes']).norm())
    assert fallback_error==0
    result=dict(shape=[n,d],widths=[g,k,r],prediction_relative_error=err,
                maximum_code_relative_error=codes,identical_assignments=True,
                fallback_code_error=fallback_error,reference_seconds=old,reused_seconds=new,
                speed_ratio=old/new,
                scope='Synthetic native-width CPU encoder control; not native fit timing or compression gain.')
    with Path(__file__).with_name('SHARED_LOCAL_REUSED_ENCODE_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
