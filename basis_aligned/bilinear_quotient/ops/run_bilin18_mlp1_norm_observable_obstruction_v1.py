"""Exact dyadic obstruction to any proper linear-plus-norm MLP1 reader state.

A: source/control/dyadic/FP64 integer bounds, matrix symmetry and saved reload,
independent integer determinant agreement. B: determinant modulo65521 nonzero.
C: full-domain mathematical scope and native price recorded (descriptive).
Zero forwards/updates, fixed first temporal/iswas readers; null is inconclusive.
All native weights remain; no nonlinear-circuit impossibility or adoption claim.
"""
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_nonzero_commutator_determinant pred_c_scope_and_price_recorded
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import run_bilin18_mlp1_joint_reader_weight_v1 as parent
from circuit_fast_screen_managed_runner import atomic_create_json

RUNNER=Path(__file__).resolve();POLY=parent.POLY
sys.path.insert(0,str(POLY))
import modular_bilinear_obstruction as M
PRIOR=POLY/'BILIN18_MLP1_NORM_OBSERVABLE_OBSTRUCTION_V1_PREREGISTRATION.md'
OUT=POLY/'BILIN18_MLP1_NORM_OBSERVABLE_OBSTRUCTION_V1_RESULT.json'
MATRIX=POLY/'BILIN18_MLP1_NORM_OBSERVABLE_OBSTRUCTION_V1_MATRIX.npz'
FILES={'prior':PRIOR,'primitive':Path(M.__file__),'parent_result':parent.OUT,'readers':parent.READERS,
       'parent_runner':Path(parent.__file__),'parent_sources':parent.BINDING}
EXPECTED={'prior': 'b32ed4bfeea07925a905db9d156e2068d5e08c4d3a9cf157e8ae7e2bb7f7d6bc', 'primitive': '34207bff5a9e9810072387c9cf3fb53ad5042652a0ee3fab4189ba8eefdbf479', 'parent_result': 'dedfd50f5111f898ebf681072685cdcf18fc8685b0f86940869fd9050443cfc3', 'readers': '2d26a6cb4487597000e65cceb9cf62145fdd68ce856e064c2169b912b9d603fc', 'parent_runner': 'cd9b4efd8ffe9fa8d4b76d2c1113e5de496c695dc20bb9313ac58dc3680abdf2', 'parent_sources': 'd5ee3731c07fe85525c4b69eaa8a7d88cf1d29b8e889e79be30602af96609460'}


def gpu_determinant(torch,matrix,prime):
    a=matrix.clone();det=1;n=len(a);swaps=0;pivots=[]
    for i in range(n):
        possible=torch.nonzero(a[i:,i],as_tuple=False)
        if not len(possible):return 0,{'pivots':pivots,'swaps':swaps,'first_missing_pivot':i}
        j=i+int(possible[0,0])
        if j!=i:
            temp=a[i].clone();a[i]=a[j];a[j]=temp;det=-det;swaps+=1
        pivot=int(a[i,i]);pivots.append(pivot);det=det*pivot%prime
        factors=(a[i+1:,i]*pow(pivot,-1,prime)).remainder(prime)
        a[i+1:,i+1:]=(a[i+1:,i+1:]-factors[:,None]*a[i,i+1:]).remainder(prime)
        a[i+1:,i]=0
    return det%prime,{'pivots':pivots,'swaps':swaps,'first_missing_pivot':None}


def main():
    observed={k:parent.sha(p) for k,p in FILES.items()};assert observed==EXPECTED
    authorities=json.loads(parent.BINDING.read_text())
    assert {p:parent.sha(p) for p in authorities}==authorities
    previous=json.loads(parent.OUT.read_text());assert previous['predictions']['pred_a_instrument']
    controls=M.controls();assert controls['passed']
    dry={'dryrun':True,'gpu_accessed':False,'model_loaded':False,'model_forwards':0,'model_updates':0,
         'matrix_shape':[1152,1152],'prime':M.PRIME,'fixed_readers':['temporal/0','iswas/0'],'authority_sha256':observed}
    if os.environ.get('BQLIB_DRYRUN')=='1' or os.environ.get('BQLIB_NO_MODEL')=='1':print(json.dumps(dry));return
    assert not OUT.exists() and not MATRIX.exists();signal.alarm(600);tic=time.perf_counter()
    backend=parent.producer.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    module=backend.model.transformer.h[1].mlp;prime=M.PRIME;bounds=[]
    def mod(values):
        return torch.tensor(M.dyadic_mod(values.detach().cpu().numpy()),dtype=torch.float64,device='cuda')
    def product(a,b):
        assert a.dtype==b.dtype==torch.float64 and a.shape[-1]==b.shape[0]
        bound=a.shape[-1]*(prime-1)**2;assert bound<2**53;bounds.append(bound)
        result=a@b
        assert bool(result.isfinite().all()) and bool(torch.equal(result,result.round()))
        return result.remainder(prime)
    def compile_forms(c,l,r,d):
        coeff=product(c,d);forms=[]
        for row in coeff:
            weighted=(row[:,None]*r).remainder(prime)
            raw=product(l.T,weighted)
            forms.append(((raw+raw.T)*pow(2,-1,prime)).remainder(prime))
        return torch.stack(forms)
    with torch.inference_mode():
        # Independent Fraction arithmetic oracle resides in primitive.controls.
        rng=np.random.default_rng(60915)
        arrays=[rng.integers(-3,4,s).astype(np.float32)/den for s,den in [((2,3),8),((5,4),4),((5,4),16),((3,5),2)]]
        toy=compile_forms(*[mod(torch.tensor(a)) for a in arrays])
        toy_expected=M.forms_mod_numpy(*arrays)
        toy_ok=np.array_equal(toy.cpu().numpy().astype(np.int64),toy_expected)
        toy_k=(product(toy[0],toy[1])-product(toy[1],toy[0])).remainder(prime)
        toy_det,_=gpu_determinant(torch,toy_k,prime)
        toy_ok=toy_ok and toy_det==M.determinant_mod(toy_k.cpu().numpy().astype(np.int64))
        saved=json.loads(parent.READERS.read_text())
        reader=torch.tensor(np.stack([np.asarray(saved['physical_readers'][t],dtype=np.float32)[:,0] for t in parent.TASKS]),dtype=torch.float32)
        q=compile_forms(mod(reader),mod(module.Left.weight),mod(module.Right.weight),mod(module.Down.weight))
        symmetric=bool(torch.equal(q,q.transpose(-1,-2)))
        forward=product(q[0],q[1]);reverse=product(q[1],q[0]);comm=(forward-reverse).remainder(prime)
        antisymmetric=bool(torch.equal((comm+comm.T).remainder(prime),torch.zeros_like(comm)))
        assert tuple(comm.shape)==(1152,1152)
        determinant,elimination=gpu_determinant(torch,comm,prime)
        matrix=comm.cpu().numpy().astype(np.uint16)
        with MATRIX.open('xb') as f:np.savez_compressed(f,commutator=matrix,prime=np.int64(prime))
        with np.load(MATRIX,allow_pickle=False) as z:
            reload=np.array_equal(z['commutator'],matrix) and int(z['prime'])==prime
        print(json.dumps({'stage':'gpu_certificate','determinant':determinant,'controls':toy_ok,'elapsed':time.perf_counter()-tic}),flush=True)
        cpu_start=time.perf_counter();cpu_det=M.determinant_mod(matrix,prime);cpu_seconds=time.perf_counter()-cpu_start
        a=controls['passed'] and toy_ok and symmetric and antisymmetric and reload and cpu_det==determinant
        b=determinant!=0
        result={'terminal':'invalid' if not a else 'proper_linear_plus_norm_quotient_impossible' if b else 'commutator_certificate_inconclusive',
            'predictions':{'pred_a_exact_instrument':a,'pred_b_nonzero_commutator_determinant':b,'pred_c_scope_and_price_recorded':True},
            'prime':prime,'determinant_mod_prime':determinant,'independent_cpu_determinant':cpu_det,'elimination':elimination,
            'controls':controls,'gpu_controls':toy_ok,'symmetric_forms':symmetric,'antisymmetric_commutator':antisymmetric,
            'matrix_reload':reload,'max_integer_dot_bound':max(bounds),'exact_integer_limit':2**53,
            'authority_sha256':observed,'runner_sha256':parent.sha(RUNNER),'matrix_sha256':parent.sha(MATRIX),
            'fixed_readers':['temporal/0','iswas/0'],'cpu_verification_seconds':cpu_seconds,
            'scope':'Exact real quadratic functions of fixed FP32 dyadic weights/readers; all real inputs and arbitrary decoder from linear coordinates plus norm. Does not exclude nonlinear producer programs, restricted domains, or finite-precision behavioral equivalence.',
            'price':{'model_forwards':0,'model_updates':0,'fit_updates':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'retained_background':'entire native model','matrix_shape':[1152,1152],'saved_matrix_bytes':MATRIX.stat().st_size,'adoption':False},
            'wall_seconds':time.perf_counter()-tic}
        atomic_create_json(OUT,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('elimination','authority_sha256')},indent=2))

if __name__=='__main__':main()
