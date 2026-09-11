#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact full-centered-U source symmetry energy.
import os,sys,json,time,signal,hashlib,math
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from full_source_quadratic_energy_v1 import energies
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    binding=json.loads((P/'FULL_SOURCE_QUADRATIC_ENERGY_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'FULL_SOURCE_QUADRATIC_ENERGY_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,output_rank_truncation=False,heads=9,source_tuple_width=2304)));return
    out=P/'FULL_SOURCE_QUADRATIC_ENERGY_V1_RESULT.json';assert not out.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True);u=sd['lm_head.weight'].double().cuda();u-=u.mean(0);metric=u.T@u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right','Down']]
    k=d.T@metric@d;o=sd['transformer.h.17.attn.c_proj.weight'].double().cuda();vc=sd['transformer.h.17.attn.c_v.weight'].double().cuda();vb=sd['transformer.h.0.attn.c_v.weight'].double().cuda();mix=float(sd['transformer.h.17.attn.lamb'])
    w=torch.cat(((1-mix)*vc,mix*vb),dim=1);generator=torch.Generator(device='cpu').manual_seed(33417)
    def permute(a):
        perm=torch.randperm(a.shape[1],generator=generator).to(a.device);signs=(2*torch.randint(0,2,(a.shape[1],),generator=generator)-1).to(a.device,a.dtype)
        return a[:,perm]*signs
    _=permute(w);control=torch.cat([permute(b) for b in w.split(128)],dim=0)
    rows={}
    for name,v in [('native',w),('independent_source_permutations',control)]:
        start=time.perf_counter();found=energies(l,r,o,v,k,9);row={key:float(value) for key,value in found.items()}
        row.update(antisymmetric_fraction=row['antisymmetric']/row['total'],head_diagonal_fraction=row['head_diagonal']/row['total'],seconds=time.perf_counter()-start)
        rows[name]=row;print(json.dumps(dict(map=name,result=row)),flush=True)
    a=rows['native'];b=rows['independent_source_permutations']
    errors=dict(total_invariance=abs(b['total']/a['total']-1),head_diagonal_invariance=abs(b['head_diagonal']/a['head_diagonal']-1),energy_closure=max(abs(x['symmetric']+x['antisymmetric']-x['total'])/x['total'] for x in rows.values()),negative_energy=max(max(0.,-x['symmetric']/x['total'],-x['antisymmetric']/x['total']) for x in rows.values()))
    valid=max(errors.values())<=1e-10 and all(math.isfinite(v) for x in rows.values() for v in x.values());difference=a['antisymmetric_fraction']-b['antisymmetric_fraction']
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_substantial_single_source_kernel':valid and a['antisymmetric_fraction']>=.10,'pred_c_source_alignment':valid and abs(difference)>=.05},maps=rows,antisymmetric_fraction_difference=difference,native_value_mix=mix,errors=errors,price=dict(body_forwards=0,output_rows=len(u),native_products=len(l),output_rank_truncation=False,source_tuple_width=2304,persistent_large_arrays=0),wall_seconds=time.perf_counter()-tic,scope='Full centered output tensor, exact weight contractions. One-source coefficient kernel may be live across sources. No task/data/causal or full-circuit-extraction claim; all interfaces and common output remain required.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
