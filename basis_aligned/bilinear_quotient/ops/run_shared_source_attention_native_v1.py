#!/usr/bin/env python3
# BQGATE: 0forwards0seq; four fixed quadratic forms, exact source-coherence norms.
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from shared_source_attention_quadratic_v1 import symmetry_energies_low_rank
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    binding=json.loads((P/'SHARED_SOURCE_ATTENTION_NATIVE_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'SHARED_SOURCE_ATTENTION_QUADRATIC_V1_CONTROL.json').read_text())['passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,forms=[0,1,4,2],heads=9,source_tuple_width=2304)));return
    out=P/'SHARED_SOURCE_ATTENTION_NATIVE_V1_RESULT.json';assert not out.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r=[sd[f'transformer.h.17.mlp.{k}.weight'].double().cuda() for k in ['Left','Right']]
    o=sd['transformer.h.17.attn.c_proj.weight'].double().cuda();vc=sd['transformer.h.17.attn.c_v.weight'].double().cuda();vb=sd['transformer.h.0.attn.c_v.weight'].double().cuda();mix=float(sd['transformer.h.17.attn.lamb'])
    w=torch.cat(((1-mix)*vc,mix*vb),dim=1)
    s=torch.load(P/'OUTPUT_VARIMAX_V1_CHECKPOINT.pt',weights_only=False,map_location='cuda');core=s['rotation'].T@s['core']
    generator=torch.Generator(device='cpu').manual_seed(33417)
    def permute(a):
        perm=torch.randperm(a.shape[1],generator=generator).to(a.device)
        signs=(2*torch.randint(0,2,(a.shape[1],),generator=generator)-1).to(device=a.device,dtype=a.dtype)
        return a[:,perm]*signs
    common=permute(w);independent=torch.cat([permute(block) for block in w.split(128)],dim=0)
    rows=[];valid=True
    for index in [0,1,4,2]:
        q=(l.T*core[index])@r;q=(q+q.T)/2
        native=symmetry_energies_low_rank(q,o,w,9)
        fixed=symmetry_energies_low_rank(q,o,common,9)
        scrambled=symmetry_energies_low_rank(q,o,independent,9)
        converted={name:{k:float(v) for k,v in values.items()} for name,values in [('native',native),('common_permutation',fixed),('independent_permutations',scrambled)]}
        norm=float(q.square().sum());scale=converted['native']['total']
        errors=dict(q_norm=abs(norm-1),common_permutation=max(abs(converted['native'][k]-converted['common_permutation'][k])/scale for k in native),independent_total=abs(converted['independent_permutations']['total']/scale-1),energy_closure=max(abs(v['symmetric']+v['antisymmetric']-v['total'])/v['total'] for v in converted.values()),negative_energy=max(max(0.,-v['symmetric']/v['total'],-v['antisymmetric']/v['total']) for v in converted.values()))
        valid=valid and max(errors.values())<=1e-10
        nf=float(native['antisymmetric']/native['total']);cf=float(scrambled['antisymmetric']/scrambled['total'])
        row=dict(form=index,energies=converted,antisymmetric_fraction=nf,independent_permutation_fraction=cf,difference=nf-cf,errors=errors)
        rows.append(row);print(json.dumps(row),flush=True)
    mean_difference=sum(v['difference'] for v in rows)/len(rows)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_substantial_single_source_kernel':valid and all(v['antisymmetric_fraction']>=.10 for v in rows),'pred_c_source_alignment':valid and abs(mean_difference)>=.05},forms=rows,mean_fraction_difference=mean_difference,native_value_mix=mix,price=dict(body_forwards=0,forms=4,head_count=9,head_width=128,source_tuple_width=2304,full_lifted_matrix_width=20736,persistent_large_arrays=0),wall_seconds=time.perf_counter()-tic,scope='Four fixed native output-combination quadratics. Antisymmetric coefficient directions vanish for one source but may be live with multiple sources; no data, circuit significance, full-tensor estimate or global simplification claim.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
