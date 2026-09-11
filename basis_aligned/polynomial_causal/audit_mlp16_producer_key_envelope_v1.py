"""Exact full-key-space envelope for frozen OV functions; CPU, no text fitting."""
import hashlib,json,time
from pathlib import Path
import torch
from producer_function_overlap_v1 import overlap
from producer_function_pairs_v1 import pairs

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter()
    prior=json.loads((P/'MLP16_PRODUCER_OVERLAP_V1_RESULT.json').read_text())
    cache=prior['cache'];assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    state=torch.load(cache['path'],weights_only=True,map_location='cpu');g=state['metric'];b=state['ov_readers']
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    k1=sd['transformer.h.17.attn.c_k.weight'].double();k2=sd['transformer.h.17.attn.c_k2.weight'].double()
    rows=[];frames=[];errors=[]
    for h in range(9):
        keys=torch.cat((k1[h*128:(h+1)*128],k2[h*128:(h+1)*128]),0)
        _,sing,right=torch.linalg.svd(keys,full_matrices=False)
        rank=int((sing>sing[0]*1e-10).sum());basis=right[:rank]
        raw=overlap(basis,b[h],torch.eye(1152))
        left,right_functions,cosines=pairs(basis,b[h],g)
        # The17left canonical functions span an optimal17-dimensional key space
        # for this subspace-alignment objective. Orthonormalize raw coordinates
        # for use as a later input-removal projector.
        frame=torch.linalg.qr(left.T).Q
        selected=overlap(frame.T,b[h],g)
        target=cosines.square().mean()
        errors.extend([abs(selected['mean_squared_cosine']-float(target)),
            float((frame-basis.T@(basis@frame)).norm()),
            float((left@g@right_functions.T-torch.diag(cosines)).norm()),
            float((left@g@left.T-torch.eye(17)).norm())])
        old=prior['folded'][h]
        row=dict(head=h,joint_key_rank=rank,raw_full_key_mean=raw['mean_squared_cosine'],
                 frozen_qk_mean=old['mean_squared_cosine'],optimal_rank17_mean=float(target),
                 improvement_over_frozen=float(target)/old['mean_squared_cosine'],
                 optimal_function_cosines=cosines.tolist(),
                 maximum_function_cosine=float(cosines[0]),
                 functions_above_point95=int((cosines>=.95).sum()),
                 selected_candidate_mean=selected['mean_squared_cosine'])
        rows.append(row);frames.append(frame);print(json.dumps(row),flush=True)
    target=P/'MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT.json';assert not target.exists()
    artifact=Path('/dev/shm/bilin18_mlp16_producer_key_envelope_v1.pt');assert not artifact.exists()
    torch.save(dict(frames=torch.stack(frames),source_cache_sha256=cache['sha256']),artifact)
    result=dict(instrument_passed=max(errors)<1e-8,heads=rows,
        mean_optimal_overlap=sum(r['optimal_rank17_mean'] for r in rows)/9,
        mean_frozen_overlap=prior['summary']['folded_mean'],
        heads_with_any_cosine_above_point95=sum(r['functions_above_point95']>0 for r in rows),
        maximum_instrument_error=max(errors),body_forwards=0,corpus_access=False,
        cache=dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),bytes=artifact.stat().st_size,ephemeral=True),
        wall_seconds=time.perf_counter()-start,
        redteam=dict(failed_claim='These frozen17QKsource features strongly share MLP16producer functions with the frozenOVfeatures.',
            plausible_method_limit='The originalQKfeatures optimize joint-routing influence, not producer/value alignment.',
            executed_check='Exact optimal canonical function alignment over each entire jointkeyrowspace;17canonical key readers achieve its full17function envelope.',
            limits='Coupled choice optimizes only producer-function overlap; actual jointQK1xQK2 influence, normalizers, base/residual branches, behavior and semantic identity remain untested for these newframes.'),
        scope='Weights-only exact function-space envelope and candidate, not convergence of a tensor fit or circuit discovery. Original registered B/C misses remain.')
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='heads'},indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
