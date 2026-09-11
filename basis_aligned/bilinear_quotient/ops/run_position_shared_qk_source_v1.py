#!/usr/bin/env python3
# BQGATE: 0forwards0seq; exact weight coefficient statistics across511positions.
import os,sys,json,time,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from joint_qk_position_influence_v1 import coefficient
from joint_qk_position_space_v1 import norm,ports
from joint_qk_source_ports_v1 import source_ports
from folded_normalized_router_v1 import rotary
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding=json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_BINDING.json').read_text());assert all(digest(p)==h for p,h in binding.items())
    assert json.loads((P/'JOINT_QK_POSITION_SPACE_V1_CONTROL.json').read_text())['instrument_passed']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=0,corpus_access=False,
                             heads=9,source_positions=511,discovery_positions=256,validation_positions=255,rank=17)));return
    out=P/'POSITION_SHARED_QK_SOURCE_V1_RESULT.json';assert not out.exists();signal.alarm(900)
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    q1,k1,q2,k2=[sd[f'transformer.h.17.attn.{key}.weight'].double().cuda().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    rs=torch.stack([rotary(p,128) for p in range(511)]).cuda();rt=rotary(511,128).cuda()
    rotations=rt.T@rs;indices=torch.arange(511,device='cuda');train=indices%2==0;validation=~train
    sampled=torch.linspace(0,510,32).round().long().cuda();rows=[];frames=[];validation_frames=[];errors=[]
    for h in range(9):
        head_start=time.perf_counter();stack=torch.cat((k1[h],k2[h]),dim=0)
        key_basis=torch.linalg.qr(stack.T).Q;compressed=stack@key_basis
        base_grams=(q1[h]@q1[h].T,q2[h]@q2[h].T,q1[h]@q2[h].T)
        grams=tuple(rotations.transpose(-1,-2)@g@rotations for g in base_grams)
        keys=(k1[h]@k1[h].T,k2[h]@k2[h].T,k1[h]@k2[h].T)
        c=coefficient(*grams,*keys);energy=norm(*grams,*keys)
        trace=(c*(stack@stack.T).T).sum((-2,-1));errors.append(float(((trace-energy)/energy).abs().max()))
        normalized=c/energy[:,None,None]
        def frame(mask):
            influence=compressed.T@normalized[mask].mean(0)@compressed
            influence=(influence+influence.T)/2
            eigen,vectors=torch.linalg.eigh(influence)
            errors.append(abs(float(influence.trace())-1))
            errors.append(max(0.,-float(eigen[0])))
            basis=key_basis@vectors[:,-17:]
            errors.append(float((basis.T@basis-torch.eye(17,device='cuda')).abs().max()))
            return basis,min(1.,float(2*eigen[-17:].sum()))
        common,upper=frame(train)
        result=ports(grams,k1[h],k2[h],common)
        other,_=frame(validation)
        overlap=torch.linalg.svdvals(common.T@other).square().mean()
        separate=[]
        for pos in sampled.tolist():
            influence=compressed.T@normalized[pos]@compressed
            _,v=torch.linalg.eigh((influence+influence.T)/2);individual=key_basis@v[:,-17:]
            separate.append(ports(tuple(g[pos] for g in grams),k1[h],k2[h],individual)['touch'])
        separate_mean=float(torch.stack(separate).mean());shared_mean=float(result['touch'][sampled].mean())
        # Direct single-position contraction with rotated key maps, independent
        # of the batched rotated-query-Gram implementation.
        checks=source_ports(q1[h],q2[h],rotations[7]@k1[h],rotations[7]@k2[h],common)
        errors.append(abs(float(checks['total']/result['total'][7])-1))
        for key in ['inside','mixed','outside']:
            errors.append(abs(float(checks[key]/checks['total']-result[key][7])))
            errors.append(max(0.,-float(result[key].min()),float(result[key].max())-1))
        row=dict(head=h,discovery_mean_touch=float(result['touch'][train].mean()),
                 discovery_mean_touch_upper_bound=upper,validation_mean_touch=float(result['touch'][validation].mean()),
                 validation_minimum_touch=float(result['touch'][validation].min()),
                 validation_inside_mean=float(result['inside'][validation].mean()),validation_mixed_mean=float(result['mixed'][validation].mean()),
                 sampled_common_mean_touch=shared_mean,sampled_separate_mean_touch=separate_mean,
                 common_to_separate_ratio=shared_mean/separate_mean,
                 split_mean_squared_principal_cosine=float(overlap),seconds=time.perf_counter()-head_start,
                 all_position_touch=result['touch'].cpu().tolist())
        rows.append(row);frames.append(common.cpu());validation_frames.append(other.cpu())
        print(json.dumps({k:v for k,v in row.items() if k!='all_position_touch'}),flush=True)
    cache=Path('/dev/shm/bilin18_position_shared_qk_source_v1.pt');assert not cache.exists()
    torch.save(dict(frames=torch.stack(frames),validation_frames=torch.stack(validation_frames),query_position=511,
                    source_positions=list(range(511)),discovery_sources=list(range(0,511,2))),cache)
    valid=max(errors)<1e-10
    result=dict(predictions={'pred_a_instrument':valid,
                            'pred_b_shared_retention':valid and all(r['common_to_separate_ratio']>=.9 for r in rows),
                            'pred_c_validation_coverage':valid and all(r['validation_mean_touch']>=.45 and r['validation_minimum_touch']>=.3 for r in rows),
                            'pred_d_split_stability':valid and all(r['split_mean_squared_principal_cosine']>=.95 for r in rows)},
                heads=rows,maximum_instrument_error=max(errors),wall_seconds=time.perf_counter()-start,
                cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
                price=dict(body_forwards=0,corpus_access=False,source_frame_numbers=9*1152*17,heads=9,source_positions=511,
                           discovery_positions=256,validation_positions=255,per_position_comparators=32,native_norm_value_background_required=True),
                scope='One weight-derived source space per head shared across distances. Equal-position influence objective, true numerator touch scored separately. Validation distances excluded from primary frame selection; no natural-data/OOD circuit or normalized-routing replacement claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='heads'}),flush=True)


if __name__=='__main__':main()
