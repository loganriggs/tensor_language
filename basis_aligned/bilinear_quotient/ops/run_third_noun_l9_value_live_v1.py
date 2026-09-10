#!/usr/bin/env python3
# BQGATE: weight-compiled local-value mixed removal, live suffix, no fitted sources.
"""A instrument; B partial materiality; C direct carry; D factor selectivity."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(POLY));sys.path.insert(0,str(ROOT))
import circuit_fast_screen_producer as P
import value_lineage_capture as C
import norm_preserving_response_hooks as H
import source_margin_gradient as G
import rotary_phase_capture as R
import mixed_state_projector_v1 as Q
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'THIRD_NOUN_L9_VALUE_LIVE_V1_BINDING.json'
OUT=POLY/'THIRD_NOUN_L9_VALUE_LIVE_V1_RESULT.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_ROWS.json').read_text())
    parent=json.loads((POLY/'THIRD_NOUN_ANIMACY_V1_RESULT.json').read_text())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'dryrun':True,'gpu_accessed':False,'model_loaded':False,'forwards':48,
                          'sequences':768,'decoder_batches':32,'decoder_states':512}));return
    assert not OUT.exists();signal.alarm(600)
    backend=P.Bilin18TorchBackend.load('cuda');torch=backend.torch;torch.set_num_threads(2)
    import torch.nn.functional as F
    tic=time.perf_counter();counts=[0,0];decoders=0;record={};baselines=[];zs=[];us=[];values=[]
    rows=[r for w in source['worlds'] for r in w['rows']];attn=backend.model.transformer.h[9].attn
    def count(_m,args,_y):counts[0]+=1;counts[1]+=len(args[0])
    def first(_m,_args,y):record['first']=y[1].detach().clone()
    handles=[backend.model.transformer.h[0].attn.register_forward_hook(count),attn.register_forward_hook(first)]
    def batch_for(start):
        part=rows[start:start+16]
        return P.ModelBatch(tuple(r['row_id'] for r in part),'base',tuple(tuple(r['ids']) for r in part),
            (source['answer_id'],)*16,(source['foil_id'],)*16,(9,)*16)
    def run(batch):
        with H.capture(backend.model,layers=(9,)) as factors:
            with G.capture(backend.model) as log:out=backend.native(batch,capture=True)
        return out,G.endpoint_logits(log['raw_logits'],batch).double(),factors[9]['factors'],record['first'].clone()
    try:
        with torch.inference_mode():
            for start in range(0,256,16):
                batch=batch_for(start)
                with C.capture(backend.model,source=8,target=9) as capture:b=run(batch)
                baselines.append(b);zs.append(b[1]);us.append(capture['normalized_input']);values.append(capture['value'])
            q=torch.tensor(Q.projector(source['corners'],2,4),device='cuda',dtype=torch.float64)
            def mixed(x):return torch.einsum('ij,wj...->wi...',q,x)
            u=torch.cat(us).view(8,32,10,1152).double();local=torch.cat(values).view_as(u).double()
            delta=mixed(u)@attn.c_v.weight.double().T
            commute=float((delta-mixed(local)).norm()/mixed(local).norm().clamp_min(1e-30))
            early=float(mixed(u)[:,:,:7].norm()/u.norm());replacement=(local-delta).float().reshape(256,10,9,128)
            delta_flat=delta.reshape(256,10,9,128);edited=[];identity_exact=True;qk_same=True;first_same=True;write_errors=[]
            final_base=[];write_deltas=[]
            for bi,start in enumerate(range(0,256,16)):
                batch=batch_for(start);base=baselines[bi]
                with C.replace_value(attn,values[bi].view(16,10,9,128),(10,)*16,heads=tuple(range(9))):identity=run(batch)
                identity_exact &= torch.equal(identity[1],base[1])
                with C.replace_value(attn,replacement[start:start+16],(10,)*16,heads=tuple(range(9))):change=run(batch)
                edited.append(change[1]);first_same &= torch.equal(base[3],identity[3]) and torch.equal(base[3],change[3])
                qk_same &= all(torch.equal(base[2][i],change[2][i]) and torch.equal(base[2][i],identity[2][i]) for i in (0,1,3,4))
                base_write=torch.stack([base[0].captured[r,'attn:09'] for r in batch.row_ids]).double()
                change_write=torch.stack([change[0].captured[r,'attn:09'] for r in batch.row_ids]).double()
                observed=base_write-change_write;write_deltas.append(observed)
                factors=[v.double() for v in base[2]];factors[2]=(1-float(attn.lamb))*delta_flat[start:start+16]
                predicted=R.final_read(*factors,positions=batch.semantic_positions).reshape(16,1152)@attn.c_proj.weight.double().T
                write_errors.append(float((observed-predicted).norm()/predicted.norm().clamp_min(1e-30)))
                final_base.append(torch.stack([base[0].captured[r,'resid:18'] for r in batch.row_ids]).double())
            z0=torch.cat(zs).view(8,32,-1);z1=torch.cat(edited).view_as(z0)
            final=torch.cat(final_base);carry=1.
            for block in backend.model.transformer.h[10:]:carry*=float(block.lambdas[0])
            direct=final-carry*torch.cat(write_deltas);offline={}
            for arm,x in [('replay',final),('direct',direct)]:
                chunks=[]
                for part in x.float().split(16):
                    chunks.append((30*torch.tanh(backend.model.lm_head(F.rms_norm(part,(1152,)))/30)).double());decoders+=1
                offline[arm]=torch.cat(chunks).view_as(z0)
            aid,fid=source['answer_id'],source['foil_id']
            margins={k:v[:,:,aid]-v[:,:,fid] for k,v in [('native',z0),('edited',z1),('direct',offline['direct'])]}
            def bridge(x,y):return {'max_abs':float((x-y).abs().max()),'relative':float((x-y).norm()/y.norm().clamp_min(1e-30))}
            bridges={'parent':bridge(margins['native'],torch.tensor(parent['margins'],device='cuda',dtype=torch.float64)),
                     'decoder':bridge(offline['replay'],z0)}
            natural=mixed(margins['native']);change=margins['native']-margins['edited'];live=mixed(change)
            predicted=mixed(margins['native']-margins['direct']);full=mixed(z0-z1);directfull=mixed(z0-offline['direct'])
            full=full-full.mean(-1,keepdim=True);directfull=directfull-directfull.mean(-1,keepdim=True)
            reports=[]
            for w in range(8):
                projection=float((live[w]*natural[w]).sum()/natural[w].square().sum())
                me=float((predicted[w]-live[w]).norm()/live[w].norm().clamp_min(1e-30))
                ve=float((directfull[w]-full[w]).norm()/full[w].norm().clamp_min(1e-30))
                spill=float((change[w]-live[w]).norm()/live[w].norm().clamp_min(1e-30))
                reports.append({'world_id':source['worlds'][w]['world_id'],'live_natural_mixed_projection':projection,
                    'direct_margin_relative_error':me,'direct_full_vector_relative_error':ve,'nonmixed_to_mixed_margin_ratio':spill,
                    'materiality_passed':projection>=.05,'direct_passed':me<=.10 and ve<=.10,'factor_selectivity_passed':spill<=.25})
            finite=all(bool(x.isfinite().all()) for x in [z0,z1,offline['direct']])
    finally:
        for h in handles:h.remove()
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in backend.model.modules()) and all('squared_attention' not in b.attn.__dict__ for b in backend.model.transformer.h)
    a=restored and finite and counts==[48,768] and decoders==32 and identity_exact and qk_same and first_same and commute<=1e-4 and early<=1e-6 and max(write_errors)<=1e-4 and all(
        b['max_abs']<=1e-3 and b['relative']<=1e-5 for b in bridges.values())
    preds={'pred_a_instrument':a,'pred_b_partial_materiality':all(r['materiality_passed'] for r in reports),
           'pred_c_direct_carry':all(r['direct_passed'] for r in reports),'pred_d_factor_selectivity':all(r['factor_selectivity_passed'] for r in reports)}
    result={'terminal':'live_value_complete' if a else 'invalid','predictions':preds,'reports':reports,'bridges':bridges,
        'identity_exact':identity_exact,'qk_unchanged':qk_same,'first_value_unchanged':first_same,
        'weight_commutation_relative':commute,'early_input_mixed_relative':early,'local_write_oracle_relative':write_errors,
        'margins':{k:v.cpu().tolist() for k,v in margins.items()},'corners':source['corners'],
        'runner_sha256':sha(RUNNER),'binding_sha256':sha(BINDING),'wall_seconds':time.perf_counter()-tic,
        'price':{'forwards':counts[0],'sequences':counts[1],'decoder_batches':decoders,'decoder_states':512,
            'fits':0,'native_parameters':sum(p.numel() for p in backend.model.parameters()),'actual_weight_saving':0},
        'scope':'Native local-value interface removal and full downstream recomputation, using four-corner native normalized inputs. Partial causal contribution only; no independent extraction/OOD program or unrelated-task selectivity.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':preds,'reports':reports,'bridges':bridges,'commutation':commute,'max_write_oracle':max(write_errors),'wall_seconds':result['wall_seconds']}));assert a


if __name__=='__main__':main()
