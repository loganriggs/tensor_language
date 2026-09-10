#!/usr/bin/env python3
# BQGATE: token and 16-leaf weight readers; fixed MLP16 swap;12forwards192seq; no outcome fitting.
"""pred_a exact conditional fold; pred_b native capability/nonzero swap;
pred_c live full-vocabulary error<=.10; pred_d hierarchy error<=.10 and
beats root/random by .10; pred_e live per-row CE MAE<=.02.
All native weights retained. Intervening attention is explicitly tested.
"""
import json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(POLY),str(ROOT)]
import torch
import torch.nn.functional as F
import circuit_fast_screen_producer as P
import circuit_unit_greedy as g
import unembedding_backward_views_v1 as C
from circuit_endpoint_capability_v1 import summarize
from induction_context_transport_v2 import digest
from circuit_fast_screen_managed_runner import atomic_create_json
OUT=POLY/'UNEMBEDDING_BACKWARD_VIEWS_V1_RESULT.json'
BINDING=POLY/'UNEMBEDDING_BACKWARD_VIEWS_V1_BINDING.json'
ARTIFACT=POLY/'UNEMBEDDING_BACKWARD_VIEWS_V1_HIERARCHY.pt'


def serial(x):return x.detach().cpu().tolist()
def center(x):return x-x.mean(-1,keepdim=True)
def margin(x):return x[:,0]-x[:,1]
def bridge(a,b):
    a=a.double();b=b.double();err=a-b
    return {'max_abs':float(err.abs().max()),'relative_l2':float(err.norm()/b.norm().clamp_min(1e-30)),
            'scaled_error':float((err.abs()/(1e-3+1e-5*b.abs())).max())}


def main():
    binding=json.loads(BINDING.read_text());assert all(digest(p)==h for p,h in binding.items())
    panels=json.loads((POLY/'CORRELATIVE_RECOMBINED_ROWS_V1.json').read_text())['panels']
    token_ids=set(range(0,50304,98))
    for name in ('A1','A2','C'):
        assert len(panels[name])==16
        for row in panels[name]:
            token_ids.update(row[k] for k in ('base_answer_id','base_foil_id','donor_answer_id','donor_foil_id'))
            assert all(row[f'{side}_semantic_position']==len(row[f'{side}_ids'])-1 for side in ('base','donor'))
    token_ids=sorted(token_ids);controls=C.controls()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps({'model_loaded':False,'gpu_accessed':False,'body_forwards':12,'sequences':192,
                          'token_readers':len(token_ids),'hierarchy_leaves':16,'controls':controls}));return
    assert not OUT.exists() and not ARTIFACT.exists();signal.alarm(900);tic=time.perf_counter()
    backend=P.Bilin18TorchBackend.load('cuda');model=backend.model;torch.set_num_threads(2)
    assert not getattr(model.config,'gated',False)
    U=model.lm_head.weight.float();m16=model.transformer.h[16].mlp;m17=model.transformer.h[17].mlp
    d16=m16.Down.weight.float();d17=m17.Down.weight.float();l17=m17.Left.weight.float();r17=m17.Right.weight.float()
    bias=m17.Down_bias.float();coefficient=float(model.transformer.h[17].lambdas[0]);eps=torch.finfo(torch.float32).eps
    counts=[0,0];reports={};finite=True
    def count(_m,args):
        counts[0]+=1;counts[1]+=len(args[0])
        if counts[0]>12 or counts[1]>192:raise RuntimeError('Frozen body price exceeded')
    counter=model.transformer.h[0].attn.register_forward_pre_hook(count)

    def body(batch,donor=None,fixed_attention=None):
        cap={};raw={};hidden={};idx=torch.arange(len(batch.row_ids),device='cuda');pos=torch.tensor(batch.semantic_positions,device='cuda')
        def output16(_m,args,out):cap['m16']=out[idx,pos].detach().clone()
        def output17(_m,args,out):cap['m17']=out[idx,pos].detach().clone()
        def attention17(_m,args,out):
            cap['attention17']=out[0][idx,pos].detach().clone()
            if fixed_attention is None:return out
            changed=out[0].clone();changed[idx,pos]=fixed_attention
            return changed,out[1]
        hooks=[m16.register_forward_hook(output16),m17.register_forward_hook(output17),
               model.transformer.h[17].attn.register_forward_hook(attention17)]
        try:
            af,z=g.forward_units(backend,batch,units=() if donor is None else ('mlp:16',),
                donor_cache=donor,return_logits=True,capture_resid=raw,capture_hidden=hidden)
            cap['raw17']=torch.stack([raw[(rid,17)] for rid in batch.row_ids])
            cap['products16']=torch.stack([hidden[(rid,g.hidden_key(16))] for rid in batch.row_ids])
            cap['final']=cap['raw17']+cap['m17']
            return af,z,cap
        finally:
            for hook in hooks:hook.remove()

    try:
        with torch.inference_mode():
            labels,means,tree=C.hierarchy(U)
            generator=torch.Generator().manual_seed(9111340)
            permutation=torch.randperm(len(U),generator=generator).to('cuda');random_labels=labels[permutation]
            random_means=torch.stack([U[random_labels==k].mean(0) for k in range(16)])
            random_means=random_means/random_means.norm(dim=1,keepdim=True).clamp_min(1e-30)*means.norm(dim=1,keepdim=True)
            root=U.mean(0,keepdim=True)
            ids=torch.tensor(token_ids,device='cuda');nt=len(token_ids)
            readers=torch.cat([U[ids],means,random_means,root])
            maps=C.compile_maps(readers,l17,r17,d17,d16,bias)
            representatives=[]
            for k in range(16):
                members=torch.where(labels==k)[0];unit=U[members]/U[members].norm(dim=1,keepdim=True).clamp_min(1e-30)
                nearest=(unit@F.normalize(means[k],dim=0)).topk(min(12,len(members))).indices
                representatives.append({'leaf':k,'size':len(members),'token_ids':serial(members[nearest])})
            torch.save({'labels':labels.cpu(),'raw_centroids':means.cpu(),'tree':tree,
                        'random_labels':random_labels.cpu(),'random_centroids_norm_matched':random_means.cpu(),
                        'root':root.cpu(),'token_readers':token_ids,'representatives':representatives,
                        'scope':'Weight-only hierarchy; no semantic names or outcome selection.'},ARTIFACT)
            for name in ('A1','A2','C'):
                rows=panels[name];base=g.batch_of(rows,'base');donor_batch=g.batch_of(rows,'donor')
                ba,bz,bc=body(base);da,dz,dc=body(donor_batch)
                donor={(rid,'mlp:16'):dc['m16'][i] for i,rid in enumerate(base.row_ids)}
                la,lz,lc=body(base,donor=donor)
                fa,fz,fc=body(base,donor=donor,fixed_attention=bc['attention17'])
                background=bc['raw17']-coefficient*(bc['products16']@d16.T)
                final,score=C.folded_state(background,dc['products16'],coefficient,readers,maps,l17,r17,d17,d16,bias,eps)
                normed=F.rms_norm(final,(1152,),eps=eps);base_normed=F.rms_norm(bc['final'],(1152,),eps=eps)
                compiled=30*torch.tanh(F.linear(normed,U)/30)
                token_compiled=30*torch.tanh(score[:,:nt]/30)
                base_raw=F.linear(base_normed,U);delta=normed-base_normed
                hierarchy=30*torch.tanh((base_raw+(delta@means.T)[:,labels])/30)
                shuffled=30*torch.tanh((base_raw+(delta@random_means.T)[:,random_labels])/30)
                global_only=30*torch.tanh((base_raw+delta@root.T)/30)
                token_residual=U[ids]-means[labels[ids]]
                recovered_score=score[:,nt:nt+16][:,labels[ids]]+normed@token_residual.T
                checks={'state':bridge(final,fc['final']),'compiled_full_logits':bridge(compiled,fz),
                        'token_logits':bridge(token_compiled,fz[:,ids]),
                        'group_plus_residual':bridge(recovered_score,score[:,:nt]),
                        'native_final_state':bridge(30*torch.tanh(base_raw/30),bz),
                        'donor_product_write':bridge(dc['products16']@d16.T+m16.Down_bias,dc['m16'])}
                live_effect=center(lz.double()-bz.double());den=live_effect.norm().clamp_min(1e-30)
                fixed_effect=center(fz.double()-bz.double());fixed_den=fixed_effect.norm().clamp_min(1e-30)
                target=torch.tensor(base.answer_ids,device='cuda');base_ce=F.cross_entropy(bz,target,reduction='none')
                live_ce=F.cross_entropy(lz,target,reduction='none')-base_ce
                arms={}
                for key,z in [('folded_token_program',compiled),('hierarchy',hierarchy),('shuffled',shuffled),('root',global_only)]:
                    err=center(z.double()-lz.double());fixed_err=center(z.double()-fz.double())
                    ce=F.cross_entropy(z,target,reduction='none')-base_ce
                    arms[key]={'live_error_relative':float(err.norm()/den),'fixed_error_relative':float(fixed_err.norm()/fixed_den),
                               'live_error_squared_per_row':serial(err.square().sum(-1)),
                               'ce_change_per_row':serial(ce),'live_ce_mae':float((ce-live_ce).abs().mean())}
                native_margin=margin(ba);donor_margin=margin(da);cue_den=native_margin+donor_margin
                reports[name]={'capability':summarize(serial(native_margin),serial(donor_margin)),
                    'positive_cue_denominators':bool((cue_den>1e-6).all()),'bridges':checks,'arms':arms,
                    'live_effect_norm':float(den),'fixed_effect_norm':float(fixed_den),
                    'live_effect_squared_per_row':serial(live_effect.square().sum(-1)),
                    'live_ce_change_per_row':serial(live_ce),
                    'live_mlp16_swap_recovery':float(((native_margin-margin(la))/cue_den).mean()),
                    'fixed_mlp16_swap_recovery':float(((native_margin-margin(fa))/cue_den).mean()),
                    'native_base_margin':serial(native_margin),'native_donor_margin':serial(donor_margin)}
                finite=finite and all(bool(torch.isfinite(v).all()) for v in (final,score,bz,dz,lz,fz,compiled,hierarchy,shuffled,global_only))
    finally:counter.remove()
    good=lambda x:x['max_abs']<=1e-3 and x['relative_l2']<=1e-5
    instrument=counts==[12,192] and finite and all(
        r['bridges']['state']['relative_l2']<=1e-5 and r['bridges']['donor_product_write']['scaled_error']<=1
        and all(good(r['bridges'][k]) for k in ('compiled_full_logits','token_logits','group_plus_residual','native_final_state')) for r in reports.values())
    capable=all(r['capability']['both_endpoints_correct']==16 and r['positive_cue_denominators'] and r['live_effect_norm']>1e-4 for r in reports.values())
    token=instrument and capable and all(r['arms']['folded_token_program']['live_error_relative']<=.10 for r in reports.values())
    hierarchy_pass=instrument and capable and all(r['arms']['hierarchy']['live_error_relative']<=.10 and
        all(r['arms'][k]['live_error_relative']-r['arms']['hierarchy']['live_error_relative']>=.10 for k in ('root','shuffled')) for r in reports.values())
    predictions={'pred_a_instrument':bool(instrument),'pred_b_native_capability':bool(capable),
                 'pred_c_further_fold_live':bool(token),'pred_d_shared_hierarchy':bool(hierarchy_pass),
                 'pred_e_signed_manipulation':bool(token and all(r['arms']['folded_token_program']['live_ce_mae']<=.02 for r in reports.values()))}
    result={'schema':'unembedding.backward_views.v1','predictions':predictions,'reports':reports,
            'token_reader_ids':token_ids,'hierarchy':representatives,'hierarchy_artifact_sha256':digest(ARTIFACT),
            'controls':controls,'price':{'body_forwards':counts[0],'sequences':counts[1],
                'native_parameters':sum(p.numel() for p in model.parameters()),'additional_compiled_map_scalars':sum(v.numel() for v in maps.values()),
                'hierarchy_label_scalars':len(labels),'hierarchy_centroid_scalars':means.numel(),'native_weight_saving':0},
            'wall_seconds':time.perf_counter()-tic,'runner_sha256':digest(RUNNER),'binding_sha256':digest(BINDING),
            'scope':'Existing authored text reused. Exact two-MLP fold conditional on native attention17 background, evaluated against fixed and live attention. Hierarchy from all unembedding weights; no new semantic circuit or OOD claim.'}
    atomic_create_json(OUT,result)
    print(json.dumps({'predictions':predictions,'price':result['price'],'wall_seconds':result['wall_seconds'],
          'reports':{n:{'recovery':r['live_mlp16_swap_recovery'],'arms':{k:{kk:v for kk,v in a.items() if not kk.endswith('per_row')} for k,a in r['arms'].items()},'bridges':r['bridges']} for n,r in reports.items()}},indent=2))


if __name__=='__main__':main()
