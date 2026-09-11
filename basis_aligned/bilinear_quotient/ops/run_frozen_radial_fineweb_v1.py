#!/usr/bin/env python3
# BQGATE: 10forwards80seq128tokens;64FineWeb rows,8frozenarms,no fitting.
"""pred_a physical replay/counts/finite; pred_b 10% MLP error reduction;
pred_c 10% KL reduction; pred_d radial-only better than bias-only. See protocol.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,hashlib,signal
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3]
P=ROOT/'basis_aligned/polynomial_causal';BQ=RUNNER.parents[1]
sys.path[:0]=[str(RUNNER.parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_reader_program_v1 import SparseReaderProgram
from radial_corrected_reader_v1 import RadialCorrectedReader
DATA=BQ/'.rowcache/fineweb_n192_skip7000.pt'
NAMES=('orthogonal0','orthogonal937','ordinary0')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding=json.loads((P/'FROZEN_RADIAL_FINEWEB_V1_BINDING.json').read_text())
    assert all(digest(path)==sha for path,sha in binding['files'].items())
    rows=torch.load(DATA,weights_only=True,map_location='cpu')[:64,:129].contiguous()
    assert rows.shape==(64,129) and rows.dtype==torch.long
    hashes=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in rows]
    assert hashes==binding['row_sha256'] and len(set(hashes))==64
    radial=json.loads((P/'FROZEN_READER_RADIAL_V1_AUDIT.json').read_text())
    assert [a['name'] for a in radial['arms']]==list(NAMES)
    for arm in radial['arms']:
        assert digest(arm['base']['path'])==arm['base']['sha256']
        assert digest(arm['cache']['path'])==arm['cache']['sha256']
    assert all(json.loads((P/'RADIAL_CORRECTED_READER_V1_CONTROL.json').read_text())['predictions'].values())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,model_loaded=False,body_forwards=10,sequences=80,
            input_tokens=128,distinct_rows=64,prediction_positions=8192,candidate_arms=8,
            fitting=False,corpus='FineWeb',historically_opened=True)));return
    out=P/'FROZEN_RADIAL_FINEWEB_V1_RESULT.json';assert not out.exists();signal.alarm(900)
    started=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();last=model.transformer.h[17];mlp=last.mlp
    assert not mlp.config.gated and len(model.transformer.h)==18 and mlp.Left.weight.shape==(4608,1152)
    down=mlp.Down.weight.double();bias=mlp.Down_bias.double()
    native_radial=down@(mlp.Left.weight.double()*mlp.Right.weight.double()).sum(1)
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].cuda()
    programs={}
    for arm in radial['arms']:
        saved=torch.load(arm['base']['path'],weights_only=True,map_location='cpu')
        base=SparseReaderProgram.from_artifact(saved,down,bias)
        correction=torch.load(arm['cache']['path'],weights_only=True,map_location='cpu')['delta'].cuda()
        programs[arm['name']]=base
        programs[arm['name']+'_radial']=RadialCorrectedReader(base,correction)
    arm_names=['bias_only','radial_only']+list(programs)
    counts=[0,0]
    def count(module,args):
        counts[0]+=1;counts[1]+=len(args[0]);assert args[0].shape[1:]==(128,1152) and counts[0]<=10
    handle=model.transformer.h[0].attn.register_forward_pre_hook(count)
    def candidate(name,x):
        if name=='bias_only':return bias.expand_as(x).float()
        if name=='radial_only':return (x.double().square().sum(-1,keepdim=True)/1152*native_radial+bias).float()
        return programs[name](x.double()).float()
    def logits(h):return 30*torch.tanh(model.lm_head(F.rms_norm(h,(1152,)))/30)
    def prefix(tokens):
        x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
        for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
        x=last.lambdas[0]*x+last.lambdas[1]*x0
        attention,v1=last.attn(F.rms_norm(x,(1152,)),v1)
        pre=x+attention;xin=F.rms_norm(pre,(1152,));native=mlp(xin)
        return xin,pre,native
    metrics={name:[] for name in arm_names};stored=[];controls=[];input_scope=[]
    try:
        for offset in range(0,64,8):
            tokens=rows[offset:offset+8,:128].contiguous().cuda()
            target=rows[offset:offset+8,1:129].contiguous().cuda()
            xin,pre,native=prefix(tokens)
            if offset==0:
                for control_name in ('native','orthogonal0_radial'):
                    capture={}
                    def head_hook(module,args,output):capture['raw']=output.detach()
                    def mlp_hook(module,args,output):
                        input_scope.append(float((args[0]-xin).norm()/xin.norm()))
                        return candidate(control_name,args[0]) if control_name!='native' else output
                    hooks=[model.lm_head.register_forward_hook(head_hook),mlp.register_forward_hook(mlp_hook)]
                    try:physical_ce=float(model(tokens,target))
                    finally:
                        for hook in hooks:hook.remove()
                    physical=30*torch.tanh(capture['raw']/30)
                    q=native if control_name=='native' else candidate(control_name,xin)
                    manual=torch.cat([logits((pre+q).reshape(-1,1152)[i:i+256]) for i in range(0,1024,256)])
                    ce=float(F.cross_entropy(manual,target.reshape(-1)))
                    controls.append(dict(arm=control_name,physical_ce=physical_ce,manual_ce=ce,
                        ce_error=abs(physical_ce-ce),logit_relative_error=float((manual-physical.reshape_as(manual)).norm()/physical.norm())))
                    del capture,physical,q,manual
            xx,pp,qq=xin.reshape(-1,1152),pre.reshape(-1,1152),native.reshape(-1,1152)
            tt=target.reshape(-1);batch={name:[] for name in arm_names}
            for start in range(0,1024,256):
                sl=slice(start,start+256);native_log=logits(pp[sl]+qq[sl]).double().log_softmax(-1)
                native_prob=native_log.exp();native_ce=-native_log.gather(1,tt[sl,None]).squeeze(1)
                native_energy=((qq[sl].double()@metric)*qq[sl].double()).sum(-1)
                for name in arm_names:
                    q=candidate(name,xx[sl]);error=q.double()-qq[sl].double()
                    err_energy=((error@metric)*error).sum(-1)
                    log=logits(pp[sl]+q).double().log_softmax(-1)
                    ce=-log.gather(1,tt[sl,None]).squeeze(1)
                    kl=(native_prob*(native_log-log)).sum(-1)
                    batch[name].append(torch.stack((native_ce,ce,ce-native_ce,kl,err_energy,native_energy),-1).cpu())
            for name in arm_names:metrics[name].append(torch.cat(batch[name]))
            stored.append(dict(input=xin.cpu(),pre=pre.cpu(),native_output=native.cpu()))
            print(json.dumps(dict(rows_completed=offset+8,body_counts=counts)),flush=True)
    finally:handle.remove()
    summaries={};row_records=[];finite=True
    for name in arm_names:
        scores=torch.cat(metrics[name]).reshape(64,128,6);metrics[name]=scores
        finite=finite and bool(torch.isfinite(scores).all())
        mean=scores.mean((0,1));summary=dict(native_ce=float(mean[0]),candidate_ce=float(mean[1]),
            ce_added=float(mean[2]),native_to_candidate_kl=float(mean[3]),
            relative_mlp_error=float(scores[:,:,4].sum()/scores[:,:,5].sum()))
        summaries[name]=summary
        for i,row in enumerate(scores.mean(1)):
            row_records.append(dict(arm=name,row=i,native_ce=float(row[0]),candidate_ce=float(row[1]),
                ce_added=float(row[2]),native_to_candidate_kl=float(row[3]),
                relative_mlp_error=float(row[4]/row[5])))
    comparisons={name:dict(mlp_error_ratio=summaries[name+'_radial']['relative_mlp_error']/summaries[name]['relative_mlp_error'],
        kl_ratio=summaries[name+'_radial']['native_to_candidate_kl']/summaries[name]['native_to_candidate_kl']) for name in NAMES}
    valid=finite and counts==[10,80] and max(input_scope)<=1e-6 and all(c['ce_error']<=1e-5 and c['logit_relative_error']<=1e-6 for c in controls)
    artifact=Path('/dev/shm/bilin18_frozen_radial_fineweb_v1.pt');assert not artifact.exists()
    torch.save(dict(rows=rows,ports={key:torch.cat([s[key] for s in stored]) for key in stored[0]},
        metrics=metrics,metric_columns=['native_ce','candidate_ce','ce_added','kl','mlp_error_energy','native_mlp_energy'],
        scope='Frozen candidate validation states; not authorized as discovery training data.'),artifact)
    result=dict(predictions={'pred_a_instrument':valid,
        'pred_b_mlp_error':valid and all(c['mlp_error_ratio']<=.9 for c in comparisons.values()),
        'pred_c_kl':valid and all(c['kl_ratio']<=.9 for c in comparisons.values()),
        'pred_d_radial_baseline':valid and summaries['radial_only']['native_to_candidate_kl']<summaries['bias_only']['native_to_candidate_kl']},
        summaries=summaries,comparisons=comparisons,physical_controls=controls,maximum_input_scope_error=max(input_scope),
        rows=row_records,data=dict(path=str(DATA),sha256=digest(DATA),row_sha256=hashes,source='HuggingFaceFW/fineweb',
            historically_opened=True,fresh_or_document_holdout=False,fitting=False),
        cache=dict(path=str(artifact),sha256=digest(artifact),bytes=artifact.stat().st_size),
        price=dict(body_forwards=counts[0],sequences=counts[1],distinct_prediction_positions=8192,native_background_retained=True,
            matrix_coefficients_per_program=7815168,support_indices_per_program=1179648,bias_coefficients=1152,radial_extra_coefficients=1152),
        seconds=time.perf_counter()-started,binding_sha256=digest(P/'FROZEN_RADIAL_FINEWEB_V1_BINDING.json'),
        scope='Frozen weight-only programs, cached FineWeb validation, no fitting or fresh/OOD/selectivity/composition claim.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','data')}),flush=True)


if __name__=='__main__':main()
