#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_source_identity pred_b_pair_identity pred_c_eight_source_closure
"""Norm-closed greedy source discovery for frozen v621 quadratic component.

v625 attribution-ranked prefixes failed; select removals by actual squared
prediction error after recomputing the subset denominator. Freeze support
order on calibration before applying it to the other panel. Greedy only,
not a proof that no better support exists.

Observe all embedding/attention/MLP writes and carry each exactly once through
native lambdas to h17. Biases stay in their producing writes. Rank source removals by norm-closed error on skip1200; freeze for skip11000.
Compare retained prefixes using both original denominator (oracle control) and
their own RMSNorm (required for closure). Both panels were previously opened
for different tests; no claim of fresh domain OOD. No upstream recapture after
ablation: this is frozen-write boundary closure, not a causal source circuit.
Gates: raw/normalized source replay<=3e-5; pair sum replay<=3e-5;
top8 source own-norm prediction error against exact native quadratic<=.10 both.
PRICE4 native forwards,0backwards/updates,0coefficient fits; source-order ranking
on calibration is explicitly data-dependent discovery.
"""
import json,os,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/output_component_sources_v626_result.json'
PREDICTIONS={'pred_a_source_identity':'<=3e-5','pred_b_pair_identity':'<=3e-5',
             'pred_c_eight_source_closure':'<=.10 both'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],documents=8,tokens=64,
        prefixes=[1,2,4,8,16,24,36],forwards_max=4,model_backwards=0,model_updates=0,fit_parameters=0,
        execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import torch.nn.functional as F
    import disk_guard
    import circuit_fast_screen_producer as producer
    sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
    from quadratic_source_census import carried_sources,pair_table,read_quadratic,greedy_boundary_order
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    p=torch.load(OUT.with_name('output_component_rank_v621_program.pt'),map_location='cuda',weights_only=True)
    model=producer.Bilin18TorchBackend.load('cuda').model
    weights=carried_sources([(float(b.lambdas[0]),float(b.lambdas[1])) for b in model.transformer.h])
    names=list(weights);assert len(names)==36
    reader=model.lm_head.weight.float().T@p['vocabulary_writer']
    bias=model.transformer.h[17].mlp.Down_bias.float()@reader
    identity=[];pair_checks=[];rows=[];order=None;forwards=0
    for panel_index,panel in enumerate(plan['panels']):
        ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:8,:65].cuda()
        sources=[];truths=[];denominators=[];programs=[];scores=torch.zeros(36,device='cuda',dtype=torch.float64)
        self_energy=cross_energy=0.;tables=[];grams=[];epsilons=[]
        for start in range(0,8,4):
            cache={};hooks=[]
            def embedding_hook(_m,args):cache['embedding']=args[0].float()
            hooks.append(model.transformer.h[0].register_forward_pre_hook(embedding_hook))
            for layer,block in enumerate(model.transformer.h):
                def attn_hook(_m,args,out,name=f'attn{layer}'):cache[name]=out[0].float()
                def mlp_hook(_m,args,out,name=f'mlp{layer}'):
                    cache[name]=out.float()
                    if name=='mlp17':cache['n']=args[0].float()
                hooks.extend([block.attn.register_forward_hook(attn_hook),block.mlp.register_forward_hook(mlp_hook)])
            def final_hook(_m,args,out):cache['final']=out[0].float()
            hooks.append(model.transformer.h[17].register_forward_hook(final_hook))
            try:model(ids[start:start+4,:-1],ids[start:start+4,1:].contiguous());forwards+=1
            finally:
                for hook in hooks:hook.remove()
            src=torch.stack([cache[name]*weights[name] for name in names],-2).flatten(0,1)
            h=src.sum(-2);native_h=(cache['final']-cache['mlp17']).flatten(0,1)
            n=cache['n'].flatten(0,1)
            identity.extend([float((h-native_h).norm()/native_h.norm()),float((F.rms_norm(h,(1152,))-n).norm()/n.norm())])
            denom=h.square().mean(-1)+torch.finfo(h.dtype).eps
            table=pair_table(src,p['readers'],p['coefficients'],denom)
            program=(n@p['readers']).square()@p['coefficients']
            pair_checks.append(float((table.sum((-1,-2))-program).norm()/program.norm()))
            scores+=table.sum(-1).abs().double().sum(0)
            tables.append(table)
            grams.append((src@src.transpose(-1,-2))/(1152*denom[:,None,None]))
            epsilons.append(torch.finfo(h.dtype).eps/denom)
            diag=table.diagonal(dim1=-2,dim2=-1).sum(-1)
            self_energy+=float(diag.double().square().sum())
            cross_energy+=float((program-diag).double().square().sum())
            sources.append(src);truths.append((cache['mlp17']@reader-bias).flatten())
            programs.append(program);denominators.append(denom)
        src=torch.cat(sources);truth=torch.cat(truths);program=torch.cat(programs);denom=torch.cat(denominators)
        if panel_index==0:
            removal=greedy_boundary_order(torch.cat(tables),torch.cat(grams),torch.cat(epsilons),truth)
            order=removal.flip(0)
        frontier=[]
        for count in plan['prefixes']:
            retained=src[:,order[:count]].sum(-2)
            oracle=read_quadratic(retained,p['readers'],p['coefficients'],denominator=denom)
            closed=read_quadratic(retained,p['readers'],p['coefficients'])
            frontier.append(dict(sources=count,
                oracle_norm_error_vs_native=float((oracle-truth).norm()/truth.norm()),
                own_norm_error_vs_native=float((closed-truth).norm()/truth.norm()),
                own_norm_error_vs_program=float((closed-program).norm()/program.norm())))
        rows.append(dict(panel=panel,rows_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            frontier=frontier,half_cross_absolute_attribution={name:float(scores[i]/len(truth)) for i,name in enumerate(names)},
            self_term_l2_over_program=(self_energy/float(program.double().square().sum()))**.5,
            cross_term_l2_over_program=(cross_energy/float(program.double().square().sum()))**.5))
    result=dict(plan=plan,rows=rows,calibration_source_order=[names[i] for i in order.tolist()],carriage_weights=weights,
        maximum_source_replay=max(identity),maximum_pair_replay=max(pair_checks),forwards=forwards,
        predictions={'pred_a_source_identity':max(identity)<=3e-5,'pred_b_pair_identity':max(pair_checks)<=3e-5,
            'pred_c_eight_source_closure':all(next(x for x in r['frontier'] if x['sources']==8)['own_norm_error_vs_native']<=.1 for r in rows)},
        scope='frozen native-write boundary census; attribution is not source causality, and retained writes still require upstream generators',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    assert forwards<=plan['forwards_max']
    disk_guard.guard_write(100000,label='v626 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
