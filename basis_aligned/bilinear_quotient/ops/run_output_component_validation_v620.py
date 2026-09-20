#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_adapters pred_b_component_fidelity pred_c_finite_edits
"""Frozen v619 component: native normalized replay and removal controls.

Use two disjoint cached FineWeb panels (8 documents each, first64 tokens).
No data selection or fitting; this is document holdout, not domain OOD.
Build exact adapters from folded z ports to normalized MLP17 input, and from
vocabulary writer back to a residual write. Preserve actual final RMSNorm and
softcap. Remove candidate vs fixed equal-norm random residual direction.
Frozen gates: adapter algebra <=3e-5; component prediction relative error<=.15
in both panels; all native removal/control losses finite. No selectivity claim.
PRICE12 batched model forwards,0backwards,0fits,0nativeweightupdates.
"""
import os,json,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/output_component_validation_v620_result.json'
PREDICTIONS={'pred_a_adapters':'<=3e-5','pred_b_component_fidelity':'<=.15 in both panels',
             'pred_c_finite_edits':'finite native and random-control losses'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip7000.pt'],documents_per_panel=8,
        tokens=64,batch=4,forwards_max=12,model_backwards=0,model_updates=0,fit_parameters=0,
        execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    import circuit_fast_screen_producer as producer
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    path=OUT.with_name('output_shared_blocks_v619_component.pt')
    component=torch.load(path,map_location='cuda',weights_only=True)
    expected=json.loads(OUT.with_name('output_shared_blocks_v619_result.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['artifact_sha256']
    backend=producer.Bilin18TorchBackend.load('cuda');model=backend.model
    U=model.lm_head.weight.float();block=model.transformer.h[17]
    E=torch.cat([torch.eye(1152,device='cuda'),model.transformer.h[16].mlp.Down.weight.float()*block.lambdas[0],
                 block.attn.c_proj.weight.float()],1)
    Qe,Re=torch.linalg.qr(E.T);Qu,Ru=torch.linalg.qr(U)
    readers=torch.linalg.solve(Re,Qe.T@component['readers'])
    residual_writer=torch.linalg.solve(Ru,Qu.T@component['writer'])
    checks=dict(input=float((E.T@readers-component['readers']).norm()/component['readers'].norm()),
                output=float((U@residual_writer-component['writer']).norm()/component['writer'].norm()))
    native_reader=U.T@component['writer']
    random=torch.randn(1152,device='cuda',generator=torch.Generator(device='cuda').manual_seed(620))
    random=random/random.norm()*residual_writer.norm()
    rows=[];forwards=0
    for panel in plan['panels']:
        ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:8,:65].cuda()
        num=den=0.;losses={k:[] for k in ['baseline','remove','random']}
        for start in range(0,8,4):
            tokens=ids[start:start+4,:-1];target=ids[start:start+4,1:].contiguous()
            for mode in losses:
                def edit(_module,args,output):
                    nonlocal num,den
                    normalized=args[0].float()
                    alpha=(normalized@readers).square()@component['coefficients']
                    if mode=='baseline':
                        actual=output.float()@native_reader
                        num+=float((alpha-actual).double().square().sum())
                        den+=float(actual.double().square().sum())
                        return output
                    vector=residual_writer if mode=='remove' else random
                    return output-alpha[...,None]*vector
                hook=block.mlp.register_forward_hook(edit)
                try: loss=model(tokens,target)
                finally:hook.remove()
                losses[mode].append(float(loss));forwards+=1
        means={k:sum(v)/len(v) for k,v in losses.items()}
        rows.append(dict(panel=panel,component_relative_error=(num/max(den,1e-30))**.5,
            losses=means,removal_delta_ce=means['remove']-means['baseline'],
            random_delta_ce=means['random']-means['baseline']))
    assert forwards<=plan['forwards_max']
    adapter=OUT.with_name('output_component_validation_v620_adapter.pt')
    disk_guard.guard_torch_save(dict(readers=readers.cpu(),coefficients=component['coefficients'].cpu(),
        residual_writer=residual_writer.cpu(),vocabulary_writer=component['writer'].cpu(),
        input_port='actual normalized MLP17 input',parent_sha256=expected['artifact_sha256']),str(adapter))
    result=dict(plan=plan,adapter_checks=checks,rows=rows,forwards=forwards,
        vocabulary_writer_centered_energy_fraction=float((component['writer']-component['writer'].mean()).square().sum()/component['writer'].square().sum()),
        predictions={'pred_a_adapters':max(checks.values())<=3e-5,
            'pred_b_component_fidelity':all(r['component_relative_error']<=.15 for r in rows),
            'pred_c_finite_edits':all(torch.isfinite(torch.tensor(list(r['losses'].values()))).all().item() for r in rows)},
        scope='conditional MLP17 component only; heldout documents, no domain OOD or semantic selectivity/reuse/simplicity claim',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000,label='v620 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
