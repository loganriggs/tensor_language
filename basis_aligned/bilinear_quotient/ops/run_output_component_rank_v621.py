#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_replay pred_b_compact_calibration pred_c_frozen_new_documents
"""Simplify the v619/v620 frozen component using its weight-ordered features.

Choose smallest prefix reaching<=.05 component error on8 skip1200 documents;
evaluate that fixed prefix on8 previously unopened skip11000 documents.
Predictions: full709-prefix <=.03 on both; selected<=64; frozen new-panel<=.05.
No refitting/reordering. This uses calibration to choose width, so only the
second panel is a fresh rank-validation test. It is not domain OOD.
PRICE4 batched native forwards,0backwards,0fits,0modelupdates.
"""
import json,os,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/output_component_rank_v621_result.json'
PREDICTIONS={'pred_a_full_replay':'<=.03 both','pred_b_compact_calibration':'selected<=64',
             'pred_c_frozen_new_documents':'<=.05'}


def main():
    plan=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip11000.pt'],documents=8,tokens=64,
        prefixes=[1,2,4,8,16,32,64,128,256,512],forwards_max=4,model_backwards=0,
        model_updates=0,fit_parameters=0,execution_policy='managed_queue_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan));return
    import torch
    import disk_guard
    import circuit_fast_screen_producer as producer
    tic=time.perf_counter();torch.set_grad_enabled(False);torch.set_num_threads(8)
    artifact=OUT.with_name('output_component_validation_v620_adapter.pt')
    p=torch.load(artifact,map_location='cuda',weights_only=True)
    counts=sorted(set([k for k in plan['prefixes'] if k<len(p['coefficients'])]+[len(p['coefficients'])]))
    backend=producer.Bilin18TorchBackend.load('cuda');model=backend.model
    native_reader=model.lm_head.weight.float().T@p['vocabulary_writer']
    rows=[];selected=None;forwards=0
    for panel_index,panel in enumerate(plan['panels']):
        source=BQ/'.rowcache'/panel
        ids=torch.load(source,map_location='cpu',weights_only=True)[:8,:65].cuda()
        numer=torch.zeros(len(counts),dtype=torch.float64,device='cuda');denom=0.
        for start in range(0,8,4):
            def capture(_m,args,output):
                nonlocal numer,denom
                truth=output.float()@native_reader
                cumulative=((args[0].float()@p['readers']).square()*p['coefficients']).cumsum(-1)
                predicted=cumulative[...,torch.tensor(counts,device='cuda')-1]
                numer+=(predicted-truth[...,None]).double().square().sum((0,1))
                denom+=float(truth.double().square().sum())
            hook=model.transformer.h[17].mlp.register_forward_hook(capture)
            try:model(ids[start:start+4,:-1],ids[start:start+4,1:].contiguous())
            finally:hook.remove()
            forwards+=1
        errors=(numer/max(denom,1e-30)).sqrt().cpu().tolist()
        if panel_index==0:
            passing=[k for k,e in zip(counts,errors) if e<=.05]
            selected=min(passing) if passing else None
        rows.append(dict(panel=panel,selected_rows_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            errors=dict(zip(counts,errors)),selected_relative_error=errors[counts.index(selected)] if selected else None))
    saved=None
    if selected:
        saved=OUT.with_name('output_component_rank_v621_program.pt')
        disk_guard.guard_torch_save(dict(readers=p['readers'][:,:selected].cpu(),coefficients=p['coefficients'][:selected].cpu(),
            residual_writer=p['residual_writer'].cpu(),vocabulary_writer=p['vocabulary_writer'].cpu(),
            input_port='actual normalized MLP17 input',selected_on=plan['panels'][0]),str(saved))
    result=dict(plan=plan,prefixes=counts,rows=rows,selected=selected,forwards=forwards,
        predictions={'pred_a_full_replay':all(r['errors'][counts[-1]]<=.03 for r in rows),
            'pred_b_compact_calibration':selected is not None and selected<=64,
            'pred_c_frozen_new_documents':selected is not None and rows[1]['selected_relative_error']<=.05},
        selected_program_values=(1152*selected+selected+1152+50304) if selected else None,
        scope='conditional component approximation; no semantic task, domain OOD, independent upstream extraction or reuse claim',
        seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000,label='v621 JSON');OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
