"""Matched causal summary variation/removal magnitudes; no replacement claim."""
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
import exact_source_edit_reference as E
import summary_record_transport_reference as S
import field_intervention_metrics as M

BASE=Path(__file__).resolve().parent
INITIAL=BASE/'QUERY_INITIALIZER_FACTORIZATION_V1_ROWS.pt'
MEDIATION=BASE/'SUMMARY_LAYOUT_MEDIATION_V1_ROWS.pt'
OUT=BASE/'SUMMARY_VARIATION_VS_REMOVAL_V1.json'


def main():
    signal.alarm(180);torch.set_num_threads(2);started=time.perf_counter();assert not OUT.exists()
    assert hashlib.sha256(INITIAL.read_bytes()).hexdigest()=='899e48cc00c27b966c43dea9bf2406fcdcc6bdb5d7b14436ee661027e3870e7e'
    assert hashlib.sha256(MEDIATION.read_bytes()).hexdigest()=='3dfdb8db3c3206b3ecfb674468fc0bfe036127a97060308de826940d7224a22e'
    initial=torch.load(INITIAL,map_location='cpu',weights_only=True)['populations']
    mediation=torch.load(MEDIATION,map_location='cpu',weights_only=True)
    package=BASE/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    assert hashlib.sha256(package.read_bytes()).hexdigest()=='e69bff6562c48bd1b59366f118157720734d353626b0a1576626bdd437a1421a'
    program=E.load_package(torch.load(package,map_location='cpu',weights_only=True))
    with torch.inference_mode():a,offset,c=S.operator(program)
    offset=offset.reshape(4,128);groups={};audits=[]
    center=lambda x:x-x.mean(-1,keepdim=True)
    rms=lambda x:float(x.square().mean().sqrt())
    for pop,block in initial.items():
        summaries=block['document_summaries'];groups[pop]={}
        for name,comparison in mediation[pop].items():
            assert block['metadata']==comparison['metadata']
            logits=comparison['query_logits'];audits.append(M.correspondence(logits['00'],block['query_logits']['native']))
            summary_removal=center(block['query_logits']['native']-block['query_logits']['2'])
            summary_change=center(logits['01']-logits['00']);order_change=center(logits['11']-logits['00'])
            groups[pop][name]={}
            for hop in range(4):
                sel=torch.tensor([m['hop']==hop for m in block['metadata']]);removal=rms(summary_removal[sel]);order=rms(order_change[sel]);change=rms(summary_change[sel])
                raw=rms(summaries[:,hop]-offset[hop]);mean=rms(offset[hop]);total=rms(summaries[:,hop])
                groups[pop][name][str(hop)]={'requests':int(sel.sum()),'summary_order_effect_rms':change,
                    'summary_removal_effect_rms':removal,'native_order_effect_rms':order,
                    'summary_order_over_removal':change/max(removal,1e-6),'summary_order_over_native_order':change/max(order,1e-6),
                    'summary_removal_denominator_floored':removal<1e-6,'native_order_denominator_floored':order<1e-6,
                    'raw_summary_deviation_from_exact_orbit_mean_rms':raw,'raw_orbit_mean_rms':mean,
                    'raw_deviation_over_total_summary_rms':raw/max(total,1e-6)}
    result={'scope':'opened matched causal variation/removal audit and exact first-layer record-orbit mean; not a constant-summary replacement test',
        'mechanical_passed':all(x['passed'] for x in audits),'matched_native_max_abs':max(x['max_abs'] for x in audits),
        'groups':groups,'orbit_mean_summary':offset.tolist(),'orbit_mean_note':'Uniform record permutation gives each position every key/value identity equally; both incidence matrices average to11^T/24 regardless of represented bijection',
        'native_coefficients_removed':0,'wall_seconds':time.perf_counter()-started,
        'input_sha256':{'initial':hashlib.sha256(INITIAL.read_bytes()).hexdigest(),'mediation':hashlib.sha256(MEDIATION.read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(result,indent=2)+'\n');assert result['mechanical_passed']
    print(json.dumps({'mechanical_passed':result['mechanical_passed'],'matched_native_max_abs':result['matched_native_max_abs'],
        'wall_seconds':result['wall_seconds'],'hop3':{p:{n:g['3'] for n,g in v.items()} for p,v in groups.items()}},indent=2))


if __name__=='__main__':main()
