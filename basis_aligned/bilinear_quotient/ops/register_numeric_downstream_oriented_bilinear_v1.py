#!/usr/bin/env python3
"""Register the audited oriented downstream bilinear null in both numeric tasks."""
import copy
import hashlib
import json
from pathlib import Path
import sys

BQ=Path(__file__).resolve().parents[1]; REPO=BQ.parents[1]
sys.path.insert(0,str(BQ))
import circuit_registry_v2 as registry
CK='680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3'
FILES={
 'numeric_oriented_prereg_v1':('basis_aligned/polynomial_causal/NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_PREREGISTRATION.md','22f9c731ae3994dbc09d3a54d4d9953782a5cc0cd73142b54236458b3779f0e2','preregistration'),
 'numeric_oriented_runner_v1':('basis_aligned/bilinear_quotient/ops/run_numeric_downstream_oriented_bilinear_v1.py','bd77571b6d80489d796ea98bc499f1b6bb265d353362524aaf20969f00c84674','implementation'),
 'numeric_oriented_result_v1':('basis_aligned/polynomial_causal/NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_RESULT.json','1aaa45e8d2a0947be8da205d690cf3c0d015613e794e6e2ffb68d7c299482706','screen_result'),
 'numeric_oriented_audit_v1':('basis_aligned/polynomial_causal/NUMERIC_DOWNSTREAM_ORIENTED_BILINEAR_V1_AUDIT.json','82f83eaf5047907477013dd79efb9bb63cc91f4027bf32327d223666937e3804','postexecution_audit'),
}
TASKS=[
 ('task_numbered_list_index_successor.json','numbered_list_index_successor.v13','numbered_list_index_successor.v14','numbered_list_index_successor'),
 ('task_numeric_sequence_continuation.json','numeric_sequence_continuation.v11','numeric_sequence_continuation.v12','numeric_sequence_continuation'),
]


def metric(name,estimate,bar): return {'name':name,'estimate':estimate,'ci95':None,'bar':bar}


def build(path,old,new,stem):
    record=json.loads(path.read_text())
    if record['claims'][-1]['claim_id']!=old: raise ValueError('moved '+old)
    for aid,(relative,expected,kind) in FILES.items():
        if hashlib.sha256((REPO/relative).read_bytes()).hexdigest()!=expected: raise ValueError('drift '+aid)
        record['artifacts'][aid]={'path':relative,'sha256':expected,'kind':kind,'status':'frozen'}
    event_id=stem+'.downstream_oriented_bilinear.v1.complete.null'
    claim=copy.deepcopy(record['claims'][-1])
    claim.update(claim_id=new,revision=int(new.rsplit('v',1)[1]),supersedes=old,status='weights_translated')
    claim['evidence_event_ids']+=[event_id]
    claim['next_missing']=('The exact shared H3+H7 cached payload remains functionally causal, but neither R590 coarse MLP responses nor the finer native left-delta/right-background and left-background/right-delta products isolate a selective successor action carrier at MLP8/10/12/14. The oriented split was exact and all 16 candidates failed FIT before SELECT. Do not repeat cached-vector ranks or these MLP component partitions. Continue only with an independently defined native suffix-state interaction quotient, or move to another circuit.')
    record['claims'].append(claim)
    event={'event_id':event_id,'test_type':'composition','stage':'complete','verdict':'null','failure_kind':'scientific_null',
      'site_id':'final_label_l0_value_through_l8h3_h7','result_artifact_id':'numeric_oriented_result_v1',
      'prereg_artifact_id':'numeric_oriented_prereg_v1','metrics':[
        metric('maximum_exactness_error',7.141377056137704e-11,'<=1e-10'),
        metric('oriented_candidate_count',16,'16'),metric('passing_FIT_candidates',0,'at least 1'),
        metric('maximum_target_cells_passed',10,'12'),metric('MLP8_oriented_copy_cells_passed',0,'12'),
        metric('observed_forwards',487,'<=632'),metric('SELECT_opened',False,'only after FIT selection')],
      'supersedes_event_id':None,'notes':{'terminal':'fit_null','scientific_status':'valid CPU-audited null',
        'decomposition':'D[(Ld)*(Rx0)], D[(Lx0)*(Rd)], D[(Ld)*(Rd)]',
        'scope':'R582 FIT only; SELECT, FINAL_TEST and OOD remain sealed'},
      'claim_id':new,'family_ids':[],'evaluation_role':'frozen FIT finer-consumer screen',
      'input_artifact_ids':['numeric_oriented_runner_v1','numeric_oriented_audit_v1'],
      'split_plan_id':'r590_source_matched_downstream_use_fit','seed':None,'checkpoint_sha256':CK,
      'replicates_event_id':None,'sections':[]}
    event['design_key']=registry.design_key(record,event); event['execution_key']=registry.execution_key(record,event)
    record['evidence_events'].append(event); registry.validate_v2(record); return record


def main():
    values=[]
    for filename,old,new,stem in TASKS:
        path=BQ/'circuits'/filename; values.append((path,old,build(path,old,new,stem)))
    with registry._lock('registry'):
        for path,old,_ in values:
            if json.loads(path.read_text())['claims'][-1]['claim_id']!=old: raise ValueError('moved during publication')
        for path,_,value in values: registry._atomic_json(path,value)
    registry.rebuild_registry_v2()
    print(json.dumps({'written':[str(path.relative_to(REPO)) for path,_,_ in values],'gpu_used':False}))


if __name__=='__main__': main()
