#!/usr/bin/env python3
# BQGATE:632bodyforwards;104prefixes<=247tokens;300seconds;no fitting.
"""pred_a native capability, sum/sector replay<=1e-5, selfdonor<=1e-4.
pred_b EACH regional family removal/donor>=.5, positive>=10/12 and20/24, unrelated<=.5.
pred_c BOTH natural halves meanabsCE<=.02,maxabs<=.1 with capability/meanhead controls.
pred_d target/control effect preservation<=10% each family in remove9/joint/donorjoint.
pred_e historical packaged score replay<=1e-6relative.
pred_f matched live scalar replay<=1e-10.
Frozen shared-read implementation versus original on every actual edited context.
632bodyforwards, plus paired scalar evaluation in each invoked hook; 300seconds.
No new corpus/OOD evidence or GPU speed prediction; CPU runtime result is separate.
"""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import run_sparse_pair_package_native_v1 as base
import extracted_circuits.sparse_even_key_producers_8_2_9_8_v1.execute as package
from shared_parent_runtime_v1 import SharedParent
from regional_cue_row_check_v1 import validate


@torch.no_grad()
def main():
    base.STEM='SHARED_PARENT_NATIVE_V1'
    rows=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())['regional']
    assert all(r['family']==i//24 for i,r in enumerate(rows))
    validate(rows[:48])
    validate(rows[48:], expected_token_differences=2)
    original=package.scalar; instances={};errors=[]
    def paired(current,tokens,p,index):
        if id(p) not in instances: instances[id(p)]=SharedParent(p)
        actual=instances[id(p)].scalar(current,tokens,index)
        reference=original(current,tokens,p,index)
        errors.append(float((actual-reference).norm()/reference.norm().clamp_min(1e-30)))
        return actual
    package.scalar=paired
    try:base.main()
    finally:package.scalar=original
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):return
    inherited=json.loads((P/'SHARED_PARENT_NATIVE_V1_RESULT.json').read_text())
    predicates={'pred_a':inherited['pred_'+'a'], 'pred_b':inherited['pred_'+'b'],
                'pred_c':inherited['pred_'+'c'], 'pred_d':inherited['pred_'+'d'],
                'pred_e':inherited['pred_'+'e']}
    result=dict(pred_f=bool(errors) and max(errors)<=1e-10,max_live_scalar_relative=max(errors),
                scalar_calls=len(errors),scope='Matched live original/shared scalar on all invoked native and edited contexts. Historical A-E remain in separate base result, including any machine-transfer failure.')
    result.update(predicates)
    out=P/'SHARED_PARENT_NATIVE_V1_REPLAY.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
