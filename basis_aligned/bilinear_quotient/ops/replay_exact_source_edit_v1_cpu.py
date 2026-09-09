#!/usr/bin/env python3
"""Fresh-process export replay; original checkpoint-directory reads are denied."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path('/workspace/tensor_language');POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(POLY)]
DENIED=[]


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT/'runs_hop'):
            DENIED.append(str(path));raise PermissionError('Original checkpoint-directory access denied in export replay')


def main():
    import torch
    import exact_source_edit_reference as R
    torch.set_num_threads(2);started=time.perf_counter()
    package_path=POLY/'EXACT_SOURCE_EDIT_V1_PROGRAM.pt'
    fixtures_path=POLY/'EXACT_SOURCE_EDIT_V1_REPLAY_INPUTS.pt'
    output=POLY/'EXACT_SOURCE_EDIT_V1_CPU_REPLAY.json';assert not output.exists()
    sys.addaudithook(guard)
    try:
        with open(ROOT/'runs_hop/attn4-rms-seed0/model.pt','rb'):pass
        raise AssertionError('checkpoint access guard is not live')
    except PermissionError:pass
    package=torch.load(package_path,map_location='cpu',weights_only=True)
    fixtures=torch.load(fixtures_path,map_location='cpu',weights_only=True)
    program=R.load_package(package);maximum=0.;relative=0.;cases=0
    with torch.inference_mode():
        for fixture in fixtures:
            tokens=fixture['tokens'];context=program.prepare(tokens)
            edits=R.make_edits(program.background,tokens,fixture['masks'],fixture['heads'])
            for arm,delta in edits.items():
                actual=program.edit(context,delta);expected=fixture['expected'][arm]
                maximum=max(maximum,float((actual-expected).abs().max()))
                relative=max(relative,float((actual-expected).square().mean().sqrt())/max(float(expected.square().mean().sqrt()),1e-6))
                cases+=len(tokens)
    passed=maximum<=1e-9 and relative<=1e-10 and len(DENIED)==1 and program.independent_constant_count()==387968
    receipt={'passed':passed,'max_abs_logit':maximum,'max_relative_rms':relative,'example_arms':cases,
             'original_checkpoint_guard_test_passed':len(DENIED)==1,'unexpected_checkpoint_accesses':DENIED[1:],
             'native_O_absent':not hasattr(program.background.layers[-1],'o'),'independent_constants':program.independent_constant_count(),
             'program_sha256':hashlib.sha256(package_path.read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(fixtures_path.read_bytes()).hexdigest(),
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
