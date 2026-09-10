#!/usr/bin/env python3
# BQGATE: R594 frozen selector/payload successor; FIT639, SELECT322 conditionally.
"""Preserve R585 scientific gates; record correctly rounded physical additions."""
import hashlib,importlib.util,json,os,signal,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();OPS=RUNNER.parent;POLY=OPS.parents[1]/'polynomial_causal'
sys.path.insert(0,str(OPS));sys.path.insert(0,str(POLY))
from circuit_fast_screen_managed_runner import atomic_create_json
BINDING=POLY/'INDUCTION_R594_BINDING.json'
OUT=POLY/'INDUCTION_R594_MANAGED_RESULT.json'
RAW=Path('/dev/shm/bilin18_induction_r594')
PRODUCER=OPS/'induction_centered_fixed_geometry_rung594.py'
REGISTERED_PREDICTIONS={
    'pred_a_selector_transfer':'Frozen R585 selector cells, unchanged',
    'pred_b_payload_transfer':'Frozen R585 payload cells, unchanged',
    'pred_c_control_selectivity':'Frozen R585 active-control and vocabulary cells, unchanged',
}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    requested_dry=bool(os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'))
    binding=json.loads(BINDING.read_text());assert all(sha(p)==v for p,v in binding.items())
    source=PRODUCER.read_bytes()
    spec=importlib.util.spec_from_loader('r594_managed_frozen',loader=None,origin=str(PRODUCER))
    p=importlib.util.module_from_spec(spec);p.__file__=str(PRODUCER)
    exec(compile(source,str(PRODUCER),'exec'),p.__dict__)
    dry=p.build_dryrun()
    if requested_dry:
        print(json.dumps(dict(dryrun=True,model_loaded=False,gpu_accessed=False,
            phase_counts=dry['phase_counts'],streaming_storage=dry['streaming_storage'])));return
    assert not OUT.exists() and not RAW.exists()
    RAW.mkdir();p.require_free_space(RAW)
    signal.alarm(1800)
    p.__r594_immutable_sha256__=sha(PRODUCER);p.__r594_adapter_sha256__=sha(RUNNER)
    started=time.time();result=p.run_science(public_root=RAW)
    result_name=p.INVALID_RESULT.name if result.get('status')=='invalid_diagnostic' else p.NORMAL_RESULT.name
    receipt_name=p.INVALID_RECEIPT.name if result.get('status')=='invalid_diagnostic' else p.NORMAL_RECEIPT.name
    raw_result=RAW/result_name;raw_receipt=RAW/receipt_name
    detail=json.loads(raw_result.read_text());receipt=json.loads(raw_receipt.read_text())
    # Preserve exact compact publication bytes on disk; the full raw tree stays in RAM.
    for source_path in (raw_result,raw_receipt):
        target=POLY/source_path.name
        with target.open('xb') as f:f.write(source_path.read_bytes())
    envelope=dict(result=detail,raw_result=str(raw_result),raw_receipt=str(raw_receipt),
        raw_result_sha256=sha(raw_result),raw_receipt_sha256=sha(raw_receipt),
        raw_storage_volatile=True,raw_root=str(RAW),runner_sha256=sha(RUNNER),
        binding_sha256=sha(BINDING),wall_seconds=time.time()-started,
        science_unchanged=True,independent_agent_review_claimed=False)
    atomic_create_json(OUT,envelope)
    print(json.dumps(dict(terminal=detail.get('terminal',detail.get('status')),
        failure_predicate=detail.get('failure_predicate'),failed_clauses=detail.get('failed_clauses'),
        model_forwards=detail.get('model_forwards',len(detail.get('executed_call_ids',[]))),
        evaluated_splits=detail.get('evaluated_splits'),wall_seconds=envelope['wall_seconds'],raw_root=str(RAW))))


if __name__=='__main__':main()
