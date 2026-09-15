import importlib.util,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('register_bracket_suffix',HERE/'register_bracket_suffix_finite_bilinear_program_v2.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_builds_v38_source_only_boundary(tmp_path):
 source=module.circuit_path(module.TAG);target=tmp_path/source.name;target.write_bytes(source.read_bytes());record=module.build(target)
 assert record['claims'][-1]['claim_id']==module.NEW
 event=record['evidence_events'][-1]
 assert event['verdict']=='held'
 assert event['metrics'][0]['estimate']<.25 and event['metrics'][1]['estimate']<.25
 assert 'quantization' in event['notes']
