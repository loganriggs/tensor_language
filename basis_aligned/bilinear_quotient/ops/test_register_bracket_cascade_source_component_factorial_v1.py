import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('register_source_factorial',HERE/'register_bracket_cascade_source_component_factorial_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def test_builds_v39_localization(tmp_path):
 source=module.circuit_path(module.TAG);target=tmp_path/source.name;target.write_bytes(source.read_bytes());record=module.build(target);assert record['claims'][-1]['claim_id']==module.NEW;event=record['evidence_events'][-1];assert event['verdict']=='held';assert event['metrics'][0]['estimate']=='key12'
