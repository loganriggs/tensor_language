import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('register_key2_relative',HERE/'register_bracket_key2_native_relative_transport_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def test_builds_v40_null(tmp_path):
 source=module.circuit_path(module.TAG);target=tmp_path/source.name;target.write_bytes(source.read_bytes());record=module.build(target);assert record['claims'][-1]['claim_id']==module.NEW;event=record['evidence_events'][-1];assert event['verdict']=='null' and event['metrics'][3]['estimate']==0
