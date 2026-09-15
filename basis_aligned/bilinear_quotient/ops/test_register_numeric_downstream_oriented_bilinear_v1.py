import importlib.util
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('register_oriented',HERE/'register_numeric_downstream_oriented_bilinear_v1.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_builds_both_current_records(tmp_path):
    for filename,old,new,stem in module.TASKS:
        source=module.BQ/'circuits'/filename; target=tmp_path/filename; target.write_bytes(source.read_bytes())
        value=module.build(target,old,new,stem)
        assert value['claims'][-1]['claim_id']==new
        event=value['evidence_events'][-1]
        assert event['verdict']=='null'
        assert event['metrics'][2]['estimate']==0
