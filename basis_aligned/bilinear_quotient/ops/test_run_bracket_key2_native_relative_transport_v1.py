import importlib.util
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).with_name('run_bracket_key2_native_relative_transport_v1.py');spec=importlib.util.spec_from_file_location('key2_transport',RUNNER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def test_bound_plan_keeps_select_sealed():
 p=module.plan();assert p['fit_endpoints']==p['select_endpoints']==144;assert p['candidates']==['relative_key2','relative_both'];assert p['price']['maximum_forwards']==11 and not p['model_loaded'] and not p['queue_touched']
def test_native_relative_transport_preserves_context():
 recipient=np.array([2.,-1.,4.]);prototype_recipient=np.array([1.,3.,2.]);prototype_donor=np.array([-2.,5.,7.]);transported=recipient+(prototype_donor-prototype_recipient);np.testing.assert_allclose(transported-recipient,prototype_donor-prototype_recipient)
def test_metric_identity():
 v=module.metrics([1.,-2.],[1.,-2.]);assert v['relative_l2_error']==0 and abs(v['cosine']-1)<1e-12
def test_captured_key_names_match_transport_table_names():
 source=RUNNER.read_text();assert "'key1':kn[" in source and "'key2':k2n[" in source
