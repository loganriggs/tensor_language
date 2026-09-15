import importlib.util
from pathlib import Path
RUNNER=Path(__file__).with_name('run_bracket_key2_query_context_scalar_v1.py');spec=importlib.util.spec_from_file_location('key2_scalar',RUNNER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def test_bound_plan_is_activation_only():
 p=module.plan();assert p['train_endpoints']==432 and p['fit_endpoints']==144;assert p['candidates']==['anchored','two_scalar'];assert p['price']=={'forwards':5,'sequences':720,'fits':2,'backwards':0,'updates':0} and not p['model_loaded'] and not p['queue_touched']
def test_metric_and_pass_rule():
 exact=module.metrics([1.,2.,3.],[1.,2.,3.]);baseline={'relative_l2_error':.2};assert module.passed(exact,baseline)
def test_prereg_forbids_behavioral_fit_and_quantization():
 text=' '.join(module.PREREG.read_text().split());assert 'No behavioral effect' in text and 'answer logit' in text and 'quantization' in text
def test_single_prototype_rotary_inputs_are_four_dimensional():
 source=RUNNER.read_text();assert "proto_r.to(torch.float32)[None,None,None,:]" in source and "[0,0,0]" in source
