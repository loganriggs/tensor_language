import importlib.util
from pathlib import Path

RUNNER=Path(__file__).with_name('run_bracket_cascade_source_component_factorial_v1.py')
spec=importlib.util.spec_from_file_location('source_factorial',RUNNER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_metrics_are_exact_for_identity():
 value=module.metrics([1,-2,3],[1,-2,3])
 assert value['cosine']==1 and value['relative_l2_error']==0
 assert value['sign_agreement']==1 and value['predicted_to_actual_norm_ratio']==1

def test_three_way_inclusion_exclusion_is_the_interaction():
 def f(a,b,c):return 2+3*a-4*b+5*c+7*a*b-11*a*c+13*b*c+17*a*b*c
 interaction=f(1,1,1)-f(1,1,0)-f(1,0,1)-f(0,1,1)+f(1,0,0)+f(0,1,0)+f(0,0,1)-f(0,0,0)
 assert interaction==17

def test_fixed_maximal_compression_order_is_literal():
 source=RUNNER.read_text()
 assert 'names=("full","key12","key1_payload","key2_payload","payload","key1","key2")' in source
