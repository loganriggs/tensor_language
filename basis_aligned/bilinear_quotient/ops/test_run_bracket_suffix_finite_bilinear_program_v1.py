import importlib.util
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).with_name('run_bracket_suffix_finite_bilinear_program_v1.py')
def load():
 spec=importlib.util.spec_from_file_location('bracket_suffix_finite',RUNNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_bound_plan():
 p=load().plan();assert p['price']['maximum_forwards']==12;assert p['candidate_order'][0]=='source_only';assert p['sites']==[13,14,15,16,17]
def test_finite_identity():
 rng=np.random.default_rng(7);L=rng.normal(size=(9,4));R=rng.normal(size=(9,4));D=rng.normal(size=(4,9));x=rng.normal(size=4);d=rng.normal(size=4);l0=L@x;r0=R@x;ld=L@d;rd=R@d
 np.testing.assert_allclose(D@(ld*r0+l0*rd+ld*rd),D@((L@(x+d))*(R@(x+d))-l0*r0),rtol=1e-12,atol=1e-12)
