import importlib.util
from pathlib import Path
import numpy as np
RUNNER=Path(__file__).with_name('run_induction_early_mlp_finite_bilinear_factor_v2.py')
def load():
 spec=importlib.util.spec_from_file_location('finite_bilinear_v2',RUNNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_bound_plan():
 p=load().plan();assert p['price']=={'forwards':24,'sequences':768,'backwards':0,'fits':0};assert p['candidate_order'][0]=='quadratic'
def test_finite_bilinear_identity():
 rng=np.random.default_rng(2);L=rng.normal(size=(7,5));R=rng.normal(size=(7,5));D=rng.normal(size=(5,7));x=rng.normal(size=5);d=rng.normal(size=5);l0=L@x;r0=R@x;ld=L@d;rd=R@d
 np.testing.assert_allclose(D@(ld*r0)+D@(l0*rd)+D@(ld*rd),D@((L@(x+d))*(R@(x+d))-l0*r0),rtol=1e-12,atol=1e-12)
