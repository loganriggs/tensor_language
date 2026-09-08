import numpy as np
import run_temporal_iswas_v23_occupied_reader_weight_writer_gauge_audit_v1 as audit

def test_right_rotation_invariance():
    rng=np.random.default_rng(2); a=rng.normal(size=(12,4)); b=rng.normal(size=(12,4)); q,_=np.linalg.qr(rng.normal(size=(4,4)))
    assert abs(audit.procrustes(a,b)-audit.procrustes(a,b@q)) < 1e-12
    assert abs(audit.gram_cosine(a,b)-audit.gram_cosine(a,b@q)) < 1e-12

def test_identical_and_orthogonal_context_maps():
    a=np.eye(4)[:,:2]; b=np.eye(4)[:,2:]
    assert abs(audit.procrustes(a,a)-1) < 1e-12
    assert audit.gram_cosine(a,b) == 0
