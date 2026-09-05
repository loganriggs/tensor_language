import run_task14_mlp6_7_fixed_direction_reader_artifact_v2 as run

def test_width_only_price_correction():
 assert run.READER_WIDTH==1152 and run.derive_price()["stored_scalars"]==2304
 assert {k:v for k,v in run.derive_price().items() if k!="stored_scalars"}=={
  k:v for k,v in run.v1.derive_price().items() if k!="stored_scalars"}

def test_rescore_exports_1152_but_preserves_distinctness_miss():
 readers={d:{"coordinates":[0.0]*1152,"l2_norm":1.0} for d in ("a","b")}
 p=run.rescore(readers,{"x":0.0},{"all_gradient_l2_norm":1.0,"inter_direction_cosine":-.3})
 assert p[run.PRED_KEYS[0]] and p[run.PRED_KEYS[1]] and not p[run.PRED_KEYS[2]]
