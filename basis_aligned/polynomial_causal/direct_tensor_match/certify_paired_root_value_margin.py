"""Conservative transfer of existing value margins through measured replay.
Triangle bound uses ||parent||/||teacher|| <= 1+parent_relative_error.
This certifies only the two opened panels and the measured output metric.
"""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
fit=json.loads((P/'QUARTIC_RANK_READOUT_V1.json').read_text());rewrite=json.loads((P/'PAIRED_ROOT_NATIVE_V2.json').read_text());original=next(r for r in fit['records'] if r['rank']==16);paired=next(r for r in rewrite['records'] if r['rank']==16)
upper=[e+d*(1+e) for e,d in zip(original['value_errors'],paired['float32_replay'])];limits=[1.1*.07744491681,1.1*.13609101747]
result=dict(rank=16,native_value_error_upper_bounds=upper,existing_limits=limits,preserves_value_margin=all(a<b for a,b in zip(upper,limits)),scope='Triangle certificate from measured FP32replay on two opened panels, common unembeddingEuclideanmetric. Not independent nativeevaluation, derivativecertificate or fullmodeladoption.')
(P/'PAIRED_ROOT_VALUE_CERTIFICATE_V2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
