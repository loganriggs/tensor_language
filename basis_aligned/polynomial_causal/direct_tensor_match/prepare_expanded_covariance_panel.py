"""Freeze additional covariance-calibration inputs without using outcomes.
Retain old24 training prefixes; exclude all32 old calibration prefixes, including
the seven opened evaluation rows and their exact duplicates.
"""
from pathlib import Path
import json,hashlib
import torch
P=Path(__file__).resolve().parent
cal=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)
cache=P.parents[1]/'bilinear_quotient/.rowcache/fineweb_n480_skip80.pt'
pool=torch.load(cache,weights_only=True)
digest=lambda row:hashlib.sha256(row[:64].numpy().tobytes()).hexdigest()
old=cal['tokens'][:24,:65].clone();seen={digest(row) for row in cal['tokens']}
assert len({digest(row) for row in old})==24
chosen=[];indices=[];excluded=[]
for i,row in enumerate(pool):
 h=digest(row)
 if h in seen:excluded.append(i);continue
 seen.add(h);chosen.append(row[:65].clone());indices.append(i)
 if len(chosen)==232:break
assert len(chosen)==232
tokens=torch.stack(chosen)
assert not ({digest(row) for row in tokens}&{digest(row) for row in cal['tokens']})
torch.save(tokens,P/'EXPANDED_COVARIANCE_TOKENS_V1.pt')
out=dict(existing_training_rows=list(range(24)),additional_cache=str(cache),additional_cache_rows=indices,excluded_duplicate_cache_rows=excluded,additional_rows=232,total_training_rows=256,input_context=64,token_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),scope='Input-only deterministic selection for expanded covariance calibration. Cache has been used elsewhere; these are training inputs, not fresh evaluation. All32 old calibration/evaluation exact64tokenprefixes excluded from additional rows. No outcomes used.')
(P/'EXPANDED_COVARIANCE_PANEL_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k not in ['additional_cache_rows','excluded_duplicate_cache_rows']})
