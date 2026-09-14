"""Frozen terminal-copy mask audit for attention5 task-defined split."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import torch


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[2]; p=root/'basis_aligned/polynomial_causal'; bq=root/'basis_aligned/bilinear_quotient'
    corpora={}
    for name in ('final_natural.pt','ood_code.pt'):
        path=bq/'.rowcache_terminal_copy_induction_v2'/name; obj=torch.load(path,map_location='cpu',weights_only=False); cells=obj['copy_cells']
        pos,neg=cells['positive'][:24],cells['matched_negative'][:24]
        corpora[name]={"positive":int(pos.sum()),"matched_negative":int(neg.sum()),"overlap":int((pos&neg).sum()),"sha256":sha(path),
                       "positive_sha256":hashlib.sha256(pos.numpy().tobytes()).hexdigest(),"negative_sha256":hashlib.sha256(neg.numpy().tobytes()).hexdigest()}
    result={"utc":datetime.now(timezone.utc).isoformat(),"pred_a":all(x['positive']>20 and x['matched_negative']>20 and x['overlap']==0 for x in corpora.values()),
            "corpora":corpora,"parent_artifact_sha256":sha(p/'ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_ARTIFACT.pt'),
            "scope":"Frozen mask/hash audit only; no new model score or head-group verdict."}
    out=p/'ATTENTION5_COPY_CONTENT_DEVIATION_SPLIT_V1_CPU_CONTROL.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
