"""Preserve each evaluated function before its resumable checkpoint advances."""
import hashlib,json,sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main(name):
    source=P/name;state=torch.load(source,map_location='cpu',weights_only=False)
    out=P/(source.stem.replace('_CHECKPOINT','')+f'_CHUNK_{state["next_chunk"]-1:02d}_BEST.pt')
    if out.exists():raise FileExistsError(out)
    frozen={k:state[k] for k in ['config','best','next_chunk','converged']}
    frozen.update(source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),schema='structured.fit.best.v1')
    torch.save(frozen,out)
    print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())))

if __name__=='__main__':main(sys.argv[1])
