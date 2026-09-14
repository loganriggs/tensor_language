"""No-fit per-row preservation exceptions, including small control effects."""
import json
from pathlib import Path
from datetime import datetime, timezone
import torch
P=Path(__file__).resolve().parent

def main():
    out=P/'SHARED_PARENT_EXCEPTION_V1_RESULT.json';assert not out.exists()
    new=torch.load(P/'SHARED_PARENT_NATIVE_V1_ARTIFACT.pt',weights_only=True)['regional'].double()
    old=torch.load(P/'SCALAR_JOINT_KEY_GRADES_CONFIRMATION_V1_ARTIFACT.pt',weights_only=True)['regional'].double()
    rows=[]
    for arm in [2,3,5]:
        a=new[:,arm]-new[:,0];b=old[:,arm]-old[:,0]
        for readout in [0,1]:
            sign=a[:,readout]*b[:,readout]<0
            rows.append(dict(arm=arm,readout=readout,sign_reversals=[dict(row=i,reference=float(b[i,readout]),candidate=float(a[i,readout])) for i in torch.where(sign)[0].tolist()],
                             material_reversals=int((sign&(b[:,readout].abs()>=1e-5)).sum()),
                             relative_over10percent=int(((a[:,readout]-b[:,readout]).abs()>.1*b[:,readout].abs()).sum()),
                             maximum_absolute_error=float((a[:,readout]-b[:,readout]).abs().max())))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,scope='Descriptive reused-panel exceptions versus exact uncompressed pair, not versus prior sparse implementation. Aggregate native passes do not erase individual or sign errors. No fitting or new gate.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
