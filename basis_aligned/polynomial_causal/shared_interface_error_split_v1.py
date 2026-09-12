"""Exact paired scalar error accounting for the frozen common producer interface."""
from pathlib import Path
import json,hashlib
import torch
from attention_source_quartic_v1 import unpack
from shared_producer_interface_v1 import apply
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.no_grad()
def main():
    torch.set_num_threads(2);out=P/'SHARED_MLP15_INTERFACE_V1_ERROR_SPLIT.json';assert not out.exists()
    ap=P/'SHARED_MLP15_INTERFACE_V1_PROGRAM.pt';receipt=json.loads((P/'SHARED_MLP15_INTERFACE_V1_RESULT.json').read_text());assert receipt['pred_a'] and sha(ap)==receipt['artifact_sha']
    artifact=torch.load(ap,weights_only=True,map_location='cpu');source=torch.load(P/'MATCHED_PARTNER_MLP15_SOURCE_V1_PORTS.pt',weights_only=True,map_location='cpu')['sources']
    b=source['background'].double();m=source['producer'].double();raw=b+m;n=raw.square().mean(-1)+torch.finfo(torch.float32).eps
    native=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True,map_location='cpu')['ports'];den=native['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
    program=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True);forms=unpack(program,b)
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows'];reports=[];errors=[]
    def q(x,y):return torch.einsum('ni,kij,nj->nk',x,forms,y)
    original=q(raw,raw)
    for name,prog in artifact['programs'].items():
        prog={k:v.double() for k,v in prog.items()};approx=apply(prog,m,artifact['bias'].double());e=approx-m
        mixed=2*q(b,e);pure=2*q(m,e)+q(e,e);reconstructed=q(b+approx,b+approx);dq=reconstructed-original
        errors.append(float((mixed+pure-dq).norm()/dq.norm()))
        for j,branch in enumerate((3,8),1):
            parent=dq[:,0]*(original[:,j]+reconstructed[:,j])/2
            partner=dq[:,j]*(original[:,0]+reconstructed[:,0])/2
            alpha=original[:,0]*original[:,j]/(n.square()*den)
            actual=(reconstructed[:,0]*reconstructed[:,j]-original[:,0]*original[:,j])/(n.square()*den)
            terms=torch.stack([parent,partner],-1)/(n.square()*den)[:,None]
            errors.append(float((actual-terms.sum(-1)).norm()/actual.norm()))
            for family in sorted({r['family'] for r in rows}):
                ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);delta=(terms[1::2]-terms[::2])[ids]
                change=(alpha[1::2]-alpha[::2])[ids];total=delta.sum(-1)
                reports.append(dict(name=name,branch=branch,family=family,
                    scalar_change_error_relative=float(total.norm()/change.norm()),
                    parent_partner_error_norms_over_change=(delta.norm(dim=0)/change.norm()).tolist(),
                    parent_partner_signed_error_projection=((delta*total[:,None]).sum(0)/total.square().sum()).tolist()))
    result=dict(identity_errors=errors,reports=reports,artifact_sha=sha(ap),source_sha=sha(Path(__file__)),scope='Exact scalar-error allocation, not CE attribution. Pair changes use existing validation rows, no fit.')
    assert max(errors)<1e-10
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
