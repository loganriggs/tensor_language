"""Exact differential of an additive source gain; CPU, no fitting or native weights."""
from pathlib import Path
import json,hashlib,time,sys
import torch

P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.no_grad()
def main(stem):
    torch.set_num_threads(2);tic=time.perf_counter()
    out=P/(stem+'_DIFFERENTIAL.json');assert not out.exists()
    path=P/(stem+'_PORTS.pt');receipt=json.loads((P/(stem+'_RESULT.json')).read_text())
    assert receipt['pred_a'] and sha(path)==receipt['artifact_sha']
    c=torch.load(path,weights_only=True,map_location='cpu');q=c['compiled']['quadratic_sources'];norm=c['compiled']['norm']
    cache=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True,map_location='cpu')
    den=cache['ports']['pre'].double().square().mean(-1)+torch.finfo(torch.float32).eps
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows']
    assert sha(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json')==c['rows_sha']
    v=q.sum(-1);vp=q[:,:,1]+2*q[:,:,2];n=norm.sum(-1);np=norm[:,1]+2*norm[:,2]
    alpha=v[:,0,None]*v[:,1:]/(n.square()*den)[:,None]
    parent=vp[:,0,None]*v[:,1:]/(n.square()*den)[:,None]
    partner=v[:,0,None]*vp[:,1:]/(n.square()*den)[:,None]
    normalization=-2*(np/n)[:,None]*alpha
    derivative=parent+partner+normalization
    def at(g):
        x=q@q.new_tensor([1.,g,g*g]);nn=norm@norm.new_tensor([1.,g,g*g])
        return x[:,0,None]*x[:,1:]/(nn.square()*den)[:,None]
    step=1e-4;fd=(at(1+step)-at(1-step))/(2*step)
    error=float((fd-derivative).norm()/derivative.norm());assert error<=1e-7
    reports=[]
    for j,branch in enumerate([3,8]):
        terms=torch.stack([parent[:,j],partner[:,j],normalization[:,j]],-1)
        for family in sorted({r['family'] for r in rows}):
            ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family])
            delta=(terms[1::2]-terms[::2])[ids]
            full=(alpha[1::2,j]-alpha[::2,j])[ids];scale=full.norm().clamp_min(1e-30)
            sec=c['scalar_sectors'][:,j];sd=(sec[1::2]-sec[::2])[ids]
            reports.append(dict(branch=branch,family=family,
                derivative_parts_norm_over_full_change=(delta.norm(dim=0)/scale).tolist(),
                derivative_parts_projection_on_full_change=((delta*full[:,None]).sum(0)/scale.square()).tolist(),
                source_degree_norm_over_full_change=(sd.norm(dim=0)/scale).tolist(),
                producer_degree_cancellation=float(sd[:,1:].norm(dim=0).sum()/sd[:,1:].sum(-1).norm().clamp_min(1e-30))))
    result=dict(finite_difference_relative_error=error,reports=reports,ports_sha=sha(path),source_sha=sha(Path(__file__)),
                execution_seconds=time.perf_counter()-tic,scope='Derivative at native gain1 of conditional branch coefficient; not finite CE attribution or full-model source intervention.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main(sys.argv[1])
