"""Two exact weight-defined shared/private output splits; no behavioral fitting.
A sum, orthogonality and nuisance-mean-null identities <=1e-9.
B each s/es/ies mean-contrast retention >=.5 separately for each split.
Report all-pair responses: null mean is not null individual-token response.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    p=Path(__file__).parent;stem='NATIVE_RELATION_OUTPUT_SPLIT_V1'
    output=p/(stem+'.json');artifact=p/(stem+'.pt');assert not output.exists() and not artifact.exists()
    source=p/'NATIVE_TOKEN_RELATION_FOLD_V1.pt'
    old=json.loads((p/'NATIVE_TOKEN_RELATION_FOLD_V1.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==old['artifact_sha256']
    saved=torch.load(source,weights_only=True,map_location='cpu');a=saved['physical_readouts'][:3];n=saved['physical_readouts'][3:5]
    v=torch.linalg.solve(a@a.T,a).T;b=torch.linalg.qr(v,mode='reduced').Q
    _,singular,vh=torch.linalg.svd(n@b,full_matrices=True)
    assert singular[-1]>1e-8
    null=b@vh[-1];inside_private=null[:,None]@(null@v)[None,:]
    nuisance_basis=torch.linalg.qr(n.T,mode='reduced').Q
    ambient_shared=nuisance_basis@(nuisance_basis.T@v)
    definitions={'inside_span':inside_private,'ambient':v-ambient_shared}
    binding=json.loads((p/'NATIVE_RELATION_NEIGHBOR_V1_BINDING.json').read_text())['files']
    checkpoint=next(k for k in binding if k.endswith('/pytorch_model.bin'))
    state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu');u=state['lm_head.weight'].double()
    atlas=json.loads((p/'BRANCH_TOKEN_RELATIONS_V1.json').read_text());results={};splits={}
    for name,private in definitions.items():
        shared=v-private;errors=[float((private+shared-v).norm()/v.norm()),
            float((private.T@shared).norm()/v.square().sum()),float((n@private).norm()/v.norm())]
        # Apply the split to a physical write equal to each unit target readout.
        responses=a@private@a@a.T
        retention=responses.diag().tolist()
        cell=dict(pred_a=max(errors)<=1e-9,pred_b=min(retention)>=.5,errors=errors,
            target_mean_retention=retention,private_writer_rank=int(torch.linalg.matrix_rank(private)),
            shared_writer_rank=int(torch.linalg.matrix_rank(shared)),
            outside_original_span_relative=float((private-b@(b.T@private)).norm()/private.norm()),relations={})
        for relation,pairs in atlas['relations'].items():
            ids=torch.tensor(pairs);delta=u[ids[:,1]]-u[ids[:,0]]
            # Compare responses to the SAME three unit physical target writes.
            original=delta@a.T;pr=delta@private@a@a.T;sh=original-pr
            cell['relations'][relation]=dict(pair_count=len(pairs),original_mean=original.mean(0).tolist(),
                private_mean=pr.mean(0).tolist(),shared_mean=sh.mean(0).tolist(),
                private_rms_fraction=(pr.square().mean(0)/original.square().mean(0)).sqrt().tolist())
        results[name]=cell;splits[name]=dict(private_writers=private,shared_writers=shared)
    torch.save(dict(splits=splits,original_writers=v,nuisance_readouts=n,source_program='NATIVE_RELATION_SPLIT_V1.pt'),artifact)
    result=dict(pred_a=all(c['pred_a'] for c in results.values()),splits=results,restricted_nuisance_singular_values=singular.tolist(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        price=dict(original_writer_floats=3456,materialized_split_writer_floats=6912,variable_products=45,scalar_program_unchanged=True),
        scope='Supplied spelling contrasts annotate frozen weights. Mean-null writer split is not tokenwise/CE preservation; '
        'ambient output directions can leave original span. No task outcomes used in construction or circuit promotion.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='splits'},indent=2))
    for name,c in results.items():print(name,json.dumps({k:v for k,v in c.items() if k!='relations'}))


if __name__=='__main__':main()
