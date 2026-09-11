"""FP32 reachable-input replay of frozen Schur compiler and parent interfaces.
A source/patch and finite checks. B full output relative errors<=1e-5.
C parent1 zero/swap effect errors<=1e-5, both cached panels.
"""
import hashlib
import json
from pathlib import Path
import torch
from schur_graph_rewrite_v1 import evaluate


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    root=Path(__file__).parent;output=root/'SCHUR_GRAPH_REWRITE_NATURAL_REPLAY_V1.json'
    assert not output.exists()
    source=root/'SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt'
    path=root/'SCHUR_GRAPH_REWRITE_V1_PATCH.pt'
    receipt=json.loads((root/'SCHUR_GRAPH_REWRITE_V1.json').read_text())
    assert receipt['pred_a'] and receipt['pred_b'] and receipt['pred_c']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['patch_sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['source_sha256']
    graph=torch.load(source,weights_only=True,map_location='cpu')
    patches=torch.load(path,weights_only=True,map_location='cpu')['patches']
    results={};sources=[source,path]
    for label,suffix in [('fineweb','SUPPRESSION_V1'),('corpus_shift','CORPUS_SHIFT_V1')]:
        cache=root/f'SHARED_NODE_PARENT1_{suffix}_ENDPOINTS.pt';sources.append(cache)
        x=torch.load(cache,weights_only=True,map_location='cpu')['ports']['input'].float()
        assert x.shape==(384,1152)
        old=evaluate(graph,patches,x,new=False);new=evaluate(graph,patches,x,new=True)
        rows=[dict(intervention='none',output_error=float((old-new).norm()/old.norm()))]
        for intervention in ('zero1','swap1'):
            before=evaluate(graph,patches,x,intervention,new=False)
            after=evaluate(graph,patches,x,intervention,new=True)
            effect=old-before;changed=new-after
            rows.append(dict(intervention=intervention,output_error=float((before-after).norm()/before.norm()),
                effect_error=float((effect-changed).norm()/effect.norm())))
            assert bool(torch.isfinite(after).all() and torch.isfinite(before).all())
        results[label]=rows
    result=dict(pred_a=True,
        pred_b=all(row['output_error']<=1e-5 for rows in results.values() for row in rows),
        pred_c=all(row.get('effect_error',0)<=1e-5 for rows in results.values() for row in rows),
        panels=results,sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        scope='Frozen surrogate and its parent-port operations on768previously cached native inputs. '
              'No model-body or full-tail replay, factor fitting, new data, circuit discovery or global optimality.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
