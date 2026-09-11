"""Quantify overlap with already documented joint32 pronoun products.

A Gram/dense projection<=1e-10; B both oldtwoinputspan capture>=.8;
C both oldtwowriter spans capture>=.9 current output direction.
Post-result alias audit, not a new gender circuit or behavioral test.
"""
import hashlib,json
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross


def main():
    torch.set_grad_enabled(False);torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    p=Path(__file__).resolve().parent;out=p/'SHARED_FUNCTION_PRIOR_ALIAS_V1_AUDIT.json';assert not out.exists()
    old=json.loads((p/'UNSUPERVISED_JOINT32_V1_RESULT.json').read_text());path=p/'UNSUPERVISED_JOINT32_V1_PROGRAM.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==old['artifact_sha256']
    programs=torch.load(path,weights_only=True,map_location='cpu')
    current=json.loads((p/'SHARED_NATIVE_FUNCTION_PRODUCTS_V1_AUDIT.json').read_text())['cache']
    assert hashlib.sha256(Path(current['path']).read_bytes()).hexdigest()==current['sha256']
    saved=torch.load(current['path'],weights_only=True,map_location='cpu');form=saved['native_form'];q=saved['native_output_readout'];wh=saved['unembedding_whitener']
    energy=form.square().sum();rows=[]
    for name in ('native','random'):
        matched=[0,9] if name=='native' else [old['stability']['native_partner'][i] for i in [0,9]]
        program=programs[name]
        for label,ids in [('matched_two',matched),('all32',list(range(32)))]:
            a=program['a'][ids].double();b=program['b'][ids].double();w=wh@program['w'][:,ids].double()
            gram=product_cross(a,b,a,b);rhs=((a@form)*b).sum(1)
            beta=torch.linalg.solve(gram,rhs);reconstruction=(a.T*beta)@b;reconstruction=(reconstruction+reconstruction.T)/2
            capture=float(rhs@beta/energy);densecapture=1-float((form-reconstruction).square().sum()/energy)
            ob=torch.linalg.qr(w,mode='reduced').Q
            rows.append(dict(start=name,scope=label,indices=ids,input_function_capture=capture,
                dense_projection_replay=abs(capture-densecapture),output_direction_capture=float((ob.T@q).square().sum()),
                individual_input_cosines=(rhs/(gram.diag()*energy).sqrt()).tolist()))
    pair=[r for r in rows if r['scope']=='matched_two']
    result=dict(predictions=dict(pred_a_instrument=max(r['dense_projection_replay'] for r in rows)<=1e-10,
        pred_b_known_input_recovery=all(r['input_function_capture']>=.8 for r in pair),
        pred_c_known_output_recovery=all(r['output_direction_capture']>=.9 for r in pair)),
        rows=rows,source=current,old_artifact=dict(path=str(path),sha256=old['artifact_sha256']),
        scope='Projection against fixed prior components; partial recovery is not equivalence or a new circuit count. Earlier failed reflection/selectivity results remain.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
