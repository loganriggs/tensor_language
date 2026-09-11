"""Full native quadratic for six weight/tokenizer-defined output contrasts.
A factor/dense, radial and leading-product identities <=1e-9.
B one real product captures >=90% traceless energy in each s/es/ies readout.
C sixteen real products capture >=90% in each of these three fixed readouts.
No text fitting; exact scalar rank curves are not full-DAG lower bounds.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent
    output=root/'NATIVE_TOKEN_RELATION_FOLD_V1.json';artifact=root/'NATIVE_TOKEN_RELATION_FOLD_V1.pt'
    assert not output.exists() and not artifact.exists()
    source=root/'BRANCH_TOKEN_RELATIONS_V1.json';atlas=json.loads(source.read_text());assert atlas['pred_a']
    binding=json.loads((root/'FROZEN_BRANCH_TENSE_V6_BINDING.json').read_text())['files']
    checkpoint=next(p for p in binding if p.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    unembedding=weights['lm_head.weight'].double()
    left,right,down=[weights[f'transformer.h.17.mlp.{name}.weight'].double() for name in ('Left','Right','Down')]
    bias=weights['transformer.h.17.mlp.Down_bias'].double()
    labels=list(atlas['relations']);readouts=[]
    for relation in labels:
        ids=torch.tensor(atlas['relations'][relation],dtype=torch.long)
        readout=(unembedding[ids[:,1]]-unembedding[ids[:,0]]).mean(0)
        readouts.append(readout/readout.norm())
    readouts=torch.stack(readouts);folded=readouts@down;folded_bias=readouts@bias
    d=left.shape[1];eye=torch.eye(d)
    torch.manual_seed(6201)
    x=torch.randn(19,d);x=x/(x.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
    reference=(((x@left.T)*(x@right.T))@down.T+bias)@readouts.T
    cells=[];compact=[];errors=[];matrices=[]
    for j,label in enumerate(labels):
        raw=(left.T*folded[j])@right;q=(raw+raw.T)/2
        alpha=q.trace()/d;centered=q-alpha*eye
        exact=torch.einsum('ni,ij,nj->n',x,q,x)+folded_bias[j]
        rebuilt=torch.einsum('ni,ij,nj->n',x,centered,x)+alpha*x.square().sum(1)+folded_bias[j]
        errors += [float((exact-reference[:,j]).norm()/reference[:,j].norm()),
                   float((rebuilt-exact).norm()/exact.norm())]
        values,vectors=torch.linalg.eigh(centered)
        order=values.abs().argsort(descending=True)
        energy=values.square().sum()
        pos=values.clamp_min(0).flip(0);neg=(-values).clamp_min(0)
        product_curve=(pos.square()+neg.square()).cumsum(0)/energy
        square_curve=values[order].square().cumsum(0)/energy
        ap=values[-1].clamp_min(0).sqrt()*vectors[:,-1]
        an=(-values[0]).clamp_min(0).sqrt()*vectors[:,0]
        a,b=ap+an,ap-an
        leading=values[-1]*torch.outer(vectors[:,-1],vectors[:,-1])+values[0]*torch.outer(vectors[:,0],vectors[:,0])
        product=(torch.outer(a,b)+torch.outer(b,a))/2
        errors.append(float((product-leading).norm()/leading.norm()))
        errors.append(float(abs((centered-product).square().sum()/energy-(1-product_curve[0]))))
        cells.append(dict(relation=label,pairs=len(atlas['relations'][label]),readout_norm=float(readouts[j].norm()),
            radial_coefficient=float(alpha),folded_bias=float(folded_bias[j]),
            radial_energy_fraction=float(d*alpha.square()/q.square().sum()),
            traceless_frobenius_norm=float(energy.sqrt()),
            real_product_capture={str(k):float(product_curve[k-1]) for k in (1,2,4,8,16,32,64,128)},
            signed_square_capture={str(k):float(square_curve[k-1]) for k in (1,2,4,8,16,32,64,128)},
            products90=int(torch.searchsorted(product_curve,torch.tensor(.9)))+1,
            squares90=int(torch.searchsorted(square_curve,torch.tensor(.9)))+1))
        compact.append(dict(relation=label,radial=alpha,leading_product_a=a,leading_product_b=b,
            top16_square_readers=vectors[:,order[:16]].T,top16_square_coefficients=values[order[:16]]))
        matrices.append(centered/centered.norm())
    selected=[r for r in cells if r['relation'] in ('add_s','add_es','y_to_ies')]
    matrix_flat=torch.stack(matrices).flatten(1)
    torch.save(dict(labels=labels,physical_readouts=readouts,folded_down=folded,folded_bias=folded_bias,
        compact=compact,scope='Exact scalar programs require native Left/Right; compact factors approximate only traceless quadratic. '
        'Radial and bias retained; final residual/RMS/tanh interfaces remain outside this program.'),artifact)
    a=max(errors)<=1e-9
    result=dict(pred_a=a,pred_b=a and all(r['real_product_capture']['1']>=.9 for r in selected),
        pred_c=a and all(r['real_product_capture']['16']>=.9 for r in selected),replay_errors=errors,cells=cells,
        relations=labels,readout_cosines=(readouts@readouts.T).tolist(),traceless_function_cosines=(matrix_flat@matrix_flat.T).tolist(),
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_sha256=binding[checkpoint],scope='Weight-only full native scalar quadratic folds. Readout directions are normalized '
        'mean spelling contrasts, not fitted semantic classes. Spectral optima apply to coefficient Frobenius error at these fixed readouts, '
        'not reachable-state error, higher arithmetic reuse, full model circuits or an optimized output mixture. '
        'No approximate factors have been behaviorally validated or adopted.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','readout_cosines','traceless_function_cosines')},indent=2))
    print(json.dumps(cells,indent=2))


if __name__=='__main__':main()
