"""Validate frozen scalar factors on cached native states without fitting.
A exact folded output replay<=1e-5 relative with bound sources.
B top16 squares full and traceless relative RMS<=.25 for all three suffix
  readouts on both FineWeb and corpus-shift panels.
C morphology paired-change relative RMS<=.25 and sign agreement>=.9 for each
  suffix readout in each A1/A2 direction. No rank selection or fitted constants.
"""
import hashlib
import json
from pathlib import Path
import torch


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    root=Path(__file__).parent;output=root/'NATIVE_TOKEN_RELATION_TRACE_VALIDATION_V1.json'
    assert not output.exists()
    path=root/'NATIVE_TOKEN_RELATION_FOLD_V1.pt';result_path=root/'NATIVE_TOKEN_RELATION_FOLD_V1.json'
    result=json.loads(result_path.read_text());assert result['pred_a']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==result['artifact_sha256']
    saved=torch.load(path,weights_only=True,map_location='cpu')
    binding=json.loads((root/'FROZEN_BRANCH_MORPHOLOGY_V1_BINDING.json').read_text())['files']
    checkpoint=next(p for p in binding if p.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    left,right=[weights[f'transformer.h.17.mlp.{name}.weight'].double() for name in ('Left','Right')]
    suffixes=['add_s','add_es','y_to_ies'];panels={};paired=[];errors=[];sources=[path,result_path]
    for label,stem in [('fineweb','SHARED_NODE_PARENT1_SUPPRESSION_V1'),
                       ('corpus_shift','SHARED_NODE_PARENT1_CORPUS_SHIFT_V1'),
                       ('morphology','FROZEN_BRANCH_MORPHOLOGY_V1')]:
        cp=root/(stem+'_ENDPOINTS.pt');rp=root/(stem+'_RESULT.json')
        receipt=json.loads(rp.read_text())
        cache_hash=hashlib.sha256(cp.read_bytes()).hexdigest()
        # All three immutable native extraction receipts bind their cache bytes.
        expected=receipt.get('cache_sha256',receipt.get('cache',{}).get('sha256'))
        if expected is None:
            raise AssertionError(('missing cache hash',label,list(receipt)))
        assert cache_hash==expected
        sources.extend([cp,rp]);cache=torch.load(cp,weights_only=True,map_location='cpu')
        x=cache['ports']['input'].double()
        exact=((x@left.T)*(x@right.T))@saved['folded_down'].T+saved['folded_bias']
        reference=cache['ports']['native_output'].double()@saved['physical_readouts'].T
        errors.extend(((exact-reference).norm(dim=0)/reference.norm(dim=0)).tolist())
        records=[];predictions=[]
        for j,compact in enumerate(saved['compact']):
            static=compact['radial']*x.square().sum(1)+saved['folded_bias'][j]
            product=(x@compact['leading_product_a'])*(x@compact['leading_product_b'])
            squares=(x@compact['top16_square_readers'].T).square()@compact['top16_square_coefficients']
            product-=torch.dot(compact['leading_product_a'],compact['leading_product_b'])/x.shape[1]*x.square().sum(1)
            square_trace=(compact['top16_square_coefficients']*compact['top16_square_readers'].square().sum(1)).sum()
            squares-=square_trace/x.shape[1]*x.square().sum(1)
            def metrics(approx):
                delta=approx+static-exact[:,j]
                return dict(full_relative_rms=float(delta.norm()/exact[:,j].norm()),
                            traceless_relative_rms=float(delta.norm()/(exact[:,j]-static).norm()),
                            full_sign_agreement=float(((approx+static).sign()==exact[:,j].sign()).double().mean()))
            records.append(dict(relation=saved['labels'][j],endpoints=len(x),
                leading_product=metrics(product),top16_squares=metrics(squares)))
            predictions.append(squares+static)
        panels[label]=records
        if label=='morphology':
            row_path=root/(stem+'_ROWS.json');sources.append(row_path)
            rows=json.loads(row_path.read_text())['rows']
            assert cache['rows_sha256']==hashlib.sha256(row_path.read_bytes()).hexdigest()
            predicted=torch.stack(predictions,1)
            for family in ('A1','A2'):
                for direction in ('base_to_suffix','suffix_to_base'):
                    ids=torch.tensor([i for i,row in enumerate(rows) if row['family']==family and row['direction']==direction])
                    true_delta=exact[2*ids+1]-exact[2*ids]
                    pred_delta=predicted[2*ids+1]-predicted[2*ids]
                    for j,relation in enumerate(saved['labels']):
                        rms=float((true_delta[:,j]-pred_delta[:,j]).norm()/true_delta[:,j].norm())
                        sign=float((true_delta[:,j].sign()==pred_delta[:,j].sign()).double().mean())
                        paired.append(dict(family=family,direction=direction,relation=relation,
                            relative_rms=rms,sign_agreement=sign,n=len(ids),
                            native_change_mean=float(true_delta[:,j].mean()),predicted_change_mean=float(pred_delta[:,j].mean())))
    a=max(errors)<=1e-5 and all(torch.isfinite(torch.tensor(errors)))
    b=all(max(r['top16_squares']['full_relative_rms'],r['top16_squares']['traceless_relative_rms'])<=.25
          for label in ('fineweb','corpus_shift') for r in panels[label] if r['relation'] in suffixes)
    c=all(r['relative_rms']<=.25 and r['sign_agreement']>=.9 for r in paired if r['relation'] in suffixes)
    answer=dict(pred_a=bool(a),pred_b=bool(a and b),pred_c=bool(a and c),maximum_native_replay=max(errors),
        panels=panels,paired_changes=paired,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Frozen weight-only factors, no coefficient or rank fitting to these inputs. Exact native scalar readout '
        'with corrected approximation trace and retained bias; final residual background/RMS/tanh and physical circuit writes are not approximated or tested here. '
        'Existing panels are validation, not new independent/OOD confirmation or evidence of arbitrary mixture complexity.')
    output.write_text(json.dumps(answer,indent=2)+'\n')
    print(json.dumps({k:v for k,v in answer.items() if k not in ('sources','paired_changes')},indent=2))
    print(json.dumps([r for r in paired if r['relation']=='add_s'],indent=2))


if __name__=='__main__':main()
