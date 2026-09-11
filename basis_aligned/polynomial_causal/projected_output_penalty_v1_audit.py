"""Exact conditional component-energy penalty path on final seed0 features.

A parent/normal/CP replay<=1e-8; B >=4x component-energy reduction at<=.001
capture loss; C lambda1e-4 full sparse stationarity improves>=10x.
Changes the objective, not a same-objective convergence repair.
"""
import hashlib,json,time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P,CK
from folded_sparse_dictionary_v1 import decode
from joint_quadratic_fit_v1 import product_cross
from chunked_bilinear_coefficient_v1 import value_gradient
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'PROJECTED_OUTPUT_PENALTY_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    parent=json.loads((P/'PROJECTED_SPARSE_DICTIONARY_FIT_V1_SEED_0.json').read_text());source=parent['cache']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    saved=torch.load(source['path'],weights_only=True,map_location='cpu')
    ids=saved['code_indices'].long();values=saved['code_values'].double()
    readers,basis,code,_=decode(saved['analysis_basis'].double(),ids,values,torch.ones(len(ids)));a,b=readers.chunk(2)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    scale=torch.cat((l,r)).norm(dim=1);normalized_values=values/scale[:,None]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;native=(l,r,wh@down)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    gram=product_cross(a,b,a,b);norms=gram.diag().sqrt();normalized=gram/norms[:,None]/norms[None,:]
    eig,vec=torch.linalg.eigh((normalized+normalized.T)/2);assert eig[0]>0
    rhs=native[2]@product_cross(l,r,a,b);h=(rhs/norms[None,:])@vec
    rows=[];selected=None;selected_diagnostics=None
    for penalty in (0.,1e-8,1e-6,1e-4,1e-2):
        coefficients=h/(eig[None,:]+penalty)
        w=(coefficients@vec.T)/norms[None,:]
        function=float((coefficients.square()*eig).sum()/total)
        cross=float((coefficients*h).sum()/total)
        component=float(coefficients.square().sum()/total)
        capture=2*cross-function
        normal=float((w@gram+penalty*w*norms.square()[None,:]-rhs).norm()/rhs.norm())
        row=dict(penalty=penalty,capture=capture,capture_loss=parent['final_capture']-capture,
            component_energy=component,component_reduction=parent['optimization']['final_details']['component_energy']/component,
            regularized_normal_residual=normal,penalized_objective=1-capture+penalty*component)
        rows.append(row);print(json.dumps(row),flush=True)
        if penalty==1e-4:selected=w;selected_diagnostics=row
    parent_replay=max(abs(rows[0]['capture']-parent['final_capture']),
                      abs(rows[0]['component_energy']-parent['optimization']['final_details']['component_energy']))
    cp=(a,b,selected)
    cp_capture=1-float((inner(cp,cp)-2*inner(native,cp)+total)/total)
    cp_replay=abs(cp_capture-selected_diagnostics['capture'])
    objective,(ga,gb,_),details=value_gradient(*native,a,b,selected,total,penalty=1e-4,chunk=256)
    reader_gradient=torch.cat((ga,gb))
    dictionary_gradient=torch.sparse.mm(code.transpose(0,1),reader_gradient)
    tangent=dictionary_gradient-(dictionary_gradient*basis).sum(1,keepdim=True)*basis
    code_gradient=((reader_gradient*scale[:,None])@basis.T).gather(1,ids)
    denominator=abs(float(objective))
    dictionary_stationarity=float(tangent.norm()*len(basis)**.5/denominator)
    code_stationarity=float(code_gradient.norm()*normalized_values.norm()/denominator)
    stationarity=max(dictionary_stationarity,code_stationarity)
    old=max(parent['optimization']['final_details']['dictionary_relative_stationarity'],
            parent['optimization']['final_details']['codes_relative_stationarity'])
    gradient_loss_replay=abs(float(objective)-selected_diagnostics['penalized_objective'])
    physical=torch.linalg.solve_triangular(wh,selected,upper=True)
    cache=Path('/dev/shm/bilin18_projected_output_penalty_v1.pt');assert not cache.exists()
    torch.save(dict(analysis_basis=basis,code_indices=ids.to(torch.int16),code_values=values,
        down=physical,retained_bias_key=saved['retained_bias_key'],source=source,penalty=1e-4),cache)
    result=dict(predictions=dict(pred_a_instrument=max(parent_replay,cp_replay,gradient_loss_replay,
                      max(row['regularized_normal_residual'] for row in rows))<=1e-8,
        pred_b_low_cost_cancellation=any(row['penalty']>0 and row['component_reduction']>=4 and row['capture_loss']<=.001 for row in rows),
        pred_c_stationarity=stationarity<=old/10),rows=rows,parent_replay=parent_replay,cp_replay=cp_replay,
        gradient_loss_replay=gradient_loss_replay,normalized_product_gram_eigen_min=float(eig[0]),
        normalized_product_gram_eigen_max=float(eig[-1]),
        selected_stationarity=dict(dictionary=dictionary_stationarity,codes=code_stationarity,combined=stationarity,
            unregularized_parent=old,reduction=old/stationarity),source=source,
        cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),seconds=time.perf_counter()-start,
        scope='Exact output-only penalty path, fixed learned features/codes. Penalized joint gradient at '
              'one point is not solver convergence or a global minimum. No live-fit/data changes.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
