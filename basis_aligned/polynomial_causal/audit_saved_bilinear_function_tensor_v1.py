"""Saved-weight functional tensor audit; no new native capability or causal claim."""
import hashlib,json,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json'
OUT=Path(__file__).with_name('SAVED_BILINEAR_FUNCTION_TENSOR_V1_RESULT.json')

def main():
    torch.set_num_threads(2);started=time.perf_counter();r=json.loads(SOURCE.read_text())
    assert r['predictions']['pred_d_exact_checkpoint_writer_reader_and_M11_tensor_contractions_close']
    def tensor(record):return torch.tensor(record['values'],dtype=torch.float64).reshape(record['shape'])
    c=r['contractions'];left=tensor(c['M11_left']);right=tensor(c['M11_right']);down=tensor(c['M11_down'])
    full=torch.einsum('oh,hi,hj->oij',down,left,right);stored=tensor(c['M11_tensor'])
    sym=(full+full.transpose(-1,-2))/2;anti=(full-full.transpose(-1,-2))/2
    selected=torch.tensor(r['M11_factor_order'][:32]);top=torch.einsum('oh,hi,hj->oij',down[:,selected],left[selected],right[selected]);top_sym=(top+top.transpose(-1,-2))/2
    names=sorted(c['writers']);assert names==['L07H07','L09H04']
    a,b=(tensor(c['writers'][name]) for name in names)
    cross=torch.einsum('ia,oij,jb->oab',a,2*sym,b)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60913);x=torch.randn(19,8,dtype=torch.float64);za=torch.randn(19,128,dtype=torch.float64);zb=torch.randn(19,128,dtype=torch.float64)
    evaluate=lambda v:((v@left.T)*(v@right.T))@down.T
    bilinear=lambda v:torch.einsum('bi,oij,bj->bo',v,sym,v)
    da=za@a.T;db=zb@b.T
    lhs=evaluate(x+da+db)-evaluate(x+da)-evaluate(x+db)+evaluate(x)
    rhs=torch.einsum('bi,oij,bj->bo',za,cross,zb)
    norm=lambda t:float(t.norm())
    controls={'factor_formula':float((evaluate(x)-bilinear(x)).abs().max())<1e-9,
        'antisymmetric_zero':float(torch.einsum('bi,oij,bj->bo',x,anti,x).abs().max())<1e-9,
        'folded_writer_cross_identity':float((lhs-rhs).abs().max())<1e-8,
        'mixed_writer_term_live':float(rhs.norm())>1e-5}
    result={'scope':'Exact-form FP64 audit of stored restricted U8 factors; full input coverage, normalization closure and fresh causal sufficiency are not established.',
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'checks':controls,'passed':all(controls.values()),
        'shape':list(full.shape),'stored_FP32_vs_factor_FP64_max_abs':float((full-stored).abs().max()),
        'functional_replay_max_abs':float((evaluate(x)-bilinear(x)).abs().max()),'folded_writer_cross_max_abs':float((lhs-rhs).abs().max()),
        'antisymmetric_coefficient_norm_fraction':norm(anti)/norm(full),'symmetrized_top32_relative_residual':norm(sym-top_sym)/norm(sym),
        'original_unsymmetrized_top32_relative_residual':r['M11_top32_tensor_relative_residual'],
        'writers':names,'cross_tensor_shape':list(cross.shape),'cross_tensor_norm':norm(cross),
        'reader_symmetric_eigenvalues':[torch.linalg.eigvalsh(q).tolist() for q in sym],
        'model_forwards':0,'model_updates':0,'wall_seconds':time.perf_counter()-started}
    assert not OUT.exists();OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reader_symmetric_eigenvalues'},indent=2))

if __name__=='__main__':main()
