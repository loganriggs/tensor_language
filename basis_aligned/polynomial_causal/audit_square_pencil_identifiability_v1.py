"""Two-slice spectral recovery of a fitted square tensor, not native discovery."""
from pathlib import Path
import torch,json
from scipy.optimize import linear_sum_assignment
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
torch.set_num_threads(2);dt=torch.float64
s=torch.load(P/'WEIGHT_SQUARE_POLISH_V1_FINAL.pt',weights_only=False,map_location='cpu');a=torch.nn.functional.normalize(s['model']['a'],dim=1).T;w=s['writer'];q,r=torch.linalg.qr(a,mode='reduced');n=a.shape[1]
sa=torch.linalg.svdvals(a);sw=torch.linalg.svdvals(w)
sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True);l,rr,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']];lp=l@q;rp=rr@q
rows=[]
for seed in [31,37,41]:
    torch.manual_seed(seed);b1,b2=torch.randn(2,w.shape[0],dtype=dt);c1=b1@w;c2=b2@w
    f1=(r*c1)@r.T;f2=(r*c2)@r.T
    pencil=torch.linalg.solve(f1.T,f2.T).T;ev,vec=torch.linalg.eig(pencil)
    imaginary=float(ev.imag.abs().max()/ev.abs().max());assert imaginary<=1e-8
    recovered=torch.nn.functional.normalize(q@vec.real,dim=0)
    similarity=(a.T@recovered).abs();ri,ci=linear_sum_assignment(-similarity.numpy());matched=similarity[ri,ci]
    # Full polynomial replay in residual-output coordinates after exactwriterfit.
    aa=a.T;ar=recovered.T;cross=w@product_cross(aa,aa,ar,ar);gram=product_cross(ar,ar,ar,ar);wr=torch.linalg.solve(gram,cross.T).T
    native_energy=((w.T@w)*product_cross(aa,aa,aa,aa)).sum();new_energy=((wr.T@wr)*gram).sum();inner=((w.T@wr)*product_cross(aa,aa,ar,ar)).sum();error=(native_energy+new_energy-2*inner)/native_energy
    nativeforms=[]
    for b in [b1,b2]:
        c=b@d;raw=(lp.T*c)@rp;nativeforms.append((raw+raw.T)/2)
    truepencil=torch.linalg.solve(nativeforms[0].T,nativeforms[1].T).T;nev=torch.linalg.eigvals(truepencil);complex_fraction=float((nev.imag.abs()>1e-7*nev.abs().clamp_min(1)).double().mean())
    ratios=(c2/c1).sort().values;spacing=(ratios[1:]-ratios[:-1]).abs()
    rows.append(dict(seed=seed,minimum_matched_reader_cosine=float(matched.min()),mean_matched_reader_cosine=float(matched.mean()),maximum_eigen_imaginary_relative=imaginary,minimum_ratio_spacing=float(spacing.min()),fitted_first_slice_condition=float(torch.linalg.cond(f1)),surrogate_relative_squared_replay_error=float(error),native_projected_slice_relative_errors=[float((x-y).norm()/x.norm()) for x,y in zip(nativeforms,[f1,f2])],native_pencil_complex_eigenvalue_fraction=complex_fraction,native_first_slice_condition=float(torch.linalg.cond(nativeforms[0]))))
# When output columns coincide, rotating equal-weight squares changes readers
# without changing the tensor: an explicit exception toidentifiability.
angle=.37;rot=torch.tensor([[__import__('math').cos(angle),-__import__('math').sin(angle)],[__import__('math').sin(angle),__import__('math').cos(angle)]],dtype=dt);collision=float((rot@rot.T-torch.eye(2,dtype=dt)).abs().max())
result=dict(rank=256,input_smallest_over_largest_singular=float(sa[-1]/sa[0]),writer_smallest_over_largest_singular=float(sw[-1]/sw[0]),numerical_full_column_rank_at_1e_minus10=bool(sa[-1]>1e-10*sa[0] and sw[-1]>1e-10*sw[0]),kruskal_sufficient_sum_if_exact_full_ranks=3*n,kruskal_required_sum=2*n+2,recovery=rows,collinear_writer_rotation_counterexample_error=collision,scope='Leurgans/Kruskal sufficientconditions for the exactfittedsquaretensor, numericalrank evidence notsymboliccertificate. NativeT remains~90.35%coefficienterror away; itsprojectedpencils are reported separately. No globalbestapproximation orsemanticcircuitidentification.')
(P/'SQUARE_PENCIL_IDENTIFIABILITY_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
