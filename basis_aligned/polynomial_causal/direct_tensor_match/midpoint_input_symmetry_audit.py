"""Audit exact input-slot symmetry of the retained normalized bilinear function."""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();y=rows['y'].flatten(0,1).double();y-=y.mean(0);den=(y@S).norm();records=[]
original=torch.load(p/'MIDPOINT_GRAPH_FROZEN_V1.pt',weights_only=True)['rank8'];pruned=torch.load(p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt',weights_only=True)['products512'];shared=torch.load(p/'MIDPOINT_GRAPH_INPUT_MODE_GRAPHS_V1.pt',weights_only=True)['rank256']
for label,e in [('original1024',original),('shared1024',shared),('pruned512',pruned)]:
 e={k:v.double() for k,v in e.items()}
 if 'Pn' in e:A=e['Pn']@e['Tn'];B=e['Pm']@e['Tm']
 else:A=e['A'];B=e['B']
 if 'output_basis' in e:C=e['output_basis']@e['output_core']
 elif 'group_writers' in e:
  H=torch.zeros(256,1024,dtype=torch.float64);H[torch.arange(1024)//4,torch.arange(1024)]=1;C=e['group_writers']@H+e['correction_writers']@e['correction_left'].T
 else:C=e['reduced_writers']
 forward=(n@A)*(m@B);reverse=(m@A)*(n@B);sym=(forward+reverse)/2;pred=(forward-e['product_mean'])@C.T;reversepred=(reverse-reverse.mean(0))@C.T;sympred=(sym-sym.mean(0))@C.T
 W=S@C;Gc=W.T@W;Ga=A.T@A;Gb=B.T@B;Gab=A.T@B;energy=(Gc*Ga*Gb).sum();cross=(Gc*Gab*Gab.T).sum();anti=(energy-cross)/2
 records.append(dict(program=label,products=len(A.T),symmetrized_products=2*len(A.T),coefficient_antisymmetric_energy_fraction=float(anti/energy),full_calibration_error=float(((pred-y)@S).norm()/den),reverse_calibration_error=float(((reversepred-y)@S).norm()/den),symmetrized_calibration_error=float(((sympred-y)@S).norm()/den),original_predicted_swap_asymmetry=float((((forward-reverse)@C.T)@S).norm()/den)))
# Dense toy checks orthogonal decomposition and distance to a symmetric teacher.
g=torch.Generator().manual_seed(261242);t=torch.randn(3,5,5,generator=g,dtype=torch.float64);t=(t+t.transpose(1,2))/2;u=torch.randn(3,5,5,generator=g,dtype=torch.float64);sym=(u+u.transpose(1,2))/2;anti=(u-u.transpose(1,2))/2;identity=float(abs((u-t).square().sum()-(sym-t).square().sum()-anti.square().sum())/(u-t).square().sum());assert identity<1e-12
out=p/'MIDPOINT_INPUT_SYMMETRY_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,toy_projection_identity=identity,scope='Exactsymmetry of source-dependent bilinear function in already-normalized n,m slots. Not symmetry ofthe fullmodel underphysically swapping residualsources and recomputing normalization. Coefficientenergy usesisotropic inputslots and fixedvocabularycenteredoutputmetric; calibration usespairednative inputs. Meanrecentered separately forreverse/symmetric controls.'),indent=2)+'\n');print(out.read_text())
