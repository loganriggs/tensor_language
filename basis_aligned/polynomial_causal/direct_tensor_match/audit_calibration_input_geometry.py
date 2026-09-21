"""Compare calibration and opened inputs in the actual fitted coefficient geometry."""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);e=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True)
z=torch.cat([d['z'][:1536],e['z'].flatten(0,1).double()]);x=(z-d['mu'])@d['inverse_root'];held=(d['z'][d['indices']]-d['mu'])@d['inverse_root']
S=x.T@x/len(x);white_eig=torch.linalg.eigvalsh(S)
# Whitening makes S approximately identity, so its individual eigenvectors are arbitrary.
# Use raw-input covariance axes to define meaningful low-variance groups instead.
delta=z-d['mu'];opened_delta=d['z'][d['indices']]-d['mu']
eig,U=torch.linalg.eigh(delta.T@delta/len(delta));trainenergy=eig.clamp_min(0);openedenergy=(opened_delta@U).square().mean(0)
rows=[]
for n in [16,64,128,256,512,1152]:
 rows.append(dict(lowest_training_axes=n,train_energy=float(trainenergy[:n].sum()),opened_energy=float(openedenergy[:n].sum()),opened_to_train_ratio=float(openedenergy[:n].sum()/trainenergy[:n].sum()),opened_energy_fraction=float(openedenergy[:n].sum()/openedenergy.sum())))
stats={}
for name,a in [('calibration16384',x),('opened448',held)]:
 radius=a.square().sum(1);stats[name]=dict(mean_squared_radius=float(radius.mean()),radius_quantiles=torch.quantile(radius,torch.tensor([.1,.5,.9,.99],dtype=radius.dtype)).tolist())
out=dict(input_statistics=stats,whitened_training_second_moment_extrema=[float(white_eig[0]),float(white_eig[-1])],raw_training_second_moment_extrema=[float(eig[0]),float(eig[-1])],low_energy_axis_comparison=rows,scope='Diagnostic of fixed input coordinates used by coefficient fitting. Opened448 selection has been repeatedly inspected. Not a fresh test or evidence of semantic identity; directions are never refit to opened data.')
(P/'CALIBRATION_INPUT_GEOMETRY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
