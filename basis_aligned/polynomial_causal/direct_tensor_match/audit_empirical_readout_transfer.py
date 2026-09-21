"""Out-of-readout-fit diagnostic on the pre-existing expanded covariance calibration pool."""
from pathlib import Path
import json,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
e=torch.load(P/'EXPANDED_COVARIANCE_STATES_V1.pt',weights_only=True);z=e['z'].flatten(0,1).double();t=e['t'].flatten().double();s=e['scale'].flatten().double();truth=e['native_phi'].flatten().double();den=(truth-truth.mean()).norm()
programs=torch.load(P/'EMPIRICAL_PAIR_READOUT_PROGRAMS_V1.pt',weights_only=True);base=programs['downstream_0'];mixed=(z@base['left_reader'])*(z@base['right_reader']);squares=(z@base['square_reader']).square();records=[]
for key,p in programs.items():
 assert all(torch.equal(p[k],base[k]) for k in ['left_reader','right_reader','square_reader'])
 reads=mixed@p['product_weights'][:,4:]+z@p['source_linear'][:,4:]+p['source_bias'][4:];reads[:,1]+=squares@p['square_weights']
 phi=((t-.5*reads[:,0])/s-p['alpha'][2])*(reads[:,1]/s-p['beta'][2]);error=phi-truth
 records.append(dict(key=key,component3_error=float(error.norm()/den),mean_squared_error=float(error.square().mean()),max_chunk_error_energy_share=float(error.reshape(232,64).square().sum(1).max()/error.square().sum())))
control=next(r for r in records if r['key']=='downstream_0');primary=next(r for r in records if r['key']=='downstream_0.1')
out=dict(records=records,primary='downstream_0.1',primary_gain=1-primary['component3_error']/control['component3_error'],predictions=dict(pred_a_transfer=primary['component3_error']<=.9*control['component3_error']),scope='Additional232training-prefix calibration rows excluded from original32prefixes. Already used for input covariance, not fresh or fully held-out. This readout fit used onlyold24trainingprefixes. Captures retain t=h@a onlyforcomponent3, so cannot claim validationofcomponents1/2or fullgoal.')
(P/'EMPIRICAL_PAIR_TRANSFER_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
