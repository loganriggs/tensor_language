from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
base=torch.load(p/'MIDPOINT_COVERAGE_256_R4_V1.pt',weights_only=True);e=torch.load(p/'MIDPOINT_PRODUCT_ANCHORED_REFIT_V1.pt',weights_only=True)['anchored'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();phi=(n@e['A'].double())*(m@e['B'].double())-e['product_mean'];prior=base['readout'].double()@base['reduced_writers'].double().T;delta=e['reduced_writers'].T-prior;target=phi@delta@S;_,s,Vh=torch.linalg.svd(target,full_matrices=False);records=[];programs={};graphs={}
y=rows['y'].flatten(0,1).double()-e['full_mean'];norm=(y@S).norm()
for rank in [0,1,4,8,16,32,64,128]:
 V=Vh[:rank].T;left=delta@S@V;writer=torch.linalg.solve(S,V);weight=prior+left@writer.T;prediction=phi@weight;relative=float(((prediction-y)@S).norm()/norm);correctionerror=float(((phi@delta-phi@left@writer.T)@S).norm()/target.norm());cost=2359296+294912+rank*(1024+1152)
 records.append(dict(rank=rank,full_calibration_variation_error=relative,correction_relative_error=correctionerror,weight_coefficients=cost,products=1024))
 if rank in [8,32]:
  prog=dict(e);prog['reduced_writers']=weight.T;programs[f'rank{rank}']=prog;graphs[f'rank{rank}']=dict(A=e['A'],B=e['B'],product_mean=e['product_mean'],full_mean=e['full_mean'],group_writers=base['reduced_writers'],correction_left=left,correction_writers=writer)
  direct=prediction;graph=phi.reshape(len(phi),256,4).sum(-1)@base['reduced_writers'].T+(phi@left)@writer.T;assert float((direct-graph).norm()/direct.norm())<1e-12
out=p/'MIDPOINT_OUTPUT_CORRECTION_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Originalcalibration weighted low-rank approximation of anchoredwriter correction. Adds reusable linear combinations of existing products; no new products. Export rank8/32 with exact graph/flat replay; native evidence pending.'),indent=2)+'\n');torch.save(programs,p/'MIDPOINT_OUTPUT_CORRECTION_V1.pt');torch.save(graphs,p/'MIDPOINT_OUTPUT_CORRECTION_GRAPHS_V1.pt');print(out.read_text())
