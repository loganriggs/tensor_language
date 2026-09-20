"""Whole-output shared combinations, optimized in each frozen student's metric."""
import json,time
from pathlib import Path
import torch
from shared_quadratic_bank import bank_gram
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_default_dtype(torch.float64);start=time.perf_counter();source=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True);fit=json.load(open(P/'NATIVE_WEIGHTED_BANK_V1.json'))['records'];target=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];xs=[p['rows'].double() for p in torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']];i,j=torch.triu_indices(4,4);rows=[];selected={};negative=[];identity=[]
 for key,s in source['students'].items():
  U,V,C=[s[n].double() for n in ['U','V','C']];transform=source['metric_transforms'][key[0]];a,b=(U,V) if transform is None else (U@transform.double(),V@transform.double());G=bank_gram(a,b);G=(G+G.T)/2;ev,B=torch.linalg.eigh(G);negative.append(float(ev.min()/ev.max()));assert float(ev.min())>=-1e-10*float(ev.max());W,sigma,_=torch.linalg.svd(C@(B*ev.clamp_min(0).sqrt()),full_matrices=False);norm=((C.T@C)*G).sum();features=[]
  for x in xs:
   q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),4,4).sum(2);features.append(q[:,i]*q[:,j])
  for rank in [1,2,4,8]:
   writer=W[:,:rank];mix=writer.T@C;approx=writer@mix;delta=C-approx;error2=float(((delta.T@delta)*G).sum()/norm);tail=float(sigma[rank:].square().sum()/norm);discrepancy=abs(error2-tail);identity.append(discrepancy);assert discrepancy<1e-8;errors=[float((phi@mix.T@writer.T-y).norm()/y.norm()) for phi,y in zip(features,target)];row=dict(metric=key[0],optimizer=key[1],seed=key[2],output_features=rank,retained_student_energy=1-error2,student_relative_error=max(0,error2)**.5,empirical_calibration_error=errors[0],empirical_evaluation_error=errors[1],reduced_coefficients=36864+1162*rank,products=26,output_additions=9*rank+1152*(rank-1),tail_identity_discrepancy=discrepancy);rows.append(row)
   if key not in selected and 1-error2>=.99:selected[key]=dict(U=U,V=V,W=writer,Z=mix,retained_student_energy=1-error2)
 best=max((r for r in fit if r['metric']=='second_floor01'),key=lambda r:r['weighted_gain']);chosen=next(r for r in rows if (r['metric'],r['optimizer'],r['seed'],r['output_features'])==(best['metric'],best['optimizer'],best['seed'],2));predictions=dict(pred_a_identity=max(identity)<1e-8 and min(negative)>=-1e-10,pred_b_share=chosen['retained_student_energy']>=.99,pred_c_native=chosen['empirical_evaluation_error']-best['empirical_evaluation_error']<.02);out=dict(records=rows,selected_ranks=[dict(key=list(k),rank=s['W'].shape[1],retained_student_energy=s['retained_student_energy']) for k,s in selected.items()],selected_second_moment_rank2=chosen,predictions=predictions,seconds=time.perf_counter()-start,scope='Approximate sharedoutputcombination DAG edit. Frozentargetstudents, no native-refit; retainedenergymetricperoriginalfit. Nativeerrorsfromreusedcachedpanels, noOOD/semanticidentity.');torch.save(dict(students=selected,teacher_scale=source['teacher_scale']),P/'DAG_OUTPUT_SHARING_V1.pt');(P/'DAG_OUTPUT_SHARING_V1.json').write_text(json.dumps(out,indent=2)+'\n');print('selected second moment rank2',chosen);print('predictions',predictions);print('selected ranks',out['selected_ranks'])
if __name__=='__main__':main()
