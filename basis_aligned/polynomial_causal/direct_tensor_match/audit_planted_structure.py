"""Functional recovery versus recovery of planted feature spaces and supports."""
import json
from pathlib import Path
import torch
from core import Model,metric,inner,normrows
from sweep import target_cases

def teachers():
 torch.manual_seed(420);result={}
 for name,kind,width,degree in [('coordinate_sparse','monomial',1,2),('bilinear_cp','cp',3,2),('sparse_tucker','tucker',3,2),('quartic_tree','tree',2,4),('shared_quartic_dag','dag',3,4)]:
  m=Model(6,3,kind,width,degree)
  with torch.no_grad():
   if name=='coordinate_sparse':
    m.weight.zero_();m.weight[0,0]=1.;m.weight[1,6]=-1.;m.weight[2,12]=.7;m.weight[0,20]=.3
   if name=='sparse_tucker':
    m.core.zero_();m.core[0,0]=1.;m.core[0,4]=-.6;m.core[1,1]=.8;m.core[1,5]=.5
   if name=='shared_quartic_dag':m.weight[:,3:]=0
  result[name]=m
 return result

def features(m):
 if m.kind=='cp':return m.atoms().detach(),metric(6,2)
 if m.kind=='tucker':return normrows(m.leaf).detach(),torch.eye(6,dtype=torch.float64)
 if m.kind=='tree':return torch.cat([normrows(m.left),normrows(m.right)]).detach(),metric(6,2)
 if m.kind=='dag':return normrows(m.bank).detach(),metric(6,2)

def main():
 torch.set_num_threads(1);p=Path(__file__).resolve().parent;data=json.loads((p/'TOY_SWEEP_V1.json').read_text());saved=torch.load(p/'TOY_SWEEP_V1.pt',weights_only=True,map_location='cpu');truth=teachers();cases=target_cases();rows=[]
 for case in cases:
  teacher=truth[case['name']];c=teacher().detach();c=c/inner(c,c,metric(6,case['degree'])).sqrt();assert (c-case['target']).abs().max()<1e-12
  idx=min((i for i,r in enumerate(data['records']) if r['case']==case['name']),key=lambda i:data['records'][i]['relative_error']);r=data['records'][idx];m=Model(6,3,r['kind'],r['width'],case['degree']);m.load_state_dict(saved[idx]['state'])
  report=dict(case=case['name'],best_loss=r['relative_error'],student_width=r['width'],teacher_width=case['width'],optimizer=r['optimizer'])
  if m.kind=='monomial':
   mask=m.weight.abs()>=m.weight.abs().max()*1e-3;reference=teacher.weight!=0;report.update(thresholded_support_exact=bool(torch.equal(mask,reference)),thresholded_support=int(mask.sum()),teacher_support=int(reference.sum()))
  else:
   a,M=features(teacher);b,_=features(m);chol=torch.linalg.cholesky(M);a=a@chol;b=b@chol
   _,sv,vh=torch.linalg.svd(b,full_matrices=False);rank=int((sv>sv[0]*1e-8).sum());basis=vh[:rank].T
   report.update(teacher_feature_energy_outside_student_span=float((a-a@basis@basis.T).norm()/a.norm()),student_feature_rank=rank)
   if m.kind=='tucker':report.update(teacher_core_nonzeros=int((teacher.core!=0).sum()),student_core_thresholded_nonzeros=int((m.core.abs()>m.core.abs().max()*1e-3).sum()))
  rows.append(report)
 out=p/'PLANTED_STRUCTURE_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=rows,scope='Best-run posthoc diagnosis. Span inclusion permits rotations and does not prove unique atom identification; wider spans can include extra directions. Threshold support is a diagnostic, not a deployed sparse program.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
