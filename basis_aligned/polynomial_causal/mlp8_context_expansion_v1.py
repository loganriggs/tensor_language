"""All ordered source terms of the exact normalized MLP8 response."""
import torch
@torch.no_grad()
def analyze(sources,fixtures,program,head8):
 torch.set_num_threads(2);p=program['mlp8'];L=p['left'].double();R=p['right'].double();D=p['down'].double();eps=torch.finfo(torch.float32).eps
 closures=[];native_errors=[];stats={};denominator=0.;labels=['residual7','initial8','attention8']
 for source,f in zip(sources,fixtures):
  x=f['inputs'];g=source.double().sum(0);delta=.5*head8.execute(program['head8'],x['current8'],x['donor_city8'],x['recipient_token'],x['donor_token'],x['city'],x['destination']).double()
  s0=g.square().mean(-1,keepdim=True)+eps;s1=(g+delta).square().mean(-1,keepdim=True)+eps
  ls=[z.double()@L.T for z in source];rs=[z.double()@R.T for z in source];ld=delta@L.T;rd=delta@R.T
  terms={}
  for i,a in enumerate(labels):
   for j,b in enumerate(labels):terms[f'norm:{a}*{b}']=(s0/s1-1)*((ls[i]*rs[j])@D.T)/s0
   terms[f'cross:delta*{a}']=((ld*rs[i])@D.T)/s1
   terms[f'cross:{a}*delta']=((ls[i]*rd)@D.T)/s1
  terms['quadratic']=((ld*rd)@D.T)/s1;terms['skip']=delta
  actual=delta+(((g+delta)@L.T)*((g+delta)@R.T))@D.T/s1-((g@L.T)*(g@R.T))@D.T/s0
  summed=sum(terms.values());closures.append(float((summed-actual).norm()/actual.norm()));native=f['expected_native_delta'];native_errors.append(float((summed-native).norm()/native.norm()))
  denominator+=float(actual.square().sum())
  for name,t in terms.items():
   acc=stats.setdefault(name,[0.,0.]);acc[0]+=float(t.square().sum());acc[1]+=float((t*actual).sum())
 return {'pred_b':len(closures)==40 and max(closures)<=1e-10,'pred_c':len(native_errors)==40 and max(native_errors)<=1e-4,'max_ordered_closure':max(closures),'max_native_delta_error':max(native_errors),'term_count':len(stats),'terms':{k:{'norm_ratio':(v[0]/denominator)**.5,'aligned_fraction':v[1]/denominator} for k,v in stats.items()},'scope':'Exact ordered source expansion on opened states. Shared denominators retain all sources. Three supplied source arrays replace one g array; no port closure, source omission, causal adoption, or simplicity gain.'}
