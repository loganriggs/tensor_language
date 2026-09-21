from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
e=torch.load(p/'MIDPOINT_CHANNEL_REFIT_V1.pt',weights_only=True)['correlation'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].double();m=rows['m'].double();y=rows['y'].double();S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();L=e['L'].double();R=e['R'].double();phi=(n@L.T)*(m@R.T)+(n@R.T)*(m@L.T);rates=[0.,1e-4,.001,.01,.1,1.];records=[]
def fit(x,y,rate):
 xm=x.mean(0);ym=y.mean(0);x=x-xm;y=y-ym;sd=x.square().mean(0).sqrt().clamp_min(1e-12);z=x/sd;gram=z.T@z/len(z);cross=z.T@y/len(z);v,Q=torch.linalg.eigh(gram);keep=v>v[-1]*1e-10 if rate==0 else torch.ones_like(v,dtype=torch.bool);w=((Q[:,keep]/(v[keep]+rate))@(Q[:,keep].T@cross))/sd[:,None];return w,xm,ym
for split in [0,1]:
 train=torch.arange(16)+16*split;valid=torch.arange(16)+16*(1-split);x=phi[train].flatten(0,1);target=y[train].flatten(0,1);vx=phi[valid].flatten(0,1);vy=y[valid].flatten(0,1)
 for rate in rates:
  w,xm,ym=fit(x,target,rate);pred=(vx-xm)@w+ym;err=float(((pred-vy)@S).norm()/((vy-ym)@S).norm());records.append(dict(split=split,ridge=rate,validation_error=err))
means={rate:sum(v['validation_error'] for v in records if v['ridge']==rate)/2 for rate in rates};chosen=min(means,key=means.get);w,xm,ym=fit(phi.flatten(0,1),y.flatten(0,1),chosen);program=dict(e);program['channel_mean']=xm;program['full_mean']=ym;program['reduced_writers']=w.T
out=p/'MIDPOINT_CHANNEL_RIDGE_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,mean_validation_error=means,chosen_ridge=chosen,scope='Conditional output-refit validation on complementary originalcalibration16document halves. Reader/channel selection was fixed using fullcalibration; these are not independent dictionary-validation data. Rates chosen without native diagnostic outcomes. Wholefullcalibration refit exported; native evidence pending.'),indent=2)+'\n');torch.save({'ridge':program},p/'MIDPOINT_CHANNEL_RIDGE_V1.pt');print(out.read_text())
