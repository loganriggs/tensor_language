"""Replace some shared mixed products with twice as many private squares at fixed projection cost."""
from pathlib import Path
import json,torch
from global_mixed_source_graph import export,score,source_reads
from shared_quadratic_products import materialize_mixed
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'GLOBAL_SOURCE_BALANCE_V1.json').read_text());key=meta['winners']['0.5'];base=torch.load(P/'GLOBAL_SOURCE_BALANCE_PROGRAMS_V1.pt',weights_only=True)[key]
root=torch.linalg.inv(d['inverse_root']);L=root@base['left_reader'];R=root@base['right_reader'];W=base['product_weights'].T/d['scales'][:,None];T=d['teacher'];F=T.flatten(1);ev,U=torch.linalg.eigh(F@F.T);B=(U*ev.pow(-.25))@U.T
energy=.5*(L.square().sum(0)*R.square().sum(0)+(L*R).sum(0).square())*(B@W).square().sum(0)
order=energy.argsort();records=[];programs={}
for k in [0,8,16,32]:
 keep=order[k:];l=L[:,keep];r=R[:,keep];w=W[:,keep];hat=materialize_mixed(l,r,w)
 if k:
  residual=T[5]-hat[5];e,V=torch.linalg.eigh(residual);ix=e.abs().argsort(descending=True)[:2*k];v=V[:,ix];sweights=e[ix]
  ww=torch.zeros((6,2*k),dtype=T.dtype);ww[5]=sweights
  big=export(torch.cat([l,v],1),torch.cat([r,v],1),torch.cat([w,ww],1),d)
 else:big=export(l,r,w,d)
 m=len(keep);p={name:val.clone() for name,val in big.items()}
 p['left_reader']=big['left_reader'][:,:m].clone();p['right_reader']=big['right_reader'][:,:m].clone();p['product_weights']=big['product_weights'][:m].clone()
 p['square_reader']=big['left_reader'][:,m:].clone();p['square_weights']=big['product_weights'][m:,5].clone();p['square_output']=torch.tensor(5)
 z=d['z'][:32];actual=source_reads(z,p);actual[:,5]+=(z@p['square_reader']).square()@p['square_weights'];expected=source_reads(z,big);replay=float((actual-expected).norm()/expected.norm());assert replay<1e-10
 floats=sum(v.numel() for v in p.values() if v.is_floating_point());assert floats==896262-4*k
 after=materialize_mixed(l,r,w)
 if k:after[5]+=(v*sweights)@v.T
 records.append(dict(removed_mixed=k,added_private_squares=2*k,source_products=m+2*k,stored_floats=floats,original_coefficient_error=float((after-T).norm()/T.norm()),native_replay=replay,**score(big,d)))
 programs[str(k)]=p
out=dict(parent=key,selection='Remove lowest individual atom energy under parent output metric; exact truncated spectral residual correction to sixth source read. No state-based candidate selection or refit.',records=records,primary_removed=16,predictions=dict(pred_a_execution=max(r['native_replay'] for r in records)<1e-10,pred_b_value_repair=all(a<=.15 and a<=1.1*b for a,b in zip(records[2]['per_mode_errors'],meta['plan']['scalar_baseline'])),pred_c_price=records[2]['stored_floats']<=896262 and records[2]['source_products']<512),scope='Opened-state graph-edit screen with actual compact private-square serialization; same input-projection coefficient budget, no fresh model or semantic validation.')
torch.save(programs,P/'GLOBAL_PRIVATE_RESIDUAL_PROGRAMS_V1.pt');(P/'GLOBAL_PRIVATE_RESIDUAL_EDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
