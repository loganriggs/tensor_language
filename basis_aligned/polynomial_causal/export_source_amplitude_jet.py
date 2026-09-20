"""Pack derivative coefficients and compare conventional quadratic baselines."""
import json,hashlib,sys,subprocess
from pathlib import Path
import numpy as np
from source_amplitude_runtime import PAIRS,effect
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    source=A/'semantic_source_jet_v2r1_result.json';r=json.loads(source.read_text());assert r['predictions']['pred_a_instrument']
    contexts={};audit=[];references=[];checks=[]
    for c in r['records']:
        g=np.array(c['gradient']);h=np.array(c['hessian']);h=(h+h.transpose(0,2,1))/2
        packed=np.concatenate([g,np.stack([h[:,i,j] for i,j in PAIRS],axis=-1)],axis=-1)
        keys=[f"{c['panel']}/{c['role']}/{c['family']}/{i}" for i in range(len(g))]
        for key,weights in zip(keys,packed):
            if key in contexts:assert contexts[key]==weights.tolist()
            contexts[key]=weights.tolist()
        av=np.array(r['plan']['amplitudes'][c['arm']],dtype=float);y=np.array(c['target']);den=max(np.linalg.norm(y),1e-30)
        linear=-g@av;diagonal=linear-.5*np.sum(np.diagonal(h,axis1=1,axis2=2)*av**2,axis=-1)
        eig,vec=np.linalg.eigh(h);idx=np.argmax(abs(eig),axis=-1);selected=vec[np.arange(len(g)),:,idx];value=eig[np.arange(len(g)),idx]
        rank1=linear-.5*value*(selected@av)**2;full=effect(packed,av)
        checks.append(float(np.max(np.abs(full-c['quadratic']))))
        audit.append(dict(panel=c['panel'],role=c['role'],family=c['family'],arm=c['arm'],relative_errors={name:float(np.linalg.norm(pred-y)/den) for name,pred in [('linear',linear),('diagonal',diagonal),('rank1',rank1),('full',full)]}))
        references.append(dict(keys=keys,amplitudes=av.tolist(),expected=c['quadratic']))
    artifact=dict(format='source_amplitude_quadratic_v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),contexts=contexts,coefficient_count=9*len(contexts),scope='Fixed text/site contexts; three native source amplitudes. New context requires full native prefix and derivative generator. Not standalone token prediction.',producer_price=r['plan'])
    target=P/'SOURCE_AMPLITUDE_QUADRATIC_V1.json';target.write_text(json.dumps(artifact,indent=2)+'\n')
    refs=P/'SOURCE_AMPLITUDE_QUADRATIC_V1_REPLAY.json';refs.write_text(json.dumps(references)+'\n')
    code="import json,sys,numpy as np;from pathlib import Path;from source_amplitude_runtime import effect;p=Path(sys.argv[1]);a=json.loads((p/'SOURCE_AMPLITUDE_QUADRATIC_V1.json').read_text());r=json.loads((p/'SOURCE_AMPLITUDE_QUADRATIC_V1_REPLAY.json').read_text());error=max(float(np.max(np.abs(effect([a['contexts'][k] for k in c['keys']],c['amplitudes'])-c['expected']))) for c in r);assert 'torch' not in sys.modules;assert error<1e-10;print(error)"
    replay=subprocess.run([sys.executable,'-c',code,str(P)],cwd=P,capture_output=True,text=True,check=True)
    result=dict(independent_cpu_replay_error=float(replay.stdout.strip()),contexts=len(contexts),coefficient_values=artifact['coefficient_count'],artifact_bytes=target.stat().st_size,baseline_values_per_context=dict(linear=3,diagonal=6,rank1=7,full=9),max_errors_by_arm={arm:{name:max(c['relative_errors'][name] for c in audit if c['arm']==arm) for name in ['linear','diagonal','rank1','full']} for arm in r['plan']['amplitudes']},records=audit)
    (P/'SOURCE_AMPLITUDE_QUADRATIC_CPU_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
