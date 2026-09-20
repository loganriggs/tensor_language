"""Score total and conditional effects without hiding the weaker edit axis."""
import numpy as np


def key(coordinate):
    return ','.join(f'{v:g}' for v in coordinate)


def score(groups,budgets):
    lookup={(b['panel'],b['family']):b for b in budgets}
    records=[];numerical=[]
    for group in groups:
        native={k:np.array(v) for k,v in group['target'].items()}
        double={k:np.array(v) for k,v in group['double'].items()}
        predictions={m:{k:np.array(v) for k,v in values.items()} for m,values in group['predictions'].items()}
        # An oracle control: it knows the true subject-only effect but ignores
        # the attractor. It is not an executable predictive baseline.
        predictions['ignore_attractor_oracle']={k:native[key((float(k.split(',')[0]),0))] for k in native}
        for s,t in group['coordinates']:
            if s==0 or t==0:
                continue
            joint=key((s,t));subject=key((s,0));attractor=key((0,t))
            for family in dict.fromkeys(group['families']):
                ids=[i for i,f in enumerate(group['families']) if f==family]
                b=lookup[group['panel'],family];ns=b['subject_number_norm'];na=b['attractor_number_norm']
                for kind,axis,budget in [('joint','0,0',abs(s)*ns+abs(t)*na),
                                          ('attractor_increment',subject,abs(t)*na),
                                          ('subject_increment',attractor,abs(s)*ns)]:
                    y=(native[joint]-native[axis])[ids]
                    yd=(double[joint]-double[axis])[ids]
                    numerical.append(float(max(np.linalg.norm(y-yd,axis=0))/budget))
                    for mode,pred in predictions.items():
                        yh=(pred[joint]-pred[axis])[ids];errors=np.linalg.norm(yh-y,axis=0)
                        records.append(dict(panel=group['panel'],family=family,coordinate=[s,t],kind=kind,mode=mode,
                                            number_error=float(errors[0]/budget),modal_error=float(max(errors[1:])/budget),
                                            absolute_errors=errors.tolist(),target_norms=np.linalg.norm(y,axis=0).tolist(),budget=float(budget)))
    return records,max(numerical)


def control():
    coordinates=[[1,0],[0,1],[1,1]]
    target={'0,0':[[0.,0.,0.,0.]]}
    for s,t in coordinates:
        target[key((s,t))]=[[10*s+.1*t+.03*s*t,0.,0.,0.]]
    group=dict(panel='toy',families=['toy'],coordinates=coordinates,target=target,double=target,predictions={'exact':target})
    records,numerical=score([group],[dict(panel='toy',family='toy',subject_number_norm=10.,attractor_number_norm=.1)])
    joint=next(r for r in records if r['mode']=='ignore_attractor_oracle' and r['kind']=='joint')
    increment=next(r for r in records if r['mode']=='ignore_attractor_oracle' and r['kind']=='attractor_increment')
    assert joint['number_error']<.1 and increment['number_error']>1.
    assert all(r['number_error']==0 for r in records if r['mode']=='exact') and numerical==0
    return dict(ignored_axis_joint_error=joint['number_error'],ignored_axis_increment_error=increment['number_error'],exact_replay_error=numerical)


if __name__=='__main__':
    print(control())
