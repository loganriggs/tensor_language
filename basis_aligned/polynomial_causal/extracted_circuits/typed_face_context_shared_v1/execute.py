"""Generate the raw MLP8 context, retaining an explicit native RMS scale."""
import coupled
import attention8

def execute(p,current8,donor_city8,rho8,token_ids,recipient_token,donor_token,city,destination,strength=.5):
    a=attention8.execute(p['context'],current8,token_ids)
    g=(rho8.double()*current8.double()+a.double()).to(current8.dtype)
    sl=slice(256,384);head=dict(p['local']['head8']);context=p['context']
    for h,c in [('q1','q1'),('k1','k1'),('q2','q2'),('k2','k2'),('current_value','value')]:head[h]=context[c][sl]
    head['output']=context['output'][:,sl];head['mixture']=context['mixture']
    local={'head8':head,'mlp8':p['local']['mlp8']}
    return coupled.execute(local,current8,donor_city8,g,recipient_token,donor_token,city,destination,strength)
