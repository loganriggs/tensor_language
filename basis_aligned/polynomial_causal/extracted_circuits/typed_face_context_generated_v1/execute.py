"""Generate the raw MLP8 context, retaining an explicit native RMS scale."""
import coupled
import attention8

def execute(p,current8,donor_city8,rho8,token_ids,recipient_token,donor_token,city,destination,strength=.5):
    a=attention8.execute(p['context'],current8,token_ids)
    g=(rho8.double()*current8.double()+a.double()).to(current8.dtype)
    return coupled.execute(p['local'],current8,donor_city8,g,recipient_token,donor_token,city,destination,strength)
