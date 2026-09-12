"""Four-mode value numerator with an explicit raw-input norm port."""
def encode(raw,program):
 raw=raw.double()
 return raw@program['readers'].T,raw.square().sum(-1)

def execute(readings,raw_norm2,program):
 b,a,c,d=readings.unbind(-1)
 numerator=(b+a)*(b-a)-c.square()-d.square()
 return numerator/(raw_norm2/program['dimension']+program['epsilon'])
