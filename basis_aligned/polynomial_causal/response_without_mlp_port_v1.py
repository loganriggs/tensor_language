"""Approximate fixed-writer response omitting pristine MLP normalization compensation."""
def predict(z,raw9,amplitude,direction,gain,left,right):
 z=z.double();a=amplitude.double()[...,None];d=direction.double();eps=1.1920928955078125e-7;rm=(z-a*d).square().mean(-1,keepdim=True)+eps;mixed=((z-a*d/2)@right.T)@left.T
 return raw9.double()+gain*(-a*d-a/rm*mixed)
