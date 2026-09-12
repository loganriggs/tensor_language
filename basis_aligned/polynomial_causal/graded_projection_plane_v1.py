"""Global candidate search on a chosen producer-subspace rotation plane.

The exact objective is a degree-four trigonometric polynomial in phi=2*theta.
Numerical roots are candidate points; interpolation/direct replay must be checked.
This does not globally optimize the entire Grassmann manifold.
"""
import numpy as np

def search(evaluate):
    angles=np.arange(9)*np.pi/9
    values=np.array([float(evaluate(t)) for t in angles])
    coeff=np.fft.fft(values)/9
    frequencies=np.array([0,1,2,3,4,-4,-3,-2,-1])
    # Multiply derivative in z=exp(2i theta) by z^4 to get degree <=8.
    polynomial=np.zeros(9,dtype=complex)
    for frequency,c in zip(frequencies,coeff):polynomial[frequency+4]+=1j*frequency*c
    roots=np.roots(np.trim_zeros(polynomial[::-1],trim='f'))
    candidates=[0.]+[float((np.angle(z)%(2*np.pi))/2) for z in roots if abs(abs(z)-1)<1e-5]
    results=[float(evaluate(t)) for t in candidates]
    index=int(np.argmin(results))
    def interpolated(t):return float(np.real(np.sum(coeff*np.exp(2j*frequencies*t))))
    checks=[np.pi*(j+.37)/13 for j in range(13)]
    replay=max(abs(float(evaluate(t))-interpolated(t)) for t in checks)
    derivative=lambda t:float(np.real(np.sum(2j*frequencies*coeff*np.exp(2j*frequencies*t))))
    return dict(angle=candidates[index],value=results[index],initial=values[0],
                improvement=values[0]-results[index],interpolation_max_error=replay,
                candidate_count=len(candidates),stationary_derivative=abs(derivative(candidates[index])),
                candidate_angles=candidates,candidate_values=results)
