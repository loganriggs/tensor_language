"""Symmetric native bilinear numerator on the shared head8.2 writer channels."""
def cross(p,down,u,z):
    lu=u@p['left_writer'].T;ru=u@p['right_writer'].T
    lz=z@p['left_writer'].T;rz=z@p['right_writer'].T
    return (lu*rz+lz*ru)@down.T

def quadratic(p,down,u):
    return ((u@p['left_writer'].T)*(u@p['right_writer'].T))@down.T
