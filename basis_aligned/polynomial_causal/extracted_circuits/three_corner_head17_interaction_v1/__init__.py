from .execute import execute

def residual_write(*args, output_matrix, **kwargs):
    return execute(*args, **kwargs) @ output_matrix.T
