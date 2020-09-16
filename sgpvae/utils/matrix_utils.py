import torch

__all__ = ['add_diagonal']


def add_diagonal(x, val=1.):
    """Adds a value to the diagonal of a matrix.

    :param x: (tensor) matrix to modify.
    :param val: (float) value to add to the diagonal.
    """
    assert x.shape[-2] == x.shape[-1], 'x must be square.'

    d = (torch.ones(x.shape[-2]) * val).diag_embed()

    return x + d
