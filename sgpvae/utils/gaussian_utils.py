import numpy as np

__all__ = ['gaussian_diagonal_ll', 'gaussian_diagonal_kl']


def gaussian_diagonal_ll(x, m, v, mask=None):
    """ The log likelihood of a diagonal Gaussian.

    :param x: (tensor) [M, D] positions at which to evaluate.
    :param m: (tensor) [M, D] mean of the Gaussian.
    :param v: (tensor) [M, D] variances of the Gaussian.
    :param mask: (tensor) [M, D] mask to apply to the likelihoods.
    """
    assert (x.shape == m.shape), 'x and mean should be the same shape.'
    assert (m.shape == v.shape), 'mean and variance should be the same shape.'

    sd = (x - m) ** 2
    ll = (- 0.5 * (2 * np.pi * v).log() - 0.5 * v.pow(-1) * sd)

    if mask is not None:
        ll *= mask

    # Sum over dimensions.
    ll = ll.sum(1)

    return ll


def gaussian_diagonal_kl(m1, v1, m2, v2):
    """Computes the KL divergence between two diagonal Gaussians.

    :param m1: (tensor) [M, D], the mean of the first Gaussian.
    :param v1: (tensor) [M, D], the variance of the first Gaussian.
    :param m2: (tensor) [M, D], the mean of the second Gaussian.
    :param v2: (tensor) [M, D], the variance of the second Gaussian.
    """
    kl = 0.5 * ((v2 / v1).log() + (v1 + (m1 - m2) ** 2) / v2 - 1)

    # Sum over dimensions.
    kl = kl.sum(1)

    return kl
