import numpy as np
import torch
import torch.nn as nn

__all__ = ['Kernel', 'RBFKernel', 'PeriodicKernel']


class Kernel(nn.Module):
    """A base class for GP kernels.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x1, x2, diag=False):
        raise NotImplementedError


class RBFKernel(Kernel):
    """A radial basis function (squared exponential) kernel.

    :param lengthscale (float, optional): initial lengthscale(s).
    :param scale (float, optional): initial scale.
    """
    def __init__(self, lengthscale=1., scale=1., **kwargs):
        super().__init__(**kwargs)

        self.lengthscale = nn.Parameter(torch.tensor(lengthscale),
                                        requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)

    def forward(self, x1, x2, diag=False):
        """Returns the covariance matrix defined by the RBF kernel."""
        x1 = x1.unsqueeze(1) if len(x1.shape) == 1 else x1
        x2 = x2.unsqueeze(1) if len(x2.shape) == 1 else x2

        assert x1.shape[-1] == x2.shape[-1], 'Inputs are different dimensions.'

        if not diag:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(0)
        else:
            assert x1.shape == x2.shape, 'Inputs must be the same shape.'

        # [M1, M2, D] or [M, D] if diag.
        sd = (x1 - x2) ** 2
        sd.clamp_min_(0)

        # Apply lengthscale and sum over dimensions.
        sd_ = (sd / self.lengthscale ** 2).sum(-1)
        cov = self.scale ** 2 * (-sd_).exp()

        return cov


class PeriodicKernel(Kernel):
    """A periodic kernel.

    :param lengthscale (float): initial lengthscale(s).
    :param period (float): initial period(s).
    :param scale (float): initial scale.
    """
    def __init__(self, lengthscale=1., period=1., scale=1., **kwargs):
        super().__init__(**kwargs)

        self.lengthscale = nn.Parameter(torch.tensor(lengthscale),
                                        requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.raw_period = nn.Parameter(torch.tensor(period).log(),
                                       requires_grad=True)

    def forward(self, x1, x2, diag=False):
        """ Returns the covariance matrix defined by the periodic kernel."""
        x1 = x1.unsqueeze(1) if len(x1.shape) == 1 else x1
        x2 = x2.unsqueeze(1) if len(x2.shape) == 1 else x2

        assert x1.shape[-1] == x2.shape[-1], 'Inputs are different dimensions.'

        if not diag:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(0)
        else:
            assert x1.shape == x2.shape, 'Inputs must be the same shape.'

        # [M1, M2, D] or [M, D] if diag.
        ad = (x1 - x2).abs()

        # Apply period.
        ad_ = 2 * (np.pi * ad / self.raw_period.exp()).sin() ** 2

        # Apply lengthscale and sum over dimensions.
        ad_ = (ad_ / self.lengthscale ** 2).sum(-1)
        cov = self.scale ** 2 * (-ad_).exp()

        return cov
