import torch
import torch.nn as nn

from .kernels import Kernel

__all__ = ['KernelList', 'AdditiveKernel', 'MultiplicativeKernel']


class KernelList(nn.ModuleList):
    """A list of kernels.

    :param kernels (list): list of kernels.
    """
    def __init__(self, kernels):
        super().__init__(kernels)

    def forward(self, x1, x2, diag=False, embed=True):
        """Returns the covariance matrix defined by kernels in list."""
        covs = [kernel.forward(x1, x2, diag) for kernel in self]

        if diag and embed:
            covs = torch.stack([cov.diag_embed() for cov in covs])
        else:
            covs = torch.stack(covs)

        return covs


class AdditiveKernel(Kernel):
    """The addition of kernels.

    :param args (list): list of kernels.
    """
    def __init__(self, *args):
        super().__init__()

        self.kernels = KernelList(args)

    def forward(self, x1, x2, diag=False):
        """Returns the covariance matrix defined by the addition of kernels
        in list."""
        cov = self.kernels.forward(x1, x2, diag, embed=False).sum(0)

        return cov


class MultiplicativeKernel(Kernel):
    """The product of kernels.

    :param args (list): list of Kernels.
    """
    def __init__(self, *args):
        super().__init__()

        self.kernels = KernelList(args)

    def forward(self, x1, x2, diag=False):
        """Returns the covariance matrix defined by the product of kernels
        in list."""
        cov = self.kernels.forward(x1, x2, diag, embed=False).prod(0)

        return cov
