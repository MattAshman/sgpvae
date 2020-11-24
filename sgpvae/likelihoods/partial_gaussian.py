import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import Normal
from .base import Likelihood
from .gaussian import NNHeteroGaussian
from .networks import LinearNN

__all__ = ['IndexNet', 'FactorNet', 'PointNet']


class FactorNet(Likelihood):
    """FactorNet from Sparse Gaussian Process Variational Autoencoders.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param h_dims (list, optional): dimensions of hidden layers.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param initi_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, h_dims=(64, 64), min_sigma=1e-3,
                 init_sigma=None, nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim

        # Rescale sigmas for multiple outputs.
        if init_sigma is not None:
            init_sigma = init_sigma * in_dim ** 0.5

        min_sigma = min_sigma * in_dim ** 0.5

        # A network for each dimension.
        self.networks = nn.ModuleList()
        for _ in range(in_dim):
            self.networks.append(NNHeteroGaussian(
                1, out_dim, h_dims, min_sigma, init_sigma,
                nonlinearity=nonlinearity))

    def forward(self, z, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        np_1 = torch.zeros(z.shape[0], z.shape[1], self.out_dim)
        np_2 = torch.zeros_like(np_1)

        # Pass through individual networks.
        for dim, z_dim in enumerate(z.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                z_in = z_dim[idx].unsqueeze(1)

                # Don't pass through if no inputs.
                if len(z_in) != 0:
                    pz = self.networks[dim](z_in)
                    mu, sigma = pz.mean, pz.stddev
                    np_1[idx, dim, :] = mu / sigma ** 2
                    np_2[idx, dim, :] = - 1. / (2. * sigma ** 2)
            else:
                z_in = z_dim.unsqueeze(1)
                pz = self.networks[dim](z_in)
                mu, sigma = pz.mean, pz.stddev
                np_1[:, dim, :] = mu / sigma ** 2
                np_2[:, dim, :] = -1. / (2. * sigma ** 2)

        # Sum natural parameters.
        np_1 = torch.sum(np_1, 1)
        np_2 = torch.sum(np_2, 1)
        sigma = (- 1. / (2. * np_2)) ** 0.5
        mu = np_1 * sigma ** 2.

        pz = Normal(mu, sigma)

        return pz


class IndexNet(Likelihood):
    """IndexNet from Sparse Gaussian Process Variational Autoencoders.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param inter_dim (int): dimension of intermediate representation.
    :param h_dims (list, optional): dimension of the encoding function.
    :param rho_dims (list, optional): dimension of the shared function.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, inter_dim, h_dims=(64, 64),
                 rho_dims=(64, 64), min_sigma=1e-3, init_sigma=None,
                 nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.inter_dim = inter_dim

        # A network for each dimension.
        self.networks = nn.ModuleList()
        for _ in range(in_dim):
            self.networks.append(LinearNN(1, inter_dim, h_dims, nonlinearity))

        # Takes the aggregation of the outputs from self.networks.
        self.rho = NNHeteroGaussian(
            inter_dim, out_dim, rho_dims, min_sigma, init_sigma, nonlinearity)

    def forward(self, z, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        out = torch.zeros(z.shape[0], z.shape[1], self.inter_dim)

        # Pass through individual networks.
        for dim, z_dim in enumerate(z.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                z_in = z_dim[idx].unsqueeze(1)

                # Don't pass through if no inputs.
                if len(z_in) != 0:
                    z_out = self.networks[dim](z_in)
                    out[idx, dim, :] = z_out

            else:
                z_in = z_dim.unsqueeze(1)
                z_out = self.networks[dim](z_in)
                out[:, dim, :] = z_out

        # Aggregation layer.
        out = torch.sum(out, 1)

        # Pass through shared network.
        pz = self.rho(out)

        return pz


class PointNet(Likelihood):
    """PointNet from Sparse Gaussian Process Variational Autoencoders.

    :param out_dim (int): dimension of the output variable.
    :param inter_dim (int): dimension of intermediate representation.
    :param h_dims (list, optional): dimension of the encoding function.
    :param rho_dims (list, optional): dimension of the shared function.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, out_dim, inter_dim, h_dims=(64, 64), rho_dims=(64, 64),
                 min_sigma=1e-3, init_sigma=None, nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.inter_dim = inter_dim

        # Takes the index of the observation dimension and it's value.
        self.h = LinearNN(
            2, inter_dim, h_dims, nonlinearity)

        # Takes the aggregation of the outputs from self.h.
        self.rho = NNHeteroGaussian(
            inter_dim, out_dim, rho_dims, min_sigma, init_sigma, nonlinearity)

    def forward(self, z, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        out = torch.zeros(z.shape[0], z.shape[1], self.inter_dim)

        # Pass through first network.
        for dim, z_dim in enumerate(z.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                z_in = z_dim[idx].unsqueeze(1)
                z_in = torch.cat([z_in, torch.ones_like(z_in)*dim], 1)
                out[idx, dim, :] = self.h(z_in)
            else:
                z_in = z_dim.unsqueeze(1)
                torch.cat([z_in, torch.ones_like(z_in)*dim], 1)
                out[:, dim, :] = self.h(z_in)

        # Aggregation layer.
        out = torch.sum(out, 1)

        # Pass through second network.
        pz = self.rho(out)

        return pz
