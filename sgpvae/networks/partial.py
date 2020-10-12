import torch
import torch.nn as nn

from torch.nn import functional as F
from .base import LinearNN, LinearGaussian

__all__ = ['IndexNet', 'FactorNet', 'PointNet']

JITTER = 1e-5


class FactorNet(nn.Module):
    """FactorNet from Spatio-Temporal Variational Autoencoders.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param h_dims (list, optional): dimensions of hidden layers.
    :param initial_sigma (float, optional): initial output sigma.
    :param initial_mu (float, optional): initial output mean.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param train_sigma (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, h_dims=(64, 64), initial_sigma=None,
                 initial_mu=None, sigma=None, train_sigma=True,
                 min_sigma=0., nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim

        # Rescale sigmas for multiple outputs.
        initial_sigma = initial_sigma * in_dim ** 0.5
        min_sigma = min_sigma * in_dim ** 0.5

        if sigma is not None:
            sigma = sigma * in_dim ** 0.5

        # A network for each dimension.
        self.networks = nn.ModuleList()
        if sigma is None:
            for _ in range(in_dim):
                self.networks.append(LinearGaussian(
                    1, out_dim, h_dims, initial_sigma, initial_mu,
                    min_sigma=min_sigma, nonlinearity=nonlinearity))
        else:
            for _ in range(in_dim):
                self.networks.append(LinearGaussian(
                    1, out_dim, h_dims, initial_sigma, initial_mu, sigma,
                    train_sigma, min_sigma, nonlinearity))

    def forward(self, x, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        np_1 = torch.zeros(x.shape[0], x.shape[1], self.out_dim)
        np_2 = torch.zeros_like(np_1)

        # Pass through individual networks.
        for dim, x_dim in enumerate(x.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                x_in = x_dim[idx].unsqueeze(1)

                # Don't pass through if no inputs.
                if len(x_in) != 0:
                    mu, sigma = self.networks[dim](x_in)
                    np_1[idx, dim, :] = mu / sigma ** 2
                    np_2[idx, dim, :] = - 1. / (2. * sigma ** 2)
            else:
                x_in = x_dim.unsqueeze(1)
                mu, sigma = self.networks[dim](x_in)
                np_1[:, dim, :] = mu / sigma ** 2
                np_2[:, dim, :] = -1. / (2. * sigma ** 2)

        # Sum natural parameters.
        np_1 = torch.sum(np_1, 1)
        np_2 = torch.sum(np_2, 1)
        sigma = (- 1. / (2. * np_2)) ** 0.5
        mu = np_1 * sigma ** 2.

        return mu, sigma


class IndexNet(nn.Module):
    """IndexNet from Spatio-Temporal Variational Autoencoders.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param inter_dim (int): dimension of intermediate representation.
    :param h_dims (list, optional): dimension of the encoding function.
    :param rho_dims (list, optional): dimension of the shared function.
    :param initial_sigma (float, optional): initial output sigma.
    :param initial_mu (float, optional): initial output mean.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param train_sigma (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, inter_dim, h_dims=(64, 64),
                 rho_dims=(64, 64), initial_sigma=None, initial_mu=None,
                 sigma=None, train_sigma=True, min_sigma=0.,
                 nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.inter_dim = inter_dim

        # A network for each dimension.
        self.networks = nn.ModuleList()
        for _ in range(in_dim):
            self.networks.append(LinearNN(1, inter_dim, h_dims, nonlinearity))

        # Takes the aggregation of the outputs from self.networks.
        self.rho = LinearGaussian(
            inter_dim, out_dim, rho_dims, initial_sigma,
            initial_mu, sigma, train_sigma, min_sigma, nonlinearity)

    def forward(self, x, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        out = torch.zeros(x.shape[0], x.shape[1], self.middle_dim)

        # Pass through individual networks.
        for dim, x_dim in enumerate(x.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                x_in = x_dim[idx].unsqueeze(1)

                # Don't pass through if no inputs.
                if len(x_in) != 0:
                    x_out = self.networks[dim](x_in)
                    out[idx, dim, :] = x_out

            else:
                x_in = x_dim.unsqueeze(1)
                x_out = self.networks[dim](x_in)
                out[:, dim, :] = x_out

        # Aggregation layer.
        out = torch.sum(out, 1)

        # Pass through shared network.
        mu, sigma = self.rho(out)

        return mu, sigma


class PointNet(nn.Module):
    """PointNet from Spatio-Temporal Variational Autoencoders.

    :param out_dim (int): dimension of the output variable.
    :param inter_dim (int): dimension of intermediate representation.
    :param h_dims (list, optional): dimension of the encoding function.
    :param rho_dims (list, optional): dimension of the shared function.
    :param initial_sigma (float, optional): initial output sigma.
    :param initial_mu (float, optional): initial output mean.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param train_sigma (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, out_dim, inter_dim, h_dims=(64, 64), rho_dims=(64, 64),
                 initial_sigma=None, initial_mu=None, sigma=None,
                 train_sigma=True, min_sigma=0., nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.inter_dim = inter_dim

        # Takes the index of the observation dimension and it's value.
        self.h = LinearNN(
            2, inter_dim, h_dims, nonlinearity)

        # Takes the aggregation of the outputs from self.h.
        self.rho = LinearGaussian(
            inter_dim, out_dim, rho_dims, initial_sigma,
            initial_mu, sigma, train_sigma, min_sigma, nonlinearity)

    def forward(self, x, mask=None):
        """Returns parameters of a diagonal Gaussian distribution."""
        out = torch.zeros(x.shape[0], x.shape[1], self.middle_dim)

        # Pass through first network.
        for dim, x_dim in enumerate(x.transpose(0, 1)):
            if mask is not None:
                idx = torch.where(mask[:, dim])[0]
                x_in = x_dim[idx].unsqueeze(1)
                x_in = torch.cat([x_in, torch.ones_like(x_in)*dim], 1)
                out[idx, dim, :] = self.h(x_in)
            else:
                x_in = x_dim.unsqueeze(1)
                torch.cat([x_in, torch.ones_like(x_in)*dim], 1)
                out[:, dim, :] = self.h(x_in)

        # Aggregation layer.
        out = torch.sum(out, 1)

        # Pass through second network.
        mu, sigma = self.rho(out)

        return mu, sigma
