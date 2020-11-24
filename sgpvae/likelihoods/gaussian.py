import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal
from .networks import LinearNN
from .base import Likelihood

__all__ = ['HomoGaussian', 'AffineHomoGaussian', 'NNHomoGaussian',
           'NNHeteroGaussian']


class HomoGaussian(Likelihood):

    def __init__(self, dim, sigma=None, sigma_grad=True, min_sigma=1e-3):
        super().__init__()

        self.min_sigma = min_sigma

        if sigma is None:
            self.log_sigma = nn.Parameter(torch.zeros(dim),
                                          requires_grad=sigma_grad)
        else:
            self.log_sigma = nn.Parameter(torch.ones(dim) * np.log(sigma),
                                          requires_grad=sigma_grad)

    def forward(self, z):
        sigma = self.log_sigma.exp().clamp(min=self.min_sigma)
        px = Normal(z, sigma)

        return px


class AffineHomoGaussian(Likelihood):
    """Affine transformation of the input for parameterising d diagonal
    Gaussian distribution with homoscedastic noise.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param sigma (float, optional): initial homoscedastic output sigma.
    :param sigma_grad (float, optional): whether to train the homoscedastic
    ouput sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_weight (tensor, optional): initial weight.
    :param init_bias (teensor, optional): initial bias.
    """
    def __init__(self, in_dim, out_dim, sigma=None, sigma_grad=True,
                 min_sigma=1e-3, init_weight=None, init_bias=None):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        if init_weight is None:
            init_weight = torch.ones(out_dim, in_dim) / in_dim
        else:
            init_weight = torch.tensor(init_weight)

        if init_bias is None:
            init_bias = torch.zeros(out_dim)
        else:
            init_bias = torch.tensor(init_bias)

        # Initial weight and bias of the affine transformation.
        self.weight = nn.Parameter(init_weight, requires_grad=True)
        self.bias = nn.Parameter(init_bias, requires_grad=True)

        self.likelihood = HomoGaussian(out_dim, sigma, sigma_grad, min_sigma)

    def forward(self, z):
        mu = self.weight.matmul(z.unsqueeze(2)).squeeze(2) + self.bias

        return self.likelihood(mu)


class NNHomoGaussian(Likelihood):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution with homoscedastic noise.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), sigma=None,
                 sigma_grad=True, min_sigma=1e-3, nonlinearity=F.relu):
        super().__init__()

        self.network = LinearNN(in_dim, out_dim, hidden_dims, nonlinearity)
        self.likelihood = HomoGaussian(out_dim, sigma, sigma_grad, min_sigma)

    def forward(self, z):
        mu = self.network(z)

        return self.likelihood(mu)


class NNHeteroGaussian(Likelihood):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution with heteroscedastic noise.
    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param init_sigma (float, optional): sets the initial output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 min_sigma=1e-3, init_sigma=None, nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.min_sigma = min_sigma
        self.network = LinearNN(in_dim, 2 * out_dim, hidden_dims, nonlinearity)

        if init_sigma is not None:
            self.network.layers[-1].bias.data[out_dim:] = torch.log(
                torch.exp(torch.tensor(init_sigma)) - 1)

    def forward(self, z, *args, **kwargs):
        output = self.network(z)
        mu = output[..., :self.out_dim]
        raw_sigma = output[..., self.out_dim:]
        sigma = F.softplus(raw_sigma) + self.min_sigma
        px = Normal(mu, sigma)

        return px
