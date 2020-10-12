import torch
import torch.nn as nn

from torch.nn import functional as F

__all__ = ['LinearNN', 'LinearGaussian', 'AffineGaussian']

JITTER = 1e-5


class LinearNN(nn.Module):
    """A fully connected neural network.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 nonlinearity=F.relu):
        super().__init__()

        self.nonlinearity = nonlinearity

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden_dims[i]))
            elif i == len(hidden_dims):
                self.layers.append(nn.Linear(hidden_dims[i-1], out_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))

    def forward(self, x):
        """Returns output of the network."""
        for layer in self.layers[:-1]:
            x = self.nonlinearity(layer(x))

        x = self.layers[-1](x)
        return x


class LinearGaussian(nn.Module):
    """A fully connected neural network for parameterising a diagonal
    Gaussian distribution.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
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
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 initial_sigma=None, initial_mu=None, sigma=None,
                 train_sigma=True, min_sigma=0., nonlinearity=F.relu):
        super().__init__()

        self.out_dim = out_dim
        self.sigma = sigma
        self.min_sigma = min_sigma

        if self.sigma is not None:
            self.network = LinearNN(in_dim, out_dim, hidden_dims, nonlinearity)
            if train_sigma:
                self.raw_sigma = nn.Parameter(torch.tensor(self.sigma).log(),
                                              requires_grad=True)
            else:
                self.raw_sigma = nn.Parameter(torch.tensor(self.sigma).log(),
                                              requires_grad=False)
        else:
            self.network = LinearNN(in_dim, 2*out_dim, hidden_dims,
                                    nonlinearity)

            if initial_sigma is not None:
                initial_sigma = (torch.tensor(initial_sigma)
                                 + JITTER * torch.randn(out_dim))
                self.network.layers[-1].bias.data[out_dim:] = torch.log(
                    torch.exp(initial_sigma) - 1)

            if initial_mu is not None:
                initial_mu = (torch.tensor(initial_mu)
                              + JITTER * torch.randn(out_dim))
                self.network.layers[-1].bias.data[:out_dim] = torch.log(
                    torch.exp(initial_mu) - 1)

    def forward(self, x, *args, **kwargs):
        """Returns parameters of a diagonal Gaussian distribution."""
        x = self.network(x)
        mu = x[..., :self.out_dim]
        if self.sigma is not None:
            sigma = (self.min_sigma +
                     (1 - self.min_sigma) * self.raw_sigma.exp() *
                     torch.ones_like(mu))
        else:
            sigma = (self.min_sigma +
                     (1 - self.min_sigma) * F.softplus(x[..., self.out_dim:]))

        return mu, sigma


class AffineGaussian(nn.Module):
    """Affine transformation of the input for parameterising a diagonal
    Gaussian.

    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param sigma (float, optional): initial homoscedastic output sigma.
    :param initial_weight (tensor, optional): initial weight.
    :param initial_bias (teensor, optional): initial bias.
    """

    def __init__(self, in_dim, out_dim, sigma=1., initial_weight=None,
                 initial_bias=None):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        if initial_weight is None:
            initial_weight = torch.ones(out_dim, in_dim) / in_dim
        else:
            initial_weight = torch.tensor(initial_weight)

        if initial_bias is None:
            initial_bias = torch.zeros(out_dim)
        else:
            initial_bias = torch.tensor(initial_bias)

        # Initial weight and bias of the affine transformation.
        self.weight = nn.Parameter(initial_weight + JITTER * torch.randn(
            out_dim, in_dim), requires_grad=True)
        self.bias = nn.Parameter(initial_bias + JITTER * torch.randn(out_dim),
                                 requires_grad=True)

        self.raw_sigma = nn.Parameter(torch.tensor(sigma).log(),
                                      requires_grad=True)

    def forward(self, x):
        """Returns parameters of a diagonal Gaussian distribution."""
        mu = self.weight.matmul(x.unsqueeze(2)).squeeze(2) + self.bias
        sigma = torch.ones_like(mu) * self.raw_sigma.exp()

        return mu, sigma
