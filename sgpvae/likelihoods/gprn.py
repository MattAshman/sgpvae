import torch.nn.functional as F

from .networks import LinearNN
from .base import Likelihood
from .gaussian import HomoGaussian

__all__ = ['GPRNHomoGaussian', 'GPRNNNHomoGaussian']


class GPRNHomoGaussian(Likelihood):
    """The likelihood function described in Gaussian Process Regresssion
    Networks with homoscedastic noise.

    :param f_dim (int): dimension of the latent space.
    :param out_dim (int): dimension of the output variable.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layeres.
    """
    def __init__(self, f_dim, out_dim, sigma=None, sigma_grad=True,
                 min_sigma=1e-3, nonlinearity=F.relu):
        super().__init__()

        self.f_dim = f_dim
        self.w_dim = out_dim
        self.likelihood = HomoGaussian(out_dim, sigma, sigma_grad, min_sigma)

    def forward(self, z):
        f, w = z[:, :self.f_dim], z[:, self.f_dim:]
        w = w.reshape(-1, self.w_dim, self.f_dim)
        mu = w.matmul(f.unsqueeze(-1)).squeeze(-1)

        return self.likelihood(mu)


class GPRNNNHomoGaussian(Likelihood):
    """The likelihood function described in Gaussian Process Regresssion
    Networks with a fully connected neural network stuck on top with
    homoscedastic noise.

    :param f_dim (int): dimension of the latent space.
    :param w_dim (int): dimension of the first hidden layer.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param sigma (float, optional): if not None, sets the initial
    homoscedastic output sigma.
    :param sigma_grad (bool, optional): whether to train the homoscedastic
    output sigma.
    :param min_sigma (float, optional): sets the minimum output sigma.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layeres.
    """

    def __init__(self, f_dim, w_dim, out_dim, hidden_dims=(64, 64),
                 sigma=None, sigma_grad=True, min_sigma=1e-3,
                 nonlinearity=F.relu):
        super().__init__()

        self.f_dim = f_dim
        self.w_dim = w_dim
        self.network = LinearNN(w_dim, out_dim, hidden_dims, nonlinearity)
        self.likelihood = HomoGaussian(out_dim, sigma, sigma_grad, min_sigma)

    def forward(self, z):
        f, w = z[:, :self.f_dim], z[:, self.f_dim:]
        w = w.reshape(-1, self.w_dim, self.f_dim)
        mu = self.network(w.matmul(f.unsqueeze(-1)).squeeze(-1))

        return self.likelihood(mu)
