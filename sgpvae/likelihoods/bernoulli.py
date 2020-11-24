import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Bernoulli
from .networks import LinearNN
from .base import Likelihood

__all__ = ['Bernoulli', 'NNBernoulli']


class Bernoulli(Likelihood):

    def __init__(self):
        super().__init__()

        self.loglikelihood = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, z):
        px = Bernoulli(torch.sigmoid(z))

        return px

    def log_prob(self, z, x):
        return self.loglikelihood(z, x)


class NNBernoulli(nn.Module):
    """A fully connected neural network for parameterising a Bernoulli
    distribution.
    :param in_dim (int): dimension of the input variable.
    :param out_dim (int): dimension of the output variable.
    :param hidden_dims (list, optional): dimensions of hidden layers.
    :param nonlinearity (function, optional): non-linearity to apply in
    between layers.
    """
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64),
                 nonlinearity=F.relu):
        super().__init__()

        self.network = LinearNN(in_dim, out_dim, hidden_dims, nonlinearity)
        self.likelihood = Bernoulli()

    def forward(self, z, x):
        mu = self.network(z)

        return self.likelihood(mu, x)

    def predict(self, z):
        mu = self.network(z)

        return self.likelihood.log_prob(mu)
