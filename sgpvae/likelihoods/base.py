import torch.nn as nn

__all__ = ['Likelihood']


class Likelihood(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError

    def log_prob(self, z, x):
        px = self(z)

        return px.log_prob(x)
