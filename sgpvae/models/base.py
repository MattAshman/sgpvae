import torch
import torch.nn as nn

from torch.distributions import Normal, kl_divergence

__all__ = ['VAE']


class VAE(nn.Module):
    """VAE with standard normal prior.

    :param likelihood: likelihood function, p(x|f).
    :param variational_dist: variational distribution, q(f|x).
    :param latent_dim (int): latent space dimensionality.
    """
    def __init__(self, likelihood, variational_dist, latent_dim):
        super().__init__()

        self.likelihood = likelihood
        self.variational_dist = variational_dist
        self.latent_dim = latent_dim

    def pf(self):
        """Return latent prior."""
        pf_mu = torch.zeros(self.latent_dim)
        pf_sigma = torch.ones(self.latent_dim)
        pf = Normal(pf_mu, pf_sigma)

        return pf

    def qf(self, y, mask=None):
        """Return latent approximate posterior."""
        qf = self.variational_dist(y, mask)

        return qf

    def elbo(self, y, mask=None, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pf = self.pf()
        qf = self.qf(y, mask)

        # KL(q(f) || p(f) term.
        kl = kl_divergence(qf, pf).sum()

        # log p(y|f) term.
        f_samples = qf.rsample((num_samples,))
        log_py_f = 0
        for f in f_samples:
            log_py_f += (self.likelihood.log_prob(f, y) * mask).sum()

        log_py_f /= num_samples
        elbo = log_py_f - kl

        return elbo / y.shape[0]

    def predict_y(self, y=None, num_samples=1):
        """Sample predictive posterior."""
        if y is None:
            qf = self.pf()
        else:
            qf = self.qf(y)

        f_samples = qf.sample((num_samples,))

        y_mus, y_sigmas, y_samples = [], [], []
        for f in f_samples:
            # Output conditional posterior distribution.
            py_f = self.likelihood(f)
            y_mus.append(py_f.mean)
            y_samples.append(py_f.sample())

        y_mu = torch.stack(y_mus).mean(0).detach()
        y_sigma = torch.stack(y_samples).std(0).detach()

        return y_mu, y_sigma, y_samples
