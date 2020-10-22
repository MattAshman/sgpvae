import torch

from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from ..utils.gaussian import gaussian_diagonal_ll

__all__ = ['elbo_estimator', 'sa_estimator', 'ds_estimator']


def elbo_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the negative GP-VAE ELBO, ready for back-propagation,
    using the reparameterisation trick and analytical results where
    possible.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """
    elbo = 0

    # Latent distributions.
    qz_mu, qz_cov, pz_mu, pz_cov, lz_y_mu, lz_y_cov = model.latent_dists(
        x, y, mask)

    # Required distributions.
    pz = MultivariateNormal(pz_mu, pz_cov)
    qz = MultivariateNormal(qz_mu, qz_cov)

    qf_mu = qz_mu[:model.f_dim, :]
    qf_var = torch.stack([cov.diag() for cov in qz_cov[:model.f_dim, ...]])
    qw_mu = qz_mu[model.f_dim:, :]
    qw_var = torch.stack([cov.diag() for cov in qz_cov[model.f_dim:, ...]])

    # Monte-Carlo estimate of ELBO gradient.
    # See Spatio-Temporal VAEs: ELBO Gradient Estimators.
    for _ in range(num_samples):
        f = qf_mu + qf_var ** 0.5 * torch.randn_like(qf_mu)
        w = qw_mu + qw_var ** 0.5 * torch.randn_like(qw_mu)

        g = torch.zeros(model.w_dim, x.shape[0])
        for n in range(x.shape[0]):
            wn = w[:, n].reshape(model.w_dim, model.f_dim)
            # wn = torch.sigmoid(wn)
            g[:, n] = wn.matmul(f[:, n])

        # log p(y|z) term.
        py_z_mu, py_z_sigma = model.decoder(g.T)
        py_z_term = gaussian_diagonal_ll(y, py_z_mu, py_z_sigma.pow(2), mask)
        py_z_term = alpha * py_z_term.sum()
        elbo += py_z_term

    # Inner summation over samples from q(f).
    elbo /= num_samples

    # KL(q(z) || p(z)) term.
    elbo -= kl_divergence(qz, pz).sum()

    return elbo


def sa_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the negative GPRN-VAE ELBO using analytical results were
    possible.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """

    elbo = elbo_estimator(model, x, y, mask, num_samples, alpha)

    return -elbo / x.shape[0]


def ds_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the negative GPRN-VAE ELBO, ready for back-propagation,
    using the reparameterisation trick and Monte-Carlo estimates.
    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """
    estimator = 0

    # Latent distributions.
    qz_mu, qz_cov, pz_mu, pz_cov, lz_y_mu, lz_y_cov = model.latent_dists(
        x, y, mask)

    qz = MultivariateNormal(qz_mu, qz_cov)
    pz = MultivariateNormal(pz_mu, pz_cov)

    lz_y_var = torch.stack([cov.diag() for cov in lz_y_cov])

    # Monte-Carlo estimate of ELBO gradient.
    # See Spatio-Temporal VAEs: ELBO Gradient Estimators.
    for _ in range(num_samples):
        z = qz.rsample()
        f = z[:model.f_dim, :]
        w = z[model.f_dim:, :]

        g = torch.zeros(model.w_dim, x.shape[0])
        for n in range(x.shape[0]):
            wn = w[:, n].reshape(model.w_dim, model.f_dim)
            # wn = torch.sigmoid(wn)
            g[:, n] = wn.matmul(f[:, n])

        # log p(y|z) term.
        py_z_mu, py_z_sigma = model.decoder(g.T)
        py_z_term = gaussian_diagonal_ll(y, py_z_mu, py_z_sigma.pow(2), mask)
        py_z_term = alpha * py_z_term.sum()
        estimator += py_z_term

        # log l(z|y) term.
        lz_y_term = gaussian_diagonal_ll(z, lz_y_mu.detach(),
                                         lz_y_var.detach())
        lz_y_term = lz_y_term.sum()
        estimator += - lz_y_term

        # log p(z) term.
        pz_term = pz.log_prob(z.detach()).sum()
        estimator += pz_term

    # Inner summation over samples from q(f).
    estimator /= num_samples

    # Outer summation over batch.
    estimator /= x.shape[0]

    return -estimator
