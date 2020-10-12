import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from ..utils.gaussian import gaussian_diagonal_ll

__all__ = ['sa_estimator', 'ds_estimator', 'elbo_estimator']


def sa_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the negative VAE ELBO, ready for back-propagation,
    using the reparameterisation trick and analytical results were possible.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """
    estimator = 0

    # Latent distributions.
    qf_mu, qf_cov, pf_mu, pf_cov = model.get_latent_dists(x, y, mask)

    # Required distributions.
    qf = MultivariateNormal(qf_mu, qf_cov)
    pf = MultivariateNormal(pf_mu, pf_cov)

    # Monte-Carlo estimate of ELBO gradient.
    for _ in range(num_samples):
        f = qf.rsample()

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = alpha * py_f_term.sum()
        estimator += py_f_term

        # Inner summation over samples from q(f).
    estimator /= num_samples

    # KL(q(f)|p(f))
    kl_term = kl_divergence(qf, pf)
    kl_term = kl_term.sum()
    estimator += - kl_term

    # Outer summation over batch.
    estimator /= x.shape[0]

    return -estimator


def ds_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the negative VAE ELBO, ready for back-propagation,
    using the reparameterisation trick and Monte Carlo estimates.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """
    estimator = 0

    # Latent distributions.
    qf_mu, qf_cov, pf_mu, pf_cov = model.get_latent_dists(x, y, mask)

    qf_var = torch.stack([cov.diag() for cov in qf_cov])
    pf_var = torch.stack([cov.diag() for cov in pf_cov])

    # Monte-Carlo estimate of ELBO gradient.
    for _ in range(num_samples):
        f = qf_mu + qf_var ** 0.5 * torch.randn_like(qf_mu)

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = alpha * py_f_term.sum()
        estimator += py_f_term

        # log q(f) term.
        qf_term = gaussian_diagonal_ll(f, qf_mu, qf_var)
        estimator += - qf_term.sum()

        # log p(f) term.
        # pf_term = pf.log_prob(f).sum()
        pf_term = gaussian_diagonal_ll(f, pf_mu, pf_var)
        estimator += pf_term.sum()

    # Inner summation over samples from q(f).
    estimator /= num_samples

    # Outer summation over batch.
    estimator /= x.shape[0]

    return -estimator


def elbo_estimator(model, x, y, mask=None, num_samples=1):
    """Estimates the VAE ELBO using the reparameterisation trick and
    analytical results were possible.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    """
    elbo = 0

    # Latent distributions.
    qf_mu, qf_cov, pf_mu, pf_cov = model.get_latent_dists(
        x, y, mask)

    # Required distributions.
    qf = MultivariateNormal(qf_mu, qf_cov)
    pf = MultivariateNormal(pf_mu, pf_cov)

    # Monte-Carlo estimate of ELBO.
    for _ in range(num_samples):
        f = qf.rsample()

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = py_f_term.sum()
        elbo += py_f_term

    # Inner summation over samples from q(f).
    elbo /= num_samples

    # KL(q(f)|p(f))
    kl_term = kl_divergence(qf, pf)
    kl_term = kl_term.sum()
    elbo += - kl_term

    return elbo
