import torch

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from ..utils.gaussian import gaussian_diagonal_ll

__all__ = ['sa_estimator', 'ds_estimator', 'elbo_estimator']


def sa_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the gradient of the negative SGP-VAE ELBO, ready for
    back-propagation, using the reparameterisation trick and analytical
    results where possible.

    :param model: SGP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """
    estimator = 0

    # Latent distributions.
    qf_mu, qf_cov, qu_mu, qu_cov, pu_mu, pu_cov, lf_y_mu, lf_y_cov = \
        model.latent_dists(x, y, mask)

    # Required distributions.
    pu = MultivariateNormal(pu_mu, pu_cov)
    qu = MultivariateNormal(qu_mu, qu_cov)

    qf_var = torch.stack([cov.diag() for cov in qf_cov])

    # Monte-Carlo estimate of ELBO gradient.
    # See Spatio-Temporal VAEs: ELBO Gradient Estimators.
    for _ in range(num_samples):
        f = qf_mu + qf_var ** 0.5 * torch.randn_like(qf_mu)

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = alpha * py_f_term.sum()
        estimator += py_f_term

    # Inner summation over samples from q(f).
    estimator /= num_samples

    # log KL(q(u)||p(u)) term.
    estimator -= kl_divergence(qu, pu).sum()

    # Outer summation over batch.
    estimator /= x.shape[0]

    return -estimator


def ds_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the gradient of the negative SGP-VAE ELBO, ready for
    back-propagation, using the reparameterisation trick and Monte Carlo
    estimates.

    :param model: SGP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    :param alpha: (float, optional) scales the likelihood, p(y|f).
    """
    estimator = 0

    # Latent distributions.
    qf_mu, qf_cov, qu_mu, qu_cov, pu_mu, pu_cov, lf_y_mu, lf_y_cov = \
        model.latent_dists(x, y, mask)

    # Required distributions.
    pu = MultivariateNormal(pu_mu, pu_cov)
    qu = MultivariateNormal(qu_mu, qu_cov)

    qf_var = torch.stack([cov.diag() for cov in qf_cov])

    # Monte-Carlo estimate of ELBO gradient.
    # See Spatio-Temporal VAEs: ELBO Gradient Estimators.
    for _ in range(num_samples):
        f = qf_mu + qf_var ** 0.5 * torch.randn_like(qf_mu)
        u = qu.rsample()

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = alpha * py_f_term.sum()

        # log q(u) term.
        qu_term = qu.log_prob(u).sum()

        # log p(u) term.
        pu_term = pu.log_prob(u).sum()

        estimator += py_f_term - qu_term + pu_term

    # Inner summation over samples from q(f).
    estimator /= num_samples

    # Outer summation over batch.
    estimator /= x.shape[0]

    return -estimator


def elbo_estimator(model, x, y, mask=None, num_samples=1):
    """Estimates the SGP-VAE ELBO using analytical results were possible.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    """
    elbo = 0

    # Latent distributions.
    qf_mu, qf_cov, qu_mu, qu_cov, pu_mu, pu_cov, lf_y_mu, lf_y_cov = \
        model.latent_dists(x, y, mask)

    # Required distributions for KL divergence.
    pu = MultivariateNormal(pu_mu, pu_cov)
    qu = MultivariateNormal(qu_mu, qu_cov)

    qf_var = torch.stack([cov.diag() for cov in qf_cov])

    # Monte-Carlo estimate of ELBO gradient.
    # See Spatio-Temporal VAEs: ELBO Gradient Estimators.
    for _ in range(num_samples):
        f = qf_mu + qf_var ** 0.5 * torch.randn_like(qf_mu)

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = py_f_term.sum()
        elbo += py_f_term

    # Inner summation over samples from q(f).
    elbo /= num_samples

    # log KL(q(u)||p(u)) term.
    elbo -= kl_divergence(qu, pu).sum()

    return elbo
