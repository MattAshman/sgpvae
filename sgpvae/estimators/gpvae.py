import torch

from torch.distributions import MultivariateNormal
from ..utils.gaussian import gaussian_diagonal_ll

__all__ = ['sa_estimator', 'ds_estimator', 'elbo_estimator']


def sa_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
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
    estimator = 0

    # Latent distributions.
    qf_mu, qf_cov, pf_mu, pf_cov, lf_y_mu, lf_y_cov = model.latent_dists(
        x, y, mask)

    # Required distributions.
    sum_cov = pf_cov + lf_y_cov
    zq = MultivariateNormal(lf_y_mu, sum_cov)

    qf_var = torch.stack([cov.diag() for cov in qf_cov])
    lf_y_var = torch.stack([cov.diag() for cov in lf_y_cov])

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

    # log l(f|y) term.
    lf_y_term = gaussian_diagonal_ll(qf_mu, lf_y_mu, lf_y_var).sum()
    lf_y_term += - 0.5 * (qf_var / lf_y_var).sum()
    estimator += - lf_y_term

    # log Zq term.
    zq_term = zq.log_prob(torch.zeros_like(lf_y_mu)).sum()
    estimator += zq_term

    # Outer summation over batch
    estimator /= x.shape[0]

    return -estimator


def ds_estimator(model, x, y, mask=None, num_samples=1, alpha=1.):
    """Estimates the negative GP-VAE ELBO, ready for back-propagation,
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
    qf_mu, qf_cov, pf_mu, pf_cov, lf_y_mu, lf_y_cov = model.latent_dists(
        x, y, mask)

    # Required distributions.
    qf = MultivariateNormal(qf_mu, qf_cov)
    pf = MultivariateNormal(pf_mu, pf_cov)

    lf_y_var = torch.stack([cov.diag() for cov in lf_y_cov])

    # Monte-Carlo estimate of ELBO gradient.
    # See Spatio-Temporal VAEs: ELBO Gradient Estimators.
    for _ in range(num_samples):
        f = qf.rsample()

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = alpha * py_f_term.sum()
        estimator += py_f_term

        # log l(f|y) term.
        lf_y_term = gaussian_diagonal_ll(f, lf_y_mu.detach(),
                                         lf_y_var.detach())
        lf_y_term = lf_y_term.sum()
        estimator += - lf_y_term

        # log p(f) term.
        pf_term = pf.log_prob(f.detach()).sum()
        estimator += pf_term

    # Inner summation over samples from q(f).
    estimator /= num_samples

    # Outer summation over batch.
    estimator /= x.shape[0]

    return -estimator


def elbo_estimator(model, x, y, mask=None, num_samples=1):
    """Estimates the GP-VAE ELBO using analytical results were possible.

    :param model: GP-VAE model.
    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param mask: (tensor, optional) mask to apply to output data.
    :param num_samples: (int, optional) number of Monte Carlo samples.
    """
    elbo = 0

    # Latent distributions.
    qf_mu, qf_cov, pf_mu, pf_cov, lf_y_mu, lf_y_cov = \
        model.get_latent_dists(x, y, mask)

    sum_cov = pf_cov + lf_y_cov

    # Required distributions.
    qf = MultivariateNormal(qf_mu, qf_cov)
    zq = MultivariateNormal(lf_y_mu, sum_cov)

    qf_var = torch.stack([cov.diag() for cov in qf_cov])
    lf_y_var = torch.stack([cov.diag() for cov in lf_y_cov])

    # Monte-Carlo estimate of ELBO.
    # See Spatio-Temporal VAEs: ELBO
    for i in range(num_samples):
        f = qf.rsample()

        # log p(y|f) term.
        py_f_mu, py_f_sigma = model.decoder(f.transpose(0, 1))
        py_f_term = gaussian_diagonal_ll(y, py_f_mu, py_f_sigma.pow(2), mask)
        py_f_term = py_f_term.sum()
        elbo += py_f_term

    # Inner summation over samples from q(f).
    elbo /= num_samples

    # log l(f|y) term.
    lf_y_term = gaussian_diagonal_ll(qf_mu, lf_y_mu, lf_y_var).sum()
    lf_y_term += - 0.5 * (qf_var / lf_y_var).sum()
    elbo += - lf_y_term

    # log Zq term.
    zq_term = zq.log_prob(torch.zeros_like(lf_y_mu)).sum()
    elbo += zq_term

    return elbo
