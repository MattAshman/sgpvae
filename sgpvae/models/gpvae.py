import copy
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal, Normal, kl_divergence
from sgpvae.kernels import KernelList
from sgpvae.utils.matrix import add_diagonal

from .base import VAE

__all__ = ['GPVAE', 'SGPVAE']

JITTER = 1e-5


class GPVAE(VAE):
    """The GP-VAE model.

    :param likelihood: likelihood function, p(x|f).
    :param variational_dist: variational distribution, l(f|x).
    :param latent_dim (int): the dimension of latent space.
    :param kernel: GP kernel.
    :param add_jitter (bool, optional): whether to add jitter to the GP prior
    covariance matrix.
    """
    def __init__(self, likelihood, variational_dist, latent_dim, kernel,
                 add_jitter=False):
        super().__init__(likelihood, variational_dist, latent_dim)

        self.add_jitter = add_jitter

        if not isinstance(kernel, list):
            kernels = [copy.deepcopy(kernel) for _ in range(latent_dim)]
            self.kernels = KernelList(kernels)

        else:
            assert len(kernel) == latent_dim, 'Number of kernels must be ' \
                                              'equal to the latent dimension.'
            self.kernels = KernelList(copy.deepcopy(kernel))

    def pf(self, x, diag=False):
        """Return latent prior."""
        pf_mu = torch.zeros(self.latent_dim, x.shape[0])
        pf_cov = self.kernels.forward(x, x, diag)

        if self.add_jitter:
            # Add jitter to improve condition number.
            pf_cov = add_diagonal(pf_cov, JITTER)

        pf_chol = pf_cov.cholesky()
        pf = MultivariateNormal(pf_mu, scale_tril=pf_chol)

        return pf

    def lf(self, y, mask=None):
        """Return the approximate likelihood."""
        lf = self.variational_dist(y, mask)

        return lf

    def qf(self, x=None, y=None, pf=None, lf=None, mask=None, diag=False,
           x_test=None):
        if pf is None:
            pf = self.pf(x, diag)

        # Mean, covariance and Cholesky factor.
        pf_cov, pf_chol = pf.covariance_matrix, pf.scale_tril

        if lf is None:
            lf = self.lf(y, mask)

        # Reshape.
        lf_mu = lf.mean.transpose(0, 1)
        lf_sigma = lf.stddev.transpose(0, 1)
        lf_cov = lf_sigma.pow(2).diag_embed()

        # A = I + Lf^{-1} \Sigma_{\phi} Lf^{-T}
        a = torch.triangular_solve(lf_cov.pow(0.5), pf_chol, upper=False)[0]
        a = a.matmul(a.transpose(-1, -2))
        a = add_diagonal(a, 1)
        a_chol = torch.cholesky(a)

        # B = La^{-1} * Lf^{-1} * \mu_{\phi}.
        b = torch.triangular_solve(lf_mu.unsqueeze(2), pf_chol, upper=False)[0]
        b = torch.triangular_solve(b, a_chol, upper=False)[0]

        if x_test is not None:
            # GP prior.
            ps = self.pf(x_test)
            ps_cov = ps.covariance_matrix

            # GP conditional prior.
            ksf = self.kernels.forward(x_test, x)
            kfs = ksf.transpose(-1, -2)

            # C = Lf^{-1} * Kfs.
            c = torch.triangular_solve(kfs, pf_chol, upper=False)[0]

            # D = La^{-1} * Lf^{-1} * Kfs
            d = torch.triangular_solve(c, a_chol, upper=False)[0]

            qs_mu = torch.triangular_solve(b, a_chol.transpose(-1, -2),
                                           upper=True)[0]
            qs_mu = c.transpose(-1, -2).matmul(qs_mu).squeeze(2)
            qs_cov = ps_cov - d.transpose(-1, -2).matmul(d)
            qs = MultivariateNormal(qs_mu, qs_cov)

            return qs

        else:
            # C = La^{-1} * Lf.
            c = torch.triangular_solve(pf_chol.transpose(-1, -2), a_chol,
                                       upper=False)[0]

            qf_mu = c.transpose(-1, -2).matmul(b).squeeze(2)
            qf_cov = pf_cov - c.transpose(-1, -2).matmul(c)
            qf = MultivariateNormal(qf_mu, qf_cov)

            return qf

    def elbo(self, x, y, mask=None, mask_q=None, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        # Use y_q and mask_q to obtain approximate likelihoods.
        if mask_q is None:
            mask_q = mask

        pf = self.pf(x)
        lf = self.lf(y, mask_q)
        qf = self.qf(pf=pf, lf=lf)

        # KL(q(f) || p(f)) term.
        kl = kl_divergence(qf, pf).sum()

        # log p(y|f) term.
        qf_sigma = torch.stack(
            [cov.diag().pow(0.5) for cov in qf.covariance_matrix])
        qf_marginals = Normal(qf.mean, qf_sigma)
        f_samples = qf_marginals.rsample((num_samples,))

        log_py_f = 0
        for f in f_samples:
            if mask is not None:
                log_py_f += (self.likelihood.log_prob(f.T, y) * mask).sum()
            else:
                log_py_f += self.likelihood.log_prob(f.T, y).sum()

        log_py_f /= num_samples
        elbo = log_py_f - kl

        return elbo / y.shape[0]

    def predict_y(self, x, y=None, mask=None, x_test=None, num_samples=1):
        """Sample predictive posterior."""
        if y is None:
            if x_test is None:
                qf = self.pf(x)
            else:
                qf = self.pf(x_test)
        else:
            qf = self.qf(x=x, y=y, mask=mask, x_test=x_test)

        f_samples = qf.sample((num_samples,))

        y_mus, y_sigmas, y_samples = [], [], []
        for f in f_samples:
            # Output conditional posterior distribution.
            py_f = self.likelihood(f.T)
            y_mus.append(py_f.mean)
            y_samples.append(py_f.sample())

        y_mu = torch.stack(y_mus).mean(0).detach()
        y_sigma = torch.stack(y_samples).std(0).detach()

        return y_mu, y_sigma, y_samples


class SGPVAE(GPVAE):
    """The SGP-VAE model.

    :param likelihood: likelihood function, p(x|f).
    :param variational_dist: variational distribution, l(f|x).
    :param latent_dim (int): the dimension of latent space.
    :param kernel: GP kernel.
    :param z (tensor): initial inducing point locations.
    :param add_jitter (bool, optional): whether to add jitter to the GP
    prior covariance matrix.
    :param fixed_inducing (bool, optional): whether to fix the inducing
    points.
    """
    def __init__(self, likelihood, variational_dist, latent_dim, kernel, z,
                 add_jitter=False, fixed_inducing=False):
        super().__init__(likelihood, variational_dist, latent_dim, kernel,
                         add_jitter)

        if fixed_inducing:
            self.z = nn.Parameter(z, requires_grad=False)
        else:
            self.z = nn.Parameter(z, requires_grad=True)

    def qf(self, x, y=None, pu=None, lf=None, mask=None, diag=False,
           x_test=None, full_cov=False):
        if pu is None:
            pu = self.pf(self.z)

        # Cholesky factor.
        pu_chol = pu.scale_tril

        if lf is None:
            lf = self.lf(y, mask)

        # Reshape.
        lf_mu = lf.mean.transpose(0, 1)
        lf_sigma = lf.stddev.transpose(0, 1)
        lf_precision = lf_sigma.pow(-2).diag_embed()

        # GP conditional prior.
        kfu = self.kernels.forward(x, self.z)
        kuf = kfu.transpose(-1, -2)

        # A = Lu^{-1} * Kuf.
        a = torch.triangular_solve(kuf, pu_chol, upper=False)[0]

        # B = I + A * \Sigma_{\phi}^{-1} * A^T.
        b = a.matmul(lf_precision).matmul(a.transpose(-1, -2))
        b = add_diagonal(b, 1)
        b_chol = torch.cholesky(b)

        # c = Lb^{-1} * Lu^{-1} * Kuf * \Sigma_{\phi}^{-1} * \mu_{\phi}
        c = a.matmul(lf_precision).matmul(lf_mu.unsqueeze(2))
        c = torch.triangular_solve(c, b_chol, upper=False)[0]

        # d = Lu^{-T} * Lb^{-T} * c
        d = torch.triangular_solve(c, b_chol.transpose(-1, -2), upper=True)[0]
        d = torch.triangular_solve(d, pu_chol.transpose(-1, -2), upper=True)[0]

        if x_test is not None:
            # GP prior.
            ps = self.pf(x_test, diag=(not full_cov))
            ps_cov = ps.covariance_matrix

            # GP conditional prior.
            ksu = self.kernels.forward(x_test, self.z)
            kus = ksu.transpose(-1, -2)

            # e = Lu^{-1} * Kus.
            e = torch.triangular_solve(kus, pu_chol, upper=False)[0]

            # g = Lb^{-1} * e
            g = torch.triangular_solve(e, b_chol, upper=False)[0]

            qs_mu = ksu.matmul(d).squeeze(2)
            qs_cov = (ps_cov - e.transpose(-1, -2).matmul(e)
                      + g.transpose(-1, -2).matmul(g))
            qs = MultivariateNormal(qs_mu, qs_cov)

            return qs, None

        else:
            pf = self.pf(x, diag)
            # Covariance and Cholesky factor.
            pf_cov, pf_chol = pf.covariance_matrix, pf.scale_tril

            # e = Lb^{-1} * a
            e = torch.triangular_solve(a, b_chol, upper=False)[0]

            qf_cov = (pf_cov - a.transpose(-1, -2).matmul(a)
                      + e.transpose(-1, -2).matmul(e))
            qf_mu = kfu.matmul(d).squeeze(2)
            qf = MultivariateNormal(qf_mu, qf_cov)

            # g = Lb^{-1} * Lu.
            g = torch.triangular_solve(pu_chol, b_chol, upper=False)[0]

            qu_mu = torch.triangular_solve(c, b_chol.transpose(-1, -2),
                                           upper=True)[0]
            qu_mu = pu_chol.matmul(qu_mu).squeeze(2)
            qu_cov = g.transpose(-1, -2).matmul(g)
            qu = MultivariateNormal(qu_mu, qu_cov)

            return qf, qu

    def elbo(self, x, y, mask=None, num_samples=1):
        """Monte Carlo estimate of the evidence lower bound."""
        pu = self.pf(self.z)
        lf = self.lf(y, mask)
        qf, qu = self.qf(x, pu=pu, lf=lf)

        # KL(q(u) || p(u)) term.
        kl = kl_divergence(qu, pu).sum()

        # log p(y|f) term.
        qf_sigma = torch.stack(
            [cov.diag().pow(0.5) for cov in qf.covariance_matrix])
        qf_marginals = Normal(qf.mean, qf_sigma)
        f_samples = qf_marginals.rsample((num_samples,))
        log_py_f = 0
        for f in f_samples:
            if mask is not None:
                log_py_f += (self.likelihood.log_prob(f.T, y) * mask).sum()
            else:
                log_py_f += self.likelihood.log_prob(f.T, y).sum()

        log_py_f /= num_samples
        elbo = log_py_f - kl

        return elbo / y.shape[0]

    def predict_y(self, x, y=None, mask=None, x_test=None, full_cov=True,
                  num_samples=1):
        """Sample predictive posterior."""
        if y is None:
            if x_test is None:
                qf = self.pf(x)
            else:
                qf = self.pf(x_test)
        else:
            qf = self.qf(x=x, y=y, mask=mask, x_test=x_test,
                         full_cov=full_cov)[0]

        if full_cov:
            f_samples = qf.sample((num_samples,))
        else:
            qf_sigma = torch.stack(
                [cov.diag().pow(0.5) for cov in qf.covariance_matrix])
            qf_marginals = Normal(qf.mean, qf_sigma)
            f_samples = qf_marginals.sample((num_samples,))

        y_mus, y_sigmas, y_samples = [], [], []
        for f in f_samples:
            # Output conditional posterior distribution.
            py_f = self.likelihood(f.T)
            y_mus.append(py_f.mean)
            y_samples.append(py_f.sample())

        y_mu = torch.stack(y_mus).mean(0).detach()
        y_sigma = torch.stack(y_samples).std(0).detach()

        return y_mu, y_sigma, y_samples
