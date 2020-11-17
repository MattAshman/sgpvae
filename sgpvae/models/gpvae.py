import copy
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
from sgpvae.kernels import KernelList
from sgpvae.utils.matrix import add_diagonal

from .base import VAE

__all__ = ['GPVAE', 'SGPVAE']

JITTER = 1e-5


class GPVAE(VAE):
    """The GP-VAE model.

    :param encoder: encoder network.
    :param decoder: decoder network.
    :param latent_dim (int): the dimension of latent space.
    :param kernel: GP kernel.
    :param add_jitter (bool, optional): whether to add jitter to the GP prior
    covariance matrix.
    """
    def __init__(self, encoder, decoder, latent_dim, kernel, add_jitter=False):
        super().__init__(encoder, decoder, latent_dim)

        self.add_jitter = add_jitter

        if not isinstance(kernel, list):
            kernels = [copy.deepcopy(kernel) for _ in range(latent_dim)]
            self.kernels = KernelList(kernels)

        else:
            assert len(kernel) == latent_dim, 'Number of kernels must be ' \
                                              'equal to the latent dimension.'
            self.kernels = KernelList(copy.deepcopy(kernel))

    def latent_prior(self, x, diag=False):
        """Return latent prior mean and covariance."""
        mf = torch.zeros(self.latent_dim, x.shape[0])
        kff = self.kernels.forward(x, x, diag)

        if self.add_jitter:
            # Add jitter to improve condition number.
            kff = add_diagonal(kff, JITTER)

        return mf, kff

    def latent_dists(self, x, y, mask=None, x_test=None):
        """Return latent posterior and prior mean and covariance."""
        if mask is not None:
            lf_y_mu, lf_y_sigma = self.encoder(y, mask)
        else:
            lf_y_mu, lf_y_sigma = self.encoder(y)

        # Reshape.
        lf_y_mu = lf_y_mu.transpose(0, 1)
        lf_y_sigma = lf_y_sigma.transpose(0, 1)
        lf_y_cov = lf_y_sigma.pow(2).diag_embed()

        # GP prior.
        pf_mu, kff = self.latent_prior(x)

        # See GPML section 3.4.3.
        kff_chol = kff.cholesky()

        # A = I + Lf^{-1} \Sigma_{\phi} Lf^{-T}
        a = torch.triangular_solve(lf_y_cov.pow(0.5), kff_chol, upper=False)[0]
        a = a.matmul(a.transpose(-1, -2))
        a = add_diagonal(a, 1)
        la = torch.cholesky(a)

        # B = La^{-1} * Lf^{-1} * \mu_{\phi}.
        b = torch.triangular_solve(lf_y_mu.unsqueeze(2), kff_chol,
                                   upper=False)[0]
        b = torch.triangular_solve(b, la, upper=False)[0]

        if x_test is not None:
            # GP prior.
            ps_mu, kss = self.latent_prior(x_test)

            # GP conditional prior.
            ksf = self.kernels.forward(x_test, x)
            kfs = ksf.transpose(-1, -2)

            # C = Lf^{-1} * Kfs.
            c = torch.triangular_solve(kfs, kff_chol, upper=False)[0]

            # D = La^{-1} * Lf^{-1} * Kfs
            d = torch.triangular_solve(c, la, upper=False)[0]

            qs_mu = torch.triangular_solve(b, la.transpose(-1, -2),
                                           upper=True)[0]
            qs_mu = c.transpose(-1, -2).matmul(qs_mu).squeeze(2)
            qs_cov = kss - d.transpose(-1, -2).matmul(d)

            return qs_mu, qs_cov, ps_mu, kss

        else:
            # C = La^{-1} * Lf.
            c = torch.triangular_solve(kff_chol.transpose(-1, -2), la,
                                       upper=False)[0]

            qf_cov = kff - c.transpose(-1, -2).matmul(c)
            qf_mu = c.transpose(-1, -2).matmul(b).squeeze(2)

            return qf_mu, qf_cov, pf_mu, kff, lf_y_mu, lf_y_cov

    def sample_posterior(self, x, y, mask=None, num_samples=1, **kwargs):
        """Sample latent posterior."""
        qf_mu, qf_cov = self.latent_dists(x, y, mask, **kwargs)[:2]

        qf = MultivariateNormal(qf_mu, qf_cov)
        samples = qf.sample((num_samples,))

        return samples


class SGPVAE(GPVAE):
    """The SGP-VAE model.

    :param encoder: encoder network.
    :param decoder: decoder network.
    :param latent_dim (int): dimension of latent space.
    :param kernel: the GP kernel.
    :param z (tensor): initial inducing point locations.
    :param add_jitter (bool, optional): whether to add jitter to the GP
    prior covariance matrix.
    :param fixed_inducing (bool, optional): whether to fix the inducing
    points.
    """
    def __init__(self, encoder, decoder, latent_dim, kernel, z,
                 add_jitter=False, fixed_inducing=False):
        super().__init__(encoder, decoder, latent_dim, kernel, add_jitter)

        if fixed_inducing:
            self.z = nn.Parameter(z, requires_grad=False)
        else:
            self.z = nn.Parameter(z, requires_grad=True)

    def latent_dists(self, x, y, mask=None, x_test=None, full_cov=False):
        # Likelihood terms.
        if mask is not None:
            lf_y_mu, lf_y_sigma = self.encoder(y, mask)
        else:
            lf_y_mu, lf_y_sigma = self.encoder(y)

        # Reshape.
        lf_y_mu = lf_y_mu.T
        lf_y_sigma = lf_y_sigma.T
        lf_y_cov = lf_y_sigma.pow(2).diag_embed()
        lf_y_precision = lf_y_sigma.pow(-2).diag_embed()

        # GP prior.
        pu_mu, kuu = self.latent_prior(self.z)

        # GP conditional prior.
        kfu = self.kernels.forward(x, self.z)
        kuf = kfu.transpose(-1, -2)

        lu = kuu.cholesky()

        # A = Lu^{-1} * Kuf.
        a = torch.triangular_solve(kuf, lu, upper=False)[0]

        # B = I + A * \Sigma_{\phi}^{-1} * A^T.
        b = a.matmul(lf_y_precision).matmul(a.transpose(-1, -2))
        b = add_diagonal(b, 1)
        lb = torch.cholesky(b)

        # c = Lb^{-1} * Lu^{-1} * Kuf * \Sigma_{\phi}^{-1} * \mu_{\phi}
        c = a.matmul(lf_y_precision).matmul(lf_y_mu.unsqueeze(2))
        c = torch.triangular_solve(c, lb, upper=False)[0]

        # d = Lu^{-T} * Lb^{-T} * c
        d = torch.triangular_solve(c, lb.transpose(-1, -2), upper=True)[0]
        d = torch.triangular_solve(d, lu.transpose(-1, -2), upper=True)[0]

        if x_test is not None:
            # GP prior.
            ps_mu, kss = self.latent_prior(x_test, diag=(not full_cov))

            # GP conditional prior.
            ksu = self.kernels.forward(x_test, self.z)
            kus = ksu.transpose(-1, -2)

            # e = Lu^{-1} * Kus.
            e = torch.triangular_solve(lu, kus, upper=False)[0]

            # g = Lb^{-1} * e
            g = torch.triangular_solve(e, lb, upper=False)[0]

            qs_cov = (kss - e.transpose(-1, -2).matmul(e)
                      + g.transpose(-1, -2).matmul(g))
            qs_mu = ksu.matmul(d).squeeze(2)

            return qs_mu, qs_cov, ps_mu, kss

        else:
            # GP prior.
            pf_mu, kff = self.latent_prior(x, diag=(not full_cov))

            # e = Lb^{-1} * a
            e = torch.triangular_solve(a, lb, upper=False)[0]

            qf_cov = (kff - a.transpose(-1, -2).matmul(a)
                      + e.transpose(-1, -2).matmul(e))
            qf_mu = kfu.matmul(d).squeeze(2)

            # g = Lb^{-1} * Lu.
            g = torch.triangular_solve(lu, lb, upper=False)[0]

            qu_cov = g.transpose(-1, -2).matmul(g)
            qu_mu = torch.triangular_solve(c, lb.transpose(-1, -2),
                                           upper=True)[0]
            qu_mu = lu.matmul(qu_mu).squeeze(2)

            return qf_mu, qf_cov, qu_mu, qu_cov, pu_mu, kuu, lf_y_mu, lf_y_cov

    def sample_posterior(self, x, y, mask=None, num_samples=1,
                         full_cov=True, **kwargs):
        """Sample latent posterior."""
        qf_mu, qf_cov = self.latent_dists(
            x, y, mask, full_cov=full_cov, **kwargs)[:2]

        if full_cov:
            qf = MultivariateNormal(qf_mu, qf_cov)
            samples = [qf.sample() for _ in range(num_samples)]
        else:
            qf_sigma = torch.stack([cov.diag() for cov in qf_cov]) ** 0.5
            samples = [qf_mu + qf_sigma * torch.randn_like(qf_mu)
                       for _ in range(num_samples)]

        return samples



