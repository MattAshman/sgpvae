import copy
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
from .kernels import KernelList
from .utils.matrix import add_diagonal

__all__ = ['VAE', 'GPVAE', 'SGPVAE']

JITTER = 1e-5


class VAE(nn.Module):
    """VAE with standard normal prior.

    :param encoder: encoder network.
    :param decoder: decoder network.
    :param latent_dim (int): latent space dimensionality.
    """
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def latent_prior(self, x):
        """Return latent prior mean and covariance."""
        pf_mu = torch.zeros(self.latent_dim, x.shape[0])
        pf_cov = torch.ones(self.latent_dim, x.shape[0]).diag_embed()

        return pf_mu, pf_cov

    def latent_dists(self, x, y, mask=None):
        """Return latent posterior and prior mean and covariance."""
        if mask is not None:
            qf_mu, qf_sigma = self.encoder(y, mask)
        else:
            qf_mu, qf_sigma = self.encoder(y)

        # Reshape.
        qf_mu = qf_mu.transpose(0, 1)
        qf_sigma = qf_sigma.transpose(0, 1)
        qf_cov = qf_sigma.pow(2).diag_embed()

        # Prior.
        pf_mu, pf_cov = self.latent_prior(x)

        return qf_mu, qf_cov, pf_mu, pf_cov

    def sample_posterior(self, x, y=None, mask=None, num_samples=1, **kwargs):
        """Sample latent posterior."""
        if y is not None:
            qf_mu, qf_cov = self.latent_dists(x, y, mask)[:2]
        else:
            qf_mu, qf_cov = self.latent_prior(x)

        qf_sigma = torch.stack([cov.diag() for cov in qf_cov]) ** 0.5
        samples = [qf_mu + qf_sigma * torch.randn_like(qf_mu)
                   for _ in range(num_samples)]

        return samples

    def predict_y(self, **kwargs):
        """Sample predictive posterior."""
        f_samples = self.sample_posterior(**kwargs)

        y_mus, y_sigmas, y_samples = [], [], []
        for f in f_samples:
            # Output conditional posterior distribution.
            y_mu, y_sigma = self.decoder(f.transpose(0, 1))
            y_mus.append(y_mu)
            y_sigmas.append(y_sigma)
            y_samples.append(y_mu + y_sigma * torch.randn_like(y_mu))

        y_mu = torch.stack(y_mus).mean(0).detach()
        y_sigma = torch.stack(y_samples).std(0).detach()

        return y_mu, y_sigma, y_samples


class GPVAE(VAE):
    """The GP-VAE model from Spatio-Temporal Variational Autoencoders.

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
        lf_y_precision = lf_y_sigma.pow(-2).diag_embed()
        lf_y_root_precision = lf_y_sigma.pow(-1).diag_embed()

        # GP prior.
        pf_mu, kff = self.latent_prior(x)

        # See GPML section 3.4.3.
        # TODO: Make more efficient?
        a = kff.matmul(lf_y_root_precision)
        at = a.transpose(-1, -2)
        w = lf_y_root_precision.matmul(a)
        w = add_diagonal(w, 1)
        winv = w.inverse()

        if x_test is not None:
            # GP prior.
            ps_mu, kss = self.latent_prior(x_test)

            # GP conditional prior.
            ksf = self.kernels.forward(x_test, x)
            kfs = ksf.transpose(-1, -2)

            # GP test posterior.
            b = lf_y_root_precision.matmul(winv.matmul(lf_y_root_precision))
            c = ksf.matmul(b)
            qs_cov = kss - c.matmul(kfs)
            qs_mu = c.matmul(lf_y_mu.unsqueeze(2))
            qs_mu = qs_mu.squeeze(2)

            return qs_mu, qs_cov, ps_mu, kss
        else:
            # GP training posterior.
            qf_cov = kff - a.matmul(winv.matmul(at))
            qf_mu = qf_cov.matmul(lf_y_precision.matmul(lf_y_mu.unsqueeze(2)))
            qf_mu = qf_mu.squeeze(2)

            return qf_mu, qf_cov, pf_mu, kff, lf_y_mu, lf_y_cov

    def sample_posterior(self, x, y, mask=None, num_samples=1, **kwargs):
        """Sample latent posterior."""
        qf_mu, qf_cov = self.latent_dists(x, y, mask, **kwargs)[:2]

        qf = MultivariateNormal(qf_mu, qf_cov)
        samples = [qf.sample() for _ in range(num_samples)]

        return samples


class SGPVAE(GPVAE):
    """The SGP-VAE model from Spatio-Temporal Variational Autoencoders.

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

        # TODO: make more efficient?
        kuu_inv = kuu.inverse()
        phi = (kuu + kuf.matmul(lf_y_precision).matmul(kfu)).inverse()
        qu_mu = kuu.matmul(phi.matmul(kuf.matmul(lf_y_precision.matmul(
            lf_y_mu.unsqueeze(2)))))

        if x_test is not None:
            # GP prior.
            ps_mu, kss = self.latent_prior(x_test, diag=(not full_cov))

            # GP conditional prior.
            ksu = self.kernels.forward(x_test, self.z)
            kus = ksu.transpose(-1, -2)

            qs_cov = kss - ksu.matmul(kuu_inv - phi).matmul(kus)
            qs_mu = ksu.matmul(kuu_inv.matmul(qu_mu)).squeeze(2)

            return qs_mu, qs_cov, ps_mu, kss
        else:
            # GP prior.
            # Note that only diagonals are needed when optimising ELBO.
            pf_mu, kff = self.latent_prior(x, diag=(not full_cov))

            qf_cov = kff - kfu.matmul(kuu_inv - phi).matmul(kuf)
            qf_mu = kfu.matmul(kuu_inv.matmul(qu_mu)).squeeze(2)

            qu_cov = kuu.matmul(phi.matmul(kuu))
            qu_mu = qu_mu.squeeze(2)

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
