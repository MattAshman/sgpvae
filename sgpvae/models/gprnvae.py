import copy
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
from sgpvae.kernels import KernelList
from sgpvae.utils.matrix import add_diagonal

__all__ = ['GPRNVAE', 'SGPRNVAE']

JITTER = 1e-5


class GPRNVAE(nn.Module):
    """The GPRN-VAE model.

    :param encoder: encoder network.
    :param decoder: decoder network.
    :param f_dim (int): the dimension of latent space.
    :param w_dim (int): the dimension of the first hidden layer,
    whose weights are distributed according to a GP.
    :param f_kernel: GP kernel over function values.
    :param w_kernel: GP kernel over weights.
    :param add_jitter (bool, optional): whether to add jitter to the GP prior
    covariance matrix.
    """
    def __init__(self, encoder, decoder, f_dim, w_dim, f_kernel, w_kernel,
                 add_jitter=False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.f_dim = f_dim
        self.w_dim = w_dim
        self.add_jitter = add_jitter

        if not isinstance(f_kernel, list):
            f_kernels = [copy.deepcopy(f_kernel) for _ in range(f_dim)]
            self.f_kernels = KernelList(f_kernels)

        else:
            assert len(f_kernel) == f_dim, 'Number of kernels must be equal ' \
                                           'to the latent dimension.'
            self.f_kernels = KernelList(copy.deepcopy(f_kernel))

        if not isinstance(w_kernel, list):
            w_kernels = [copy.deepcopy(w_kernel) for _ in range(f_dim * w_dim)]
            self.w_kernels = KernelList(w_kernels)

        else:
            assert len(w_kernel) == w_dim, 'Number of kernels must be equal ' \
                                           'to the first hidden layer ' \
                                           'dimension.'
            self.w_kernels = KernelList(copy.deepcopy(w_kernel))

    def f_prior(self, x, diag=False):
        """Return latent prior mean and covariance."""
        mf = torch.zeros(self.f_dim, x.shape[0])
        kff = self.f_kernels.forward(x, x, diag)

        if self.add_jitter:
            # Add jitter to improve condition number.
            kff = add_diagonal(kff, JITTER)

        return mf, kff

    def w_prior(self, x, diag=False):
        """Return latent prior mean and covariance."""
        mw = torch.zeros((self.f_dim * self.w_dim), x.shape[0])
        kww = self.w_kernels.forward(x, x, diag)

        if self.add_jitter:
            # Add jitter to improve condition number.
            kww = add_diagonal(kww, JITTER)

        return mw, kww

    def latent_prior(self, x, diag=False):
        mf, kff = self.f_prior(x, diag)
        mw, kww = self.w_prior(x, diag)

        mz = torch.cat([mf, mw], dim=0)
        kzz = torch.cat([kff, kww], dim=0)

        return mz, kzz

    def latent_dists(self, x, y, mask=None, x_test=None):
        """Return latent posterior and prior mean and covariance."""
        if mask is not None:
            lz_y_mu, lz_y_sigma = self.encoder(y, mask)
        else:
            lz_y_mu, lz_y_sigma = self.encoder(y)

        # Reshape.
        lz_y_mu = lz_y_mu.transpose(0, 1)
        lz_y_sigma = lz_y_sigma.transpose(0, 1)
        lz_y_cov = lz_y_sigma.pow(2).diag_embed()
        lz_y_precision = lz_y_sigma.pow(-2).diag_embed()
        lz_y_root_precision = lz_y_sigma.pow(-1).diag_embed()

        # GP prior.
        pz_mu, kzz = self.latent_prior(x)

        # See GPML section 3.4.3.
        # TODO: Make more efficient?
        a = kzz.matmul(lz_y_root_precision)
        at = a.transpose(-1, -2)
        w = lz_y_root_precision.matmul(a)
        w = add_diagonal(w, 1)
        winv = w.inverse()

        if x_test is not None:
            # GP prior.
            ps_mu, kss = self.latent_prior(x_test)

            # GP conditional prior.
            ksz = self.kernels.forward(x_test, x)
            kzs = ksz.transpose(-1, -2)

            # GP test posterior.
            b = lz_y_root_precision.matmul(winv.matmul(lz_y_root_precision))
            c = ksz.matmul(b)
            qs_cov = kss - c.matmul(kzs)
            qs_mu = c.matmul(lz_y_mu.unsqueeze(2))
            qs_mu = qs_mu.squeeze(2)

            return qs_mu, qs_cov, ps_mu, kss
        else:
            # GP training posterior.
            qz_cov = kzz - a.matmul(winv.matmul(at))
            qz_mu = qz_cov.matmul(lz_y_precision.matmul(lz_y_mu.unsqueeze(2)))
            qz_mu = qz_mu.squeeze(2)

            return qz_mu, qz_cov, pz_mu, kzz, lz_y_mu, lz_y_cov

    def sample_posterior(self, x, y, mask=None, num_samples=1, **kwargs):
        """Sample latent posterior."""
        qz_mu, qz_cov = self.latent_dists(x, y, mask, **kwargs)[:2]

        qf_mu, qf_cov = qz_mu[:self.f_dim, :], qz_cov[:self.f_dim, ...]
        qw_mu, qw_cov = qz_mu[self.f_dim:, :], qz_cov[self.f_dim:, ...]

        qf = MultivariateNormal(qf_mu, qf_cov)
        qw = MultivariateNormal(qw_mu, qw_cov)
        f_samples = [qf.sample() for _ in range(num_samples)]
        w_samples = [qw.sample() for _ in range(num_samples)]

        return f_samples, w_samples

    def predict_y(self, x, y, **kwargs):
        """Sample predictive posterior."""
        f_samples, w_samples = self.sample_posterior(x, y, **kwargs)

        y_mus, y_sigmas, y_samples = [], [], []
        for f, w in zip(f_samples, w_samples):
            # Output conditional posterior distribution.
            g = torch.zeros(self.w_dim, x.shape[0])
            for n in range(x.shape[0]):
                wn = w[:, n].reshape(self.w_dim, self.f_dim)
                # wn = torch.sigmoid(wn)
                g[:, n] = wn.matmul(f[:, n])

            y_mu, y_sigma = self.decoder(g.T)
            y_mus.append(y_mu)
            y_sigmas.append(y_sigma)
            y_samples.append(y_mu + y_sigma * torch.randn_like(y_mu))

        y_mu = torch.stack(y_mus).mean(0).detach()
        y_sigma = torch.stack(y_samples).std(0).detach()

        return y_mu, y_sigma, y_samples


class SGPRNVAE(GPRNVAE):
    """The SGPRN-VAE.

    :param encoder: encoder network.
    :param decoder: decoder network.
    :param f_dim (int): the dimension of latent space.
    :param w_dim (int): the dimension of the first hidden layer,
    whose weights are distributed according to a GP.
    :param f_kernel: GP kernel over function values.
    :param w_kernel: GP kernel over weights.
    :param z (tensor): initial inducing point locations.
    :param add_jitter (bool, optional): whether to add jitter to the GP
    prior covariance matrix.
    :param fixed_inducing (bool, optional): whether to fix the inducing
    points.
    """
    def __init__(self, encoder, decoder, f_dim, w_dim, f_kernel, w_kernel, z,
                 add_jitter=False, fixed_inducing=False):
        super().__init__(encoder, decoder, f_dim, w_dim, f_kernel, w_kernel,
                         add_jitter)

        if fixed_inducing:
            self.z = nn.Parameter(z, requires_grad=False)
        else:
            self.z = nn.Parameter(z, requires_grad=True)

    def latent_dists(self, x, y, mask=None, x_test=None, full_cov=False):
        # Likelihood terms.
        if mask is not None:
            lz_y_mu, lz_y_sigma = self.encoder(y, mask)
        else:
            lz_y_mu, lz_y_sigma = self.encoder(y)

        # Reshape.
        lz_y_mu = lz_y_mu.T
        lz_y_sigma = lz_y_sigma.T
        lz_y_cov = lz_y_sigma.pow(2).diag_embed()
        lz_y_precision = lz_y_sigma.pow(-2).diag_embed()

        # GP prior.
        pu_mu, kuu = self.latent_prior(self.z)

        # GP conditional prior.
        kzu = self.kernels.forward(x, self.z)
        kuz = kzu.transpose(-1, -2)

        # TODO: make more efficient?
        kuu_inv = kuu.inverse()
        phi = (kuu + kuz.matmul(lz_y_precision).matmul(kzu)).inverse()
        qu_mu = kuu.matmul(phi.matmul(kuz.matmul(lz_y_precision.matmul(
            lz_y_mu.unsqueeze(2)))))

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
            pz_mu, kzz = self.latent_prior(x, diag=(not full_cov))

            qz_cov = kzz - kzu.matmul(kuu_inv - phi).matmul(kuz)
            qz_mu = kzu.matmul(kuu_inv.matmul(qu_mu)).squeeze(2)

            qu_cov = kuu.matmul(phi.matmul(kuu))
            qu_mu = qu_mu.squeeze(2)

            return qz_mu, qz_cov, qu_mu, qu_cov, pu_mu, kuu, lz_y_mu, lz_y_cov

    def sample_posterior(self, x, y, mask=None, num_samples=1,
                         full_cov=True, **kwargs):
        """Sample latent posterior."""
        qz_mu, qz_cov = self.latent_dists(x, y, mask, **kwargs)[:4]

        qf_mu, qf_cov = qz_mu[:self.f_dim, :], qz_cov[:self.f_dim, ...]
        qw_mu, qw_cov = qz_mu[self.f_dim:, :], qz_cov[self.f_dim:, ...]

        if full_cov:
            qf = MultivariateNormal(qf_mu, qf_cov)
            qw = MultivariateNormal(qw_mu, qw_cov)
            f_samples = [qf.sample() for _ in range(num_samples)]
            w_samples = [qw.sample() for _ in range(num_samples)]
        else:
            qf_sigma = torch.stack([cov.diag() for cov in qf_cov]) ** 0.5
            qw_sigma = torch.stack([cov.diag() for cov in qw_cov]) ** 0.5
            f_samples = [qf_mu + qf_sigma * torch.randn_like(qf_mu)
                         for _ in range(num_samples)]
            w_samples = [qw_mu + qw_sigma * torch.randn_like(qw_mu)
                         for _ in range(num_samples)]

        return f_samples, w_samples