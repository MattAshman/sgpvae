import torch
import torch.nn as nn

__all__ = ['VAE']


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