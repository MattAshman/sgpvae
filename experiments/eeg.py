import numpy as np
import pandas as pd
import tqdm
import torch
import sgpvae
from data.eeg import load

torch.set_default_dtype(torch.float64)


def main(args):
    # Load EEG data.
    _, train, test = load()
    x = np.array(train.index)
    y = np.array(train)

    # Normalise observations.
    y_mean, y_std = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    y = (y - y_mean) / y_std

    # Set up DataLoader.
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = sgpvae.utils.dataset.TupleDataset(x, y, contains_nan=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    # Model construction.
    kernel = sgpvae.kernels.RBFKernel(
        lengthscale=args.init_lengthscale, scale=args.init_scale)
    decoder = sgpvae.networks.LinearGaussian(
        in_dim=args.latent_dim, out_dim=y.shape[1],
        hidden_dims=args.decoder_dims, sigma=args.sigma)

    if args.pn == 'factornet':
        encoder = sgpvae.networks.FactorNet(
            in_dim=y.shape[1], out_dim=args.latent_dim,
            h_dims=args.h_dims, min_sigma=args.min_sigma)
    elif args.pn == 'indexnet':
        encoder = sgpvae.networks.IndexNet(
            in_dim=y.shape[1], out_dim=args.latent_dim,
            inter_dim=args.inter_dim, h_dims=args.h_dims,
            rho_dims=args.rho_dims, min_sigma=args.min_sigma)
    elif args.pn == 'pointnet':
        encoder = sgpvae.networks.PointNet(
            out_dim=args.latent_dim, inter_dim=args.inter_dim,
            h_dims=args.h_dims, rho_dims=args.rho_dims,
            min_sigma=args.min_sigma)
    elif args.pn == 'zeroimputation':
        encoder = sgpvae.networks.LinearGaussian(
            in_dim=y.shape[1], out_dim=args.latent_dim,
            hidden_dims=args.h_dims, min_sigma=args.min_sigma)
    else:
        raise ValueError('{} is not a partial inference network.'.format(
            args.pn))

    # Construct SGP-VAE model and choose loss function.
    if args.model == 'gpvae':
        model = sgpvae.models.GPVAE(encoder, decoder, args.latent_dim,
                                    kernel, add_jitter=args.add_jitter)
        loss_fn = sgpvae.estimators.gpvae.sa_estimator

    elif args.model == 'sgpvae':
        z_init = torch.linspace(
            0, x[-1].item(), steps=args.num_inducing).unsqueeze(1)

        model = sgpvae.models.SGPVAE(
            encoder, decoder, args.latent_dim, kernel, z_init,
            add_jitter=args.add_jitter, fixed_inducing=args.fixed_inducing)
        loss_fn = sgpvae.estimators.sgpvae.sa_estimator

    elif args.model == 'vae':
        model = sgpvae.models.VAE(encoder, decoder, args.latent_dim)
        loss_fn = sgpvae.estimators.vae.sa_estimator

    else:
        raise ValueError('{} is not a model.'.format(args.model))

    # Model training.
    model.train(True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_iter = tqdm.tqdm(range(args.epochs), desc='Epoch')
    for _ in epoch_iter:
        losses = []
        for batch in loader:
            x_b, y_b, m_b, idx_b = batch

            optimiser.zero_grad()
            loss = loss_fn(model, x=x_b, y=y_b, mask=m_b, num_samples=1)
            loss.backward()
            optimiser.step()

            losses.append(loss.item())

        epoch_iter.set_postfix(loss=np.mean(losses))

    # Evaluate model performance.
    mean, sigma = model.predict_y(
        x=x, y=y, mask=dataset.m, num_samples=100)
    mean, sigma = mean.numpy(), sigma.numpy()
    mean = mean * y_std + y_mean
    sigma = sigma * y_std
    pred = pd.DataFrame(mean, index=train.index,
                        columns=train.columns)
    var = pd.DataFrame(sigma ** 2, index=train.index,
                       columns=train.columns)

    smse = sgpvae.metric.smse(pred, test).mean()
    mll = sgpvae.metric.mll(pred, var, test).mean()

    print('SMSE: {:.3f}\n'.format(smse))
    print('MLL: {:.3f}\n'.format(mll))
