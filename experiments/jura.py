import argparse
import numpy as np
import pandas as pd
import tqdm
import torch

from scipy.cluster.vq import kmeans2

# Install modules from parent directory.
import sys
sys.path.append('../')
import sgpvae
from data.jura import load

torch.set_default_dtype(torch.float64)


def main(args):
    # Load Jura data.
    train, test = load()

    # Extract data into numpy arrays.
    x = [[i, j] for (i, j) in train.index]
    x = np.array(x)
    y = np.array(train)

    # Normalise observations.
    y_mean, y_std = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
    y = (y - y_mean) / y_std

    # Set up DataLoader.
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = sgpvae.utils.dataset.TupleDataset(x, y, missing=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    # Model construction.
    if args.model in ['gpvae', 'sgpvae', 'vae']:
        kernel = sgpvae.kernels.RBFKernel(
            lengthscale=args.init_lengthscale, scale=args.init_scale)
        decoder = sgpvae.networks.LinearGaussian(
            in_dim=args.latent_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)

        latent_dim = args.latent_dim

    elif args.model in ['gprnvae', 'sgprnvae']:
        f_kernel = sgpvae.kernels.RBFKernel(
            lengthscale=args.init_f_lengthscale, scale=args.init_f_scale)
        w_kernel = sgpvae.kernels.RBFKernel(
            lengthscale=args.init_w_lengthscale, scale=args.init_w_scale)

        decoder = sgpvae.networks.LinearGaussian(
            in_dim=args.w_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)

        latent_dim = args.f_dim + args.f_dim * args.w_dim

    else:
        raise ValueError('{} is not a model'.format(args.model))

    if args.pinference_net == 'factornet':
        encoder = sgpvae.networks.FactorNet(
            in_dim=y.shape[1], out_dim=latent_dim,
            h_dims=args.h_dims, min_sigma=args.min_sigma,
            initial_sigma=args.initial_sigma)

    elif args.pinference_net == 'indexnet':
        encoder = sgpvae.networks.IndexNet(
            in_dim=y.shape[1], out_dim=latent_dim,
            inter_dim=args.inter_dim, h_dims=args.h_dims,
            rho_dims=args.rho_dims, min_sigma=args.min_sigma,
            initial_sigma=args.initial_sigma)

    elif args.pinference_net == 'pointnet':
        encoder = sgpvae.networks.PointNet(
            out_dim=latent_dim, inter_dim=args.inter_dim,
            h_dims=args.h_dims, rho_dims=args.rho_dims,
            min_sigma=args.min_sigma, initial_sigma=args.initial_sigma)

    elif args.pinference_net == 'zeroimputation':
        encoder = sgpvae.networks.LinearGaussian(
            in_dim=y.shape[1], out_dim=latent_dim,
            hidden_dims=args.h_dims, min_sigma=args.min_sigma,
            initial_sigma=args.initial_sigma)

    else:
        raise ValueError('{} is not a partial inference network.'.format(
            args.pinference_net))

    # Construct SGP-VAE model and choose loss function.
    if args.model == 'gpvae':
        model = sgpvae.models.GPVAE(
            encoder, decoder, args.latent_dim, kernel,
            add_jitter=args.add_jitter)
        loss_fn = sgpvae.estimators.gpvae.sa_estimator
        elbo_estimator = sgpvae.estimators.gpvae.elbo_estimator

    elif args.model == 'sgpvae':
        z_init = kmeans2(x.numpy(), args.num_inducing, minit='points')[0]
        z_init = torch.tensor(z_init)

        model = sgpvae.models.SGPVAE(
            encoder, decoder, args.latent_dim, kernel, z_init,
            add_jitter=args.add_jitter, fixed_inducing=args.fixed_inducing)
        loss_fn = sgpvae.estimators.sgpvae.sa_estimator
        elbo_estimator = sgpvae.estimators.sgpvae.elbo_estimator

    elif args.model == 'vae':
        model = sgpvae.models.VAE(encoder, decoder, args.latent_dim)
        loss_fn = sgpvae.estimators.vae.sa_estimator
        elbo_estimator = sgpvae.estimators.vae.elbo_estimator

    elif args.model == 'gprnvae':
        model = sgpvae.models.GPRNVAE(
            encoder, decoder, args.f_dim, args.w_dim, f_kernel, w_kernel,
            add_jitter=args.add_jitter)
        loss_fn = sgpvae.estimators.gprnvae.sa_estimator
        elbo_estimator = sgpvae.estimators.gprnvae.elbo_estimator

    elif args.model == 'sgprnvae':
        z_init = kmeans2(x.numpy(), args.num_inducing, minit='points')[0]
        z_init = torch.tensor(z_init)

        model = sgpvae.models.SGPRNVAE(
            encoder, decoder, args.f_dim, args.w_dim, f_kernel, w_kernel,
            z_init, add_jitter=args.add_jitter,
            fixed_inducing=args.fixed_inducing)
        loss_fn = sgpvae.estimators.sgprnvae.sa_estimator
        elbo_estimator = sgpvae.estimators.sgprnvae.elbo_estimator

    else:
        raise ValueError('{} is not a model.'.format(args.model))

    # Model training.
    model.train(True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_iter = tqdm.tqdm(range(args.epochs), desc='Epoch')
    for epoch in epoch_iter:
        losses = []
        for batch in loader:
            x_b, y_b, m_b, idx_b = batch

            optimiser.zero_grad()
            loss = loss_fn(model, x=x_b, y=y_b, mask=m_b, num_samples=1)
            loss.backward()
            optimiser.step()

            losses.append(loss.item())

        epoch_iter.set_postfix(loss=np.mean(losses))

        if epoch % args.cache_freq == 0:
            elbo = elbo_estimator(model, dataset.x, dataset.y, dataset.m,
                                  num_samples=100)
            tqdm.tqdm.write('ELBO: {:.3f}'.format(elbo))

    # Evaluate model performance.
    elbo = elbo_estimator(model, dataset.x, dataset.y, dataset.m,
                          num_samples=100)
    mean, sigma = model.predict_y(
        x=x, y=y, mask=dataset.m, num_samples=100)[:2]

    mean, sigma = mean.numpy(), sigma.numpy()
    mean = mean * y_std + y_mean
    sigma = sigma * y_std

    pred = pd.DataFrame(mean, index=train.index,
                        columns=train.columns)
    var = pd.DataFrame(sigma ** 2, index=train.index,
                       columns=train.columns)

    mae = sgpvae.utils.metric.mae(pred, test).mean()
    mll = sgpvae.utils.metric.mll(pred, var, test).mean()

    print('\nMAE: {:.3f}'.format(mae))
    print('MLL: {:.3f}'.format(mll))
    print('ELBO: {:.3f}'.format(elbo))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Kernel.
    parser.add_argument('--init_lengthscale', default=1., type=float)
    parser.add_argument('--init_scale', default=1., type=float)
    parser.add_argument('--init_f_lengthscale', default=1., type=float)
    parser.add_argument('--init_f_scale', default=1., type=float)
    parser.add_argument('--init_w_lengthscale', default=1., type=float)
    parser.add_argument('--init_w_scale', default=1., type=float)

    # GPVAE.
    parser.add_argument('--model', default='gpvae')
    parser.add_argument('--pinference_net', default='indexnet', type=str)
    parser.add_argument('--latent_dim', default=2, type=int)
    parser.add_argument('--f_dim', default=2, type=int)
    parser.add_argument('--w_dim', default=3, type=int)
    parser.add_argument('--decoder_dims', default=[20], nargs='+',
                        type=int)
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--h_dims', default=[20], nargs='+', type=int)
    parser.add_argument('--rho_dims', default=[20], nargs='+', type=int)
    parser.add_argument('--inter_dim', default=20, type=int)
    parser.add_argument('--num_inducing', default=64, type=int)
    parser.add_argument('--add_jitter', default=True,
                        type=sgpvae.utils.misc.str2bool)
    parser.add_argument('--min_sigma', default=1e-3, type=float)
    parser.add_argument('--initial_sigma', default=.1, type=float)

    # Training.
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--cache_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    args = parser.parse_args()
    main(args)
