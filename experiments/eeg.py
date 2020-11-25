import argparse
import numpy as np
import pandas as pd
import tqdm
import torch

# Install modules from parent directory.
import sys
sys.path.append('../')
import sgpvae

from data.eeg import load
from sgpvae.utils.misc import str2bool, save

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
    dataset = sgpvae.utils.dataset.TupleDataset(x, y, missing=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    # Model construction.

    # GP prior kernel.
    kernel = sgpvae.kernels.RBFKernel(
        lengthscale=args.init_lengthscale, scale=args.init_scale)

    # Likelihood function.
    if args.likelihood == 'gprn':
        print('Using GPRN likelihood function.')
        likelihood = sgpvae.likelihoods.GPRNHomoGaussian(
            f_dim=args.f_dim, out_dim=y.shape[1], sigma=args.sigma)
        latent_dim = args.f_dim + args.f_dim * y.shape[1]

    elif args.likelihood == 'gprn-nn':
        print('Using GPRN-NN likelihood function.')
        likelihood = sgpvae.likelihoods.GPRNNNHomoGaussian(
            f_dim=args.f_dim, w_dim=args.w_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.f_dim + args.f_dim * args.w_dim

    elif args.likelihood == 'nn-gprn':
        print('Using NN-GPRN likelihood function.')
        likelihood = sgpvae.likelihoods.NNGPRNHomoGaussian(
            f_dim=args.f_dim, w_dim=args.w_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.f_dim + args.w_dim * y.shape[1]

    elif args.likelihood == 'linear':
        print('Using linear likelihood function.')
        likelihood = sgpvae.likelihoods.AffineHomoGaussian(
            in_dim=args.latent_dim, out_dim=y.shape[1], sigma=args.sigma)

    else:
        print('Using NN likelihood function.')
        likelihood = sgpvae.likelihoods.NNHomoGaussian(
            in_dim=args.latent_dim, out_dim=y.shape[1],
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.latent_dim

    # Approximate likelihood function.
    if args.pinference_net == 'factornet':
        variational_dist = sgpvae.likelihoods.FactorNet(
            in_dim=y.shape[1], out_dim=latent_dim,
            h_dims=args.h_dims, min_sigma=args.min_sigma,
            init_sigma=args.initial_sigma)

    elif args.pinference_net == 'indexnet':
        variational_dist = sgpvae.likelihoods.IndexNet(
            in_dim=y.shape[1], out_dim=latent_dim,
            inter_dim=args.inter_dim, h_dims=args.h_dims,
            rho_dims=args.rho_dims, min_sigma=args.min_sigma,
            init_sigma=args.initial_sigma)

    elif args.pinference_net == 'pointnet':
        variational_dist = sgpvae.likelihoods.PointNet(
            out_dim=latent_dim, inter_dim=args.inter_dim,
            h_dims=args.h_dims, rho_dims=args.rho_dims,
            min_sigma=args.min_sigma, initial_sigma=args.initial_sigma)

    elif args.pinference_net == 'zeroimputation':
        variational_dist = sgpvae.likelihoods.NNHeteroGaussian(
            in_dim=y.shape[1], out_dim=latent_dim,
            hidden_dims=args.h_dims, min_sigma=args.min_sigma,
            init_sigma=args.initial_sigma)

    else:
        raise ValueError('{} is not a partial inference network.'.format(
            args.pinference_net))

    # Construct SGP-VAE model.
    if args.model == 'gpvae':
        model = sgpvae.models.GPVAE(
            likelihood, variational_dist, latent_dim, kernel,
            add_jitter=args.add_jitter)

    elif args.model == 'sgpvae':
        z_init = torch.linspace(
            0, x[-1].item(), steps=args.num_inducing).unsqueeze(1)

        model = sgpvae.models.SGPVAE(
            likelihood, variational_dist, args.latent_dim, kernel, z_init,
            add_jitter=args.add_jitter, fixed_inducing=args.fixed_inducing)

    elif args.model == 'vae':
        model = sgpvae.models.VAE(likelihood, variational_dist, args.latent_dim)

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
            # loss = -model.elbo(x_b, y_b, m_b, num_samples=1)
            loss = -model.elbo(x_b, y_b, m_b, num_samples=1)
            loss.backward()
            optimiser.step()

            losses.append(loss.item())

        epoch_iter.set_postfix(loss=np.mean(losses))

        if epoch % args.cache_freq == 0:
            elbo = model.elbo(dataset.x, dataset.y, dataset.m, num_samples=100)
            elbo *= dataset.x.shape[0]

            tqdm.tqdm.write('ELBO: {:.3f}'.format(elbo))

    # Evaluate model performance.
    elbo = model.elbo(dataset.x, dataset.y, dataset.m, num_samples=100)
    elbo *= dataset.x.shape[0]
    mean, sigma = model.predict_y(
        x=dataset.x, y=dataset.y, mask=dataset.m, num_samples=100)[:2]

    mean, sigma = mean.numpy(), sigma.numpy()
    mean = mean * y_std + y_mean
    sigma = sigma * y_std

    pred = pd.DataFrame(mean, index=train.index,
                        columns=train.columns)
    var = pd.DataFrame(sigma ** 2, index=train.index,
                       columns=train.columns)

    smse = sgpvae.utils.metric.smse(pred, test).mean()
    mll = sgpvae.utils.metric.mll(pred, var, test).mean()

    print('\nSMSE: {:.3f}'.format(smse))
    print('MLL: {:.3f}'.format(mll))
    print('ELBO: {:.3f}'.format(elbo))

    if args.save:
        metrics = {'ELBO': elbo, 'SMSE': smse, 'MLL': mll}
        save(vars(args), metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Kernel.
    parser.add_argument('--init_lengthscale', default=0.05, type=float)
    parser.add_argument('--init_scale', default=1., type=float)

    # GPVAE.
    parser.add_argument('--model', default='gpvae')
    parser.add_argument('--likelihood', default='nn', type=str)
    parser.add_argument('--pinference_net', default='indexnet', type=str)
    parser.add_argument('--latent_dim', default=3, type=int)
    parser.add_argument('--f_dim', default=3, type=int)
    parser.add_argument('--w_dim', default=3, type=int)
    parser.add_argument('--decoder_dims', default=[20], nargs='+',
                        type=int)
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--h_dims', default=[20], nargs='+', type=int)
    parser.add_argument('--rho_dims', default=[20], nargs='+', type=int)
    parser.add_argument('--inter_dim', default=20, type=int)
    parser.add_argument('--num_inducing', default=64, type=int)
    parser.add_argument('--fixed_inducing', default=False, type=str2bool)
    parser.add_argument('--add_jitter', default=True,
                        type=sgpvae.utils.misc.str2bool)
    parser.add_argument('--min_sigma', default=1e-3, type=float)
    parser.add_argument('--initial_sigma', default=.1, type=float)

    # Training.
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--cache_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    # General.
    parser.add_argument('--save', default=False, type=str2bool)
    parser.add_argument('--results_dir', default='./_results/eeg/', type=str)

    args = parser.parse_args()
    main(args)
