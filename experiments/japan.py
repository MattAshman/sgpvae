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
from data.japan import load

torch.set_default_dtype(torch.float64)


def preprocess():
    data = load()

    df = data['all']
    train = data['train']
    train_1980 = data['train_1980']
    test_1980 = data['test_1980']
    train_1981 = data['train_1981']
    test_1981 = data['test_1981']

    inputs = ['lat', 'lon', 'elev', 'day']
    observations = ['PRCP', 'TMAX', 'TMIN', 'TAVG', 'SNWD']

    # Extract data into numpy arrays and perform preprocessing.
    x_train = np.array(list(map(lambda x: x[inputs].to_numpy(), train)))
    y_train = np.array(list(map(lambda x: x[observations].to_numpy(), train)))

    # 1980 data.
    x_train_1980 = np.array(list(map(lambda x: x[inputs].to_numpy(),
                                     train_1980)))
    y_train_1980 = np.array(list(map(lambda x: x[observations].to_numpy(),
                                     train_1980)))
    x_test_1980 = np.array(list(map(lambda x: x[inputs].to_numpy(),
                                    test_1980)))

    # 1981 data.
    x_train_1981 = np.array(list(map(lambda x: x[inputs].to_numpy(),
                                     train_1981)))
    y_train_1981 = np.array(list(map(lambda x: x[observations].to_numpy(),
                                     train_1981)))
    x_test_1981 = np.array(list(map(lambda x: x[inputs].to_numpy(),
                                    test_1981)))

    # Normalise data.
    y_mean = df[observations].mean().to_numpy()
    y_std = df[observations].std().to_numpy()
    x_mean = df[inputs].mean().to_numpy()
    x_std = df[inputs].std().to_numpy()

    y_train = list(map(lambda x: (x - y_mean) / y_std, y_train))
    x_train = list(map(lambda x: (x - x_mean) / x_std, x_train))
    y_train_1980 = list(map(lambda x: (x - y_mean) / y_std, y_train_1980))
    x_train_1980 = list(map(lambda x: (x - x_mean) / x_std, x_train_1980))
    y_train_1981 = list(map(lambda x: (x - y_mean) / y_std, y_train_1981))
    x_train_1981 = list(map(lambda x: (x - x_mean) / x_std, x_train_1981))
    x_test_1980 = list(map(lambda x: (x - x_mean) / x_std, x_test_1980))
    x_test_1981 = list(map(lambda x: (x - x_mean) / x_std, x_test_1981))

    # Convert to tensors.
    x_train = list(map(lambda x: torch.tensor(x), x_train))
    y_train = list(map(lambda x: torch.tensor(x), y_train))
    x_train_1980 = list(map(lambda x: torch.tensor(x), x_train_1980))
    y_train_1980 = list(map(lambda x: torch.tensor(x), y_train_1980))
    x_train_1981 = list(map(lambda x: torch.tensor(x), x_train_1981))
    y_train_1981 = list(map(lambda x: torch.tensor(x), y_train_1981))
    x_test_1980 = list(map(lambda x: torch.tensor(x), x_test_1980))
    x_test_1981 = list(map(lambda x: torch.tensor(x), x_test_1981))

    preprocessed_data = {'x_train': x_train,
                         'y_train': y_train,
                         'x_train_1980': x_train_1980,
                         'y_train_1980': y_train_1980,
                         'x_train_1981': x_train_1981,
                         'y_train_1981': y_train_1981,
                         'x_test_1980': x_test_1980,
                         'x_test_1981': x_test_1981,
                         'test_1980': test_1980,
                         'test_1981': test_1981,
                         'y_mean': y_mean,
                         'y_std': y_std,
                         'x_mean': x_mean,
                         'x_std': x_std
                         }

    return preprocessed_data


def main(args):
    data = preprocess()

    # Set up dataset and dataloaders.
    train_dataset = sgpvae.utils.dataset.MetaTupleDataset(
        data['x_train'], data['y_train'], missing=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False)

    # For model evaluation.
    train_1980_dataset = sgpvae.utils.dataset.MetaTupleDataset(
        data['x_train_1980'], data['y_train_1980'], missing=True)
    train_1981_dataset = sgpvae.utils.dataset.MetaTupleDataset(
        data['x_train_1981'], data['y_train_1981'], missing=True)

    # Model construction.
    kernel = sgpvae.kernels.RBFKernel(
        lengthscale=args.init_lengthscale, scale=args.init_scale)
    decoder = sgpvae.networks.LinearGaussian(
        in_dim=args.latent_dim, out_dim=5,
        hidden_dims=args.decoder_dims, sigma=args.sigma)

    if args.pinference_net == 'factornet':
        encoder = sgpvae.networks.FactorNet(
            in_dim=5, out_dim=args.latent_dim,
            h_dims=args.h_dims, min_sigma=args.min_sigma, initial_sigma=.1)

    elif args.pinference_net == 'indexnet':
        encoder = sgpvae.networks.IndexNet(
            in_dim=5, out_dim=args.latent_dim,
            inter_dim=args.inter_dim, h_dims=args.h_dims,
            rho_dims=args.rho_dims, min_sigma=args.min_sigma)

    elif args.pinference_net == 'pointnet':
        encoder = sgpvae.networks.PointNet(
            out_dim=args.latent_dim, inter_dim=args.inter_dim,
            h_dims=args.h_dims, rho_dims=args.rho_dims,
            min_sigma=args.min_sigma)

    elif args.pinference_net == 'zeroimputation':
        encoder = sgpvae.networks.LinearGaussian(
            in_dim=5, out_dim=args.latent_dim,
            hidden_dims=args.h_dims, min_sigma=args.min_sigma)

    else:
        raise ValueError('{} is not a partial inference network.'.format(
            args.pinference_net))

    # Construct SGP-VAE model and choose loss function.
    if args.model == 'gpvae':
        model = sgpvae.models.GPVAE(encoder, decoder, args.latent_dim,
                                    kernel, add_jitter=args.add_jitter)
        loss_fn = sgpvae.estimators.gpvae.ds_estimator
        elbo_estimator = sgpvae.estimators.gpvae.elbo_estimator

    elif args.model == 'sgpvae':
        z_init = kmeans2(data['x_train'][0].numpy(), k=args.num_inducing,
                         minit='points')[0]
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

    else:
        raise ValueError('{} is not a model.'.format(args.model))
