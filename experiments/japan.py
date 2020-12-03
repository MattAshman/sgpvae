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
from sgpvae.utils.misc import str2bool, save

torch.set_default_dtype(torch.float64)

inputs = ['lat', 'lon', 'elev', 'day']
observations = ['PRCP', 'TMAX', 'TMIN', 'TAVG', 'SNWD']


def preprocess():
    data = load()

    df = data['all']
    train = data['train']
    train_1980 = data['train_1980']
    test_1980 = data['test_1980']
    train_1981 = data['train_1981']
    test_1981 = data['test_1981']

    # Extract data into numpy arrays and perform preprocessing.
    x_train = list(map(lambda x: x[inputs].to_numpy(), train))
    y_train = list(map(lambda x: x[observations].to_numpy(), train))

    # 1980 data.
    x_train_1980 = list(map(lambda x: x[inputs].to_numpy(), train_1980))
    y_train_1980 = list(map(lambda x: x[observations].to_numpy(), train_1980))
    x_test_1980 = list(map(lambda x: x[inputs].to_numpy(), test_1980))

    # 1981 data.
    x_train_1981 = list(map(lambda x: x[inputs].to_numpy(), train_1981))
    y_train_1981 = list(map(lambda x: x[observations].to_numpy(), train_1981))
    x_test_1981 = list(map(lambda x: x[inputs].to_numpy(), test_1981))

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


def predict(model, dataset, test, observations, y_mean, y_std):
    preds = []
    variances = []

    test_iter = tqdm.tqdm(zip(dataset.x, dataset.y, dataset.m, test),
                          desc='Evaluation')

    for x_b, y_b, m_b, df_b in test_iter:
        x_b = x_b.squeeze(0)
        x_test = x_b.clone()
        y_b = y_b.squeeze(0)
        m_b = m_b.squeeze(0)

        # Find rows where some observations are present.
        valid_idx = torch.where(m_b.sum(1) > 0)[0]
        x_b = x_b[valid_idx]
        y_b = y_b[valid_idx]
        m_b = m_b[valid_idx]
        # idx_b = df_b.index[valid_idx]

        # Test predictions.
        mean, sigma = model.predict_y(
            x=x_b, y=y_b, mask=m_b, x_test=x_test, num_samples=10)[:2]

        mean = mean.numpy() * y_std + y_mean
        sigma = sigma.numpy() * y_std

        # Convert to DataFrame and add to lists.
        pred = pd.DataFrame(mean, index=df_b.index, columns=observations)
        var = pd.DataFrame(sigma**2, index=df_b.index, columns=observations)

        preds.append(pred)
        variances.append(var)

    preds = pd.concat(preds)
    variances = pd.concat(variances)

    return preds, variances


def main(args):
    data = preprocess()

    # Set up dataset and dataloaders.
    dataset = sgpvae.utils.dataset.MetaTupleDataset(
        data['x_train'], data['y_train'], missing=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False)

    # For model evaluation.
    train_1980_dataset = sgpvae.utils.dataset.MetaTupleDataset(
        data['x_train_1980'], data['y_train_1980'], missing=True)
    train_1981_dataset = sgpvae.utils.dataset.MetaTupleDataset(
        data['x_train_1981'], data['y_train_1981'], missing=True)

    # Model construction.
    kernel = sgpvae.kernels.RBFKernel(
        lengthscale=args.init_lengthscale, scale=args.init_scale)

    # Transform GPRN weights.
    w_transform = torch.sigmoid if args.transform else None

    # Likelihood function.
    if args.likelihood == 'gprn':
        print('Using GPRN likelihood function.')
        likelihood = sgpvae.likelihoods.GPRNHomoGaussian(
            f_dim=args.f_dim, out_dim=len(observations), sigma=args.sigma,
            w_transform=w_transform)
        latent_dim = args.f_dim + args.f_dim * len(observations)

    elif args.likelihood == 'gprn-nn':
        print('Using GPRN-NN likelihood function.')
        likelihood = sgpvae.likelihoods.GPRNNNHomoGaussian(
            f_dim=args.f_dim, w_dim=args.w_dim, out_dim=len(observations),
            hidden_dims=args.decoder_dims, sigma=args.sigma,
            w_transform=w_transform)
        latent_dim = args.f_dim + args.f_dim * args.w_dim

    elif args.likelihood == 'nn-gprn':
        print('Using NN-GPRN likelihood function.')
        likelihood = sgpvae.likelihoods.NNGPRNHomoGaussian(
            f_dim=args.f_dim, w_dim=args.w_dim, out_dim=len(observations),
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.f_dim + args.w_dim * len(observations)

    elif args.likelihood == 'linear':
        print('Using linear likelihood function.')
        likelihood = sgpvae.likelihoods.AffineHomoGaussian(
            in_dim=args.latent_dim, out_dim=len(observations), sigma=args.sigma)
        latent_dim = args.latent_dim

    else:
        print('Using NN likelihood function.')
        likelihood = sgpvae.likelihoods.NNHomoGaussian(
            in_dim=args.latent_dim, out_dim=len(observations),
            hidden_dims=args.decoder_dims, sigma=args.sigma)
        latent_dim = args.latent_dim

    # Approximate likelihood function.
    if args.pinference_net == 'factornet':
        variational_dist = sgpvae.likelihoods.FactorNet(
            in_dim=len(observations), out_dim=latent_dim,
            h_dims=args.h_dims, min_sigma=args.min_sigma)

    elif args.pinference_net == 'indexnet':
        variational_dist = sgpvae.likelihoods.IndexNet(
            in_dim=len(observations), out_dim=latent_dim,
            inter_dim=args.inter_dim, h_dims=args.h_dims,
            rho_dims=args.rho_dims, min_sigma=args.min_sigma)

    elif args.pinference_net == 'pointnet':
        variational_dist = sgpvae.likelihoods.PointNet(
            out_dim=latent_dim, inter_dim=args.inter_dim,
            h_dims=args.h_dims, rho_dims=args.rho_dims,
            min_sigma=args.min_sigma)

    elif args.pinference_net == 'zeroimputation':
        variational_dist = sgpvae.likelihoods.NNHeteroGaussian(
            in_dim=len(observations), out_dim=latent_dim,
            hidden_dims=args.h_dims, min_sigma=args.min_sigma)

    else:
        raise ValueError('{} is not a partial inference network.'.format(
            args.pinference_net))

    # Construct SGP-VAE model.
    if args.model == 'gpvae':
        model = sgpvae.models.GPVAE(
            likelihood, variational_dist, latent_dim, kernel,
            add_jitter=args.add_jitter)

    elif args.model == 'sgpvae':
        z_init = kmeans2(data['x_train'][0].numpy(), k=args.num_inducing,
                         minit='points')[0]
        z_init = torch.tensor(z_init)

        model = sgpvae.models.SGPVAE(
            likelihood, variational_dist, latent_dim, kernel, z_init,
            add_jitter=args.add_jitter, fixed_inducing=args.fixed_inducing)

    elif args.model == 'vae':
        model = sgpvae.models.VAE(likelihood, variational_dist, latent_dim)

    else:
        raise ValueError('{} is not a model.'.format(args.model))

    # Model training.
    model.train(True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_iter = tqdm.tqdm(range(args.epochs), desc='Epoch', leave=True)
    for epoch in epoch_iter:
        losses = []
        batch_iter = tqdm.tqdm(iter(loader), desc='Batch', leave=False)
        for x_b, y_b, m_b, idx_b in batch_iter:
            # Get rid of 3rd-dimension.
            x_b = x_b.squeeze(0)
            y_b = y_b.squeeze(0)
            m_b = m_b.squeeze(0)

            optimiser.zero_grad()
            loss = -model.elbo(x_b, y_b, m_b, num_samples=1)
            loss.backward()
            optimiser.step()

            losses.append(loss.item())

        epoch_iter.set_postfix(loss=np.mean(losses))

        if epoch % args.cache_freq == 0:
            with torch.no_grad():
                pred_1980, var_1980 = predict(
                    model, train_1980_dataset, data['test_1980'], observations,
                    data['y_mean'], data['y_std'])
                rmse_1980 = sgpvae.utils.metric.rmse(
                    pred_1980, pd.concat(data['test_1980']))
                mll_1980 = sgpvae.utils.metric.mll(
                    pred_1980, var_1980, pd.concat(data['test_1980']))

                pred_1981, var_1981 = predict(
                    model, train_1981_dataset, data['test_1981'], observations,
                    data['y_mean'], data['y_std'])
                rmse_1981 = sgpvae.utils.metric.rmse(
                    pred_1981, pd.concat(data['test_1981']))
                mll_1981 = sgpvae.utils.metric.mll(
                    pred_1981, var_1981, pd.concat(data['test_1981']))

            tqdm.tqdm.write('\nRMSE 1980: {:.3f}'.format(rmse_1980['TAVG']))
            tqdm.tqdm.write('MLL 1980: {:.3f}'.format(mll_1980['TAVG']))
            tqdm.tqdm.write('\nRMSE 1981: {:.3f}'.format(rmse_1981['TAVG']))
            tqdm.tqdm.write('MLL 1981: {:.3f}'.format(mll_1981['TAVG']))

    # Evaluate model performance.
    with torch.no_grad():
        elbo = 0
        for (x_b, y_b, m_b, idx_b) in loader:
            # Get rid of 3rd-dimension.
            x_b = x_b.squeeze(0)
            y_b = y_b.squeeze(0)
            m_b = m_b.squeeze(0)

            elbo += model.elbo(x_b, y_b, m_b, num_samples=10)

        pred_1980, var_1980 = predict(
            model, train_1980_dataset, data['test_1980'], observations,
            data['y_mean'], data['y_std'])
        rmse_1980 = sgpvae.utils.metric.rmse(
            pred_1980, pd.concat(data['test_1980']))
        mll_1980 = sgpvae.utils.metric.mll(
            pred_1980, var_1980, pd.concat(data['test_1980']))

        pred_1981, var_1981 = predict(
            model, train_1981_dataset, data['test_1981'], observations,
            data['y_mean'], data['y_std'])
        rmse_1981 = sgpvae.utils.metric.rmse(
            pred_1981, pd.concat(data['test_1980']))
        mll_1981 = sgpvae.utils.metric.mll(
            pred_1981, var_1981, pd.concat(data['test_1980']))

    print('\nELBO: {:.3f}'.format(elbo))
    print('\nRMSE 1980: {:.3f}'.format(rmse_1980))
    print('MLL 1980: {:.3f}'.format(mll_1980))
    print('\nRMSE 1981: {:.3f}'.format(rmse_1981))
    print('MLL 1981: {:.3f}'.format(mll_1981))

    if args.save:
        metrics = {'ELBO': elbo, 'RMSE 1980': rmse_1980, 'mll_1980': mll_1980,
                   'RMSE 1981': rmse_1981, 'mll_1981': mll_1981}
        save(vars(args), metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Kernel.
    parser.add_argument('--init_lengthscale', default=1., type=float)
    parser.add_argument('--init_scale', default=1., type=float)

    # GPVAE.
    parser.add_argument('--model', default='sgpvae')
    parser.add_argument('--likelihood', default='nn', type=str)
    parser.add_argument('--pinference_net', default='indexnet', type=str)
    parser.add_argument('--latent_dim', default=3, type=int)
    parser.add_argument('--f_dim', default=3, type=int)
    parser.add_argument('--w_dim', default=3, type=int)
    parser.add_argument('--decoder_dims', default=[20, 20], nargs='+',
                        type=int)
    parser.add_argument('--sigma', default=0.1, type=float)
    parser.add_argument('--h_dims', default=[20, 20], nargs='+', type=int)
    parser.add_argument('--rho_dims', default=[20, 20], nargs='+', type=int)
    parser.add_argument('--inter_dim', default=20, type=int)
    parser.add_argument('--num_inducing', default=100, type=int)
    parser.add_argument('--fixed_inducing', default=False,
                        type=sgpvae.utils.misc.str2bool)
    parser.add_argument('--add_jitter', default=True,
                        type=sgpvae.utils.misc.str2bool)
    parser.add_argument('--min_sigma', default=1e-3, type=float)
    parser.add_argument('--initial_sigma', default=.1, type=float)
    parser.add_argument('--transform', default=False, type=str2bool)

    # Training.
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--cache_freq', default=5, type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    # General.
    parser.add_argument('--save', default=False, type=str2bool)
    parser.add_argument('--results_dir', default='./_results/japan/',
                        type=str)

    args = parser.parse_args()
    main(args)