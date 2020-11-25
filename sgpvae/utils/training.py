import copy
import torch
import numpy as np

__all__ = ['elbo_subset', 'elbo_mm']


def elbo_subset(model, x, y, mask, num_samples=1, k=1, p=0.5):
    elbo = 0
    for _ in range(k):
        mask_q = copy.deepcopy(mask)
        for n, row in enumerate(mask):
            row_q = copy.deepcopy(row)
            data_idx = torch.where(row == 1)[0]
            to_remove = np.random.binomial(len(data_idx) - 1, p=p)
            remove_idx = np.random.choice(data_idx.numpy(), size=[to_remove],
                                          replace=False)

            row_q[remove_idx] = 0
            mask_q[n, :] = row_q

        elbo += model.elbo(x, y, mask, mask_q, num_samples)

    return elbo


def elbo_mm(model, x, y, mask, num_samples, k, p=0.5):
    """Monte Carlo estimate of the evidence lower bound from the Multimodal
    VAE."""
    elbo = model.elbo(x, y, mask, num_samples=num_samples)

    elbos_n = 0
    for n in range(y.shape[1]):
        mask_q = copy.deepcopy(mask)
        mask_q[:, n] = 0
        elbos_n += model.elbo(x, y, mask, mask_q, num_samples)

    elbos_k = 0
    for _ in range(k):
        mask_q = copy.deepcopy(mask)
        for n, row in enumerate(mask):
            row_q = copy.deepcopy(row)
            data_idx = torch.where(row == 1)[0]
            to_remove = np.random.binomial(len(data_idx)-1, p=p)
            remove_idx = np.random.choice(data_idx.numpy(), size=[to_remove],
                                          replace=False)

            row_q[remove_idx] = 0
            mask_q[n, :] = row_q

        elbos_k += model.elbo(x, y, mask, mask_q, num_samples)

    elbo += elbos_n + elbos_k

    return elbo
