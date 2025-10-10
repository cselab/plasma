#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import LBFGS


def compute_residual_loss(Ti, Te, ne, psi, params):
    res_ion_heat = Ti
    res_ele_heat = Te
    res_ele_dens = ne
    res_cur_diff = psi
    return torch.mean(res_ion_heat**2 +
                      res_ele_heat**2 +
                      res_ele_dens**2 +
                      res_cur_diff**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-plots', action='store_true', default=False, help='show plots if set')
    args = parser.parse_args()

    show_plots = args.show_plots

    seed = 2349873
    rng = np.random.default_rng(seed=seed)
    num_epochs = 500
    lr = 5e-4

    nrho = 64
    nt = 63

    tend = 1.0

    # space and time coordinates
    rho = np.linspace(0, 1, nrho, endpoint=True)
    t = np.linspace(0, tend, nt+1, endpoint=True)

    drho = rho[1] - rho[0]
    dt = t[1] - t[0]

    params = {
        'grid': {'drho': drho, 'dt': dt, 'nrho': nrho, 'nt': nt},
    }

    # unknown fields
    Ti  = torch.zeros(nrho * (nt + 1), requires_grad=True) # ion temp
    Te  = torch.zeros(nrho * (nt + 1), requires_grad=True) # electron temp
    ne  = torch.zeros(nrho * (nt + 1), requires_grad=True) # electron density
    psi = torch.zeros(nrho * (nt + 1), requires_grad=True) # poloidal flux

    # optimization

    optim = LBFGS([Ti, Te, ne, psi], lr=lr)

    epochs = list(range(num_epochs))
    losses = []
    for epoch in epochs:
        def closure():
            optim.zero_grad()
            loss = compute_residual_loss(Ti, Te, ne, psi, params)
            loss.backward()
            return loss

        optim.step(closure)
        if epoch % 100 == 0:
            l = compute_residual_loss(Ti, Te, ne, psi, params).detach().item()
            print(f"epoch {epoch:06d}, loss {l:.6e}")

        losses.append(l)


if __name__ == '__main__':
    main()
