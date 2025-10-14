#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import LBFGS


def get_circular_geometry(R, a, rho_norm, B_0, elongation_LCFS, device):
    """
    Simplified geometry; purely circular profile.
    Volume is V = 2 pi^2 R a^2 kappa

    Arguments:
        R: main radius
        a: minor radius
        rho_norm: normalized coordinate grid in [0, 1]
        B_0: Toroidal magnetic field on axis
        elongation_LCFS: 1 is for perfect circular cross section, > 1 for taller than wide cross section
        device: torch device
    """
    rho = rho_norm * a

    elongation = 1 + rho * (elongation_LCFS - 1)

    # toroidal flux
    phi = np.pi * B_0 * rho**2

    volume = 2 * np.pi**2 * R * rho**2 * elongation
    area = np.pi * rho**2 * elongation

    # \nabla V * a
    vpr = 4 * np.pi**2 * R * rho * elongation * a + volume / elongation * (elongation_LCFS - 1)

    # S' = dS/drnorm for area integrals on cell grid
    spr = 2 * np.pi * rho * elongation * a + area / elongation * (elongation_LCFS - 1)

    # Geometry variables for general geometry form of transport equations.
    # With circular geometry approximation.

    # g0: <\nabla V>
    g0 = vpr / a

    # g1: <(\nabla V)^2>
    g1 = vpr**2 / a**2

    # g2: <(\nabla V)^2 / R^2>
    g2 = g1 / R**2

    # g3: <1/R^2> (done without a elongation correction)
    g3 = 1 / (R**2 * (1 - (rho / R) ** 2) ** (3.0 / 2.0))

    return {
        'Vprime': torch.from_numpy(vpr / a).to(device),
        'g1': torch.from_numpy(g1).to(device),
        'g2': torch.from_numpy(g2).to(device),
        'g3': torch.from_numpy(g3).to(device),
    }

def compute_residual_loss(Ti, Te, ne, psi, params):
    geom = params['geometry']
    Vp = geom['Vprime']
    ni = 1.0


    # ion heat equation

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    B_0 = 1.0
    params = {
        'grid': {'drho': drho, 'dt': dt, 'nrho': nrho, 'nt': nt},
        'geometry': get_circular_geometry(R=1, a=0.1, rho_norm=rho,
                                          B_0=B_0, elongation_LCFS=1.0,
                                          device=device),
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
