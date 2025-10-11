#!/usr/bin/env python3

import torch
import numpy as np
import argparse

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_residual_loss(Ti, Te, ne, psi, params):
    """
    Compute the residual loss for the ODIL solver.
    Normalized approach to avoid numerical issues.
    """
    grid = params['grid']
    nrho = grid['nrho']
    nt = grid['nt']
    dt = grid['dt']
    dx = grid['drho']
    
    # Reshape fields from flattened to 2D (rho, t)
    Ti_2d = Ti.view(nrho, nt + 1)
    Te_2d = Te.view(nrho, nt + 1)
    ne_2d = ne.view(nrho, nt + 1)
    psi_2d = psi.view(nrho, nt + 1)
    
    # Transport coefficients (normalized)
    D_i = 0.01  # Ion thermal diffusivity
    D_e = 0.01  # Electron thermal diffusivity
    D_n = 0.01  # Particle diffusivity
    D_psi = 0.01 # Resistivity
    
    loss = 0.0
    
    # Compute residuals for each time step (except initial)
    for t in range(1, nt + 1):
        # Get current and previous profiles
        Ti_curr = Ti_2d[:, t]
        Ti_prev = Ti_2d[:, t-1]
        Te_curr = Te_2d[:, t]
        Te_prev = Te_2d[:, t-1]
        ne_curr = ne_2d[:, t]
        ne_prev = ne_2d[:, t-1]
        psi_curr = psi_2d[:, t]
        psi_prev = psi_2d[:, t-1]
        
        # Compute second derivatives using finite differences
        d2Ti_dx2 = torch.zeros_like(Ti_curr)
        d2Te_dx2 = torch.zeros_like(Te_curr)
        d2ne_dx2 = torch.zeros_like(ne_curr)
        d2psi_dx2 = torch.zeros_like(psi_curr)
        
        # Interior points
        d2Ti_dx2[1:-1] = (Ti_curr[2:] - 2*Ti_curr[1:-1] + Ti_curr[:-2]) / (dx**2)
        d2Te_dx2[1:-1] = (Te_curr[2:] - 2*Te_curr[1:-1] + Te_curr[:-2]) / (dx**2)
        d2ne_dx2[1:-1] = (ne_curr[2:] - 2*ne_curr[1:-1] + ne_curr[:-2]) / (dx**2)
        d2psi_dx2[1:-1] = (psi_curr[2:] - 2*psi_curr[1:-1] + psi_curr[:-2]) / (dx**2)
        
        # Boundary points
        d2Ti_dx2[0] = (Ti_curr[1] - 2*Ti_curr[0] + 0) / (dx**2)  # u(-1) = 0
        d2Te_dx2[0] = (Te_curr[1] - 2*Te_curr[0] + 0) / (dx**2)
        d2ne_dx2[0] = (ne_curr[1] - 2*ne_curr[0] + 0) / (dx**2)
        d2psi_dx2[0] = (psi_curr[1] - 2*psi_curr[0] + 0) / (dx**2)
        
        d2Ti_dx2[-1] = (0 - 2*Ti_curr[-1] + Ti_curr[-2]) / (dx**2)  # u(nx) = 0
        d2Te_dx2[-1] = (0 - 2*Te_curr[-1] + Te_curr[-2]) / (dx**2)
        d2ne_dx2[-1] = (0 - 2*ne_curr[-1] + ne_curr[-2]) / (dx**2)
        d2psi_dx2[-1] = (0 - 2*psi_curr[-1] + psi_curr[-2]) / (dx**2)
        
        # PDE residuals: ∂u/∂t - D ∂²u/∂x² = 0
        res_Ti = (Ti_curr - Ti_prev) / dt - D_i * d2Ti_dx2
        res_Te = (Te_curr - Te_prev) / dt - D_e * d2Te_dx2
        res_ne = (ne_curr - ne_prev) / dt - D_n * d2ne_dx2
        res_psi = (psi_curr - psi_prev) / dt - D_psi * d2psi_dx2
        
        # Boundary condition residuals
        bc_residual_Ti_0 = Ti_curr[0]  # u(0,t) = 0
        bc_residual_Ti_1 = Ti_curr[-1]  # u(1,t) = 0
        bc_residual_Te_0 = Te_curr[0]
        bc_residual_Te_1 = Te_curr[-1]
        bc_residual_ne_0 = ne_curr[0]
        bc_residual_ne_1 = ne_curr[-1]
        bc_residual_psi_0 = psi_curr[0]
        bc_residual_psi_1 = psi_curr[-1]
        
        # Total loss
        loss += torch.mean(res_Ti[1:-1]**2)
        loss += torch.mean(res_Te[1:-1]**2)
        loss += torch.mean(res_ne[1:-1]**2)
        loss += torch.mean(res_psi[1:-1]**2)
        
        # Boundary conditions
        loss += 100 * (bc_residual_Ti_0**2 + bc_residual_Ti_1**2)
        loss += 100 * (bc_residual_Te_0**2 + bc_residual_Te_1**2)
        loss += 100 * (bc_residual_ne_0**2 + bc_residual_ne_1**2)
        loss += 100 * (bc_residual_psi_0**2 + bc_residual_psi_1**2)
    
    # Initial conditions
    if 'initial_profiles' in params:
        Ti_init = params['initial_profiles']['Ti']
        Te_init = params['initial_profiles']['Te']
        ne_init = params['initial_profiles']['ne']
        psi_init = params['initial_profiles']['psi']
        
        ic_residual_Ti = Ti_2d[:, 0] - Ti_init
        ic_residual_Te = Te_2d[:, 0] - Te_init
        ic_residual_ne = ne_2d[:, 0] - ne_init
        ic_residual_psi = psi_2d[:, 0] - psi_init
        
        loss += 100 * torch.mean(ic_residual_Ti**2)
        loss += 100 * torch.mean(ic_residual_Te**2)
        loss += 100 * torch.mean(ic_residual_ne**2)
        loss += 100 * torch.mean(ic_residual_psi**2)
    
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-plots', action='store_true', default=False, help='show plots if set')
    args = parser.parse_args()

    show_plots = args.show_plots

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grid parameters
    nrho = 16
    nt = 10
    dx = 1.0 / nrho
    dt = 0.01

    # Create grid
    rho = np.linspace(0, 1, nrho, endpoint=True)
    
    # Initial profiles (normalized, zero at boundaries)
    Ti_init = (1 - rho**2)  # Normalized ion temperature
    Te_init = (1 - rho**2)  # Normalized electron temperature
    ne_init = (1 - rho**2)  # Normalized electron density
    psi_init = (1 - rho**2)  # Normalized poloidal flux
    
    # Convert to torch tensors
    Ti_init = torch.from_numpy(Ti_init).to(device)
    Te_init = torch.from_numpy(Te_init).to(device)
    ne_init = torch.from_numpy(ne_init).to(device)
    psi_init = torch.from_numpy(psi_init).to(device)
    
    # Create full solution arrays
    Ti = Ti_init.unsqueeze(0).repeat(nt + 1, 1).T  # Shape: (nrho, nt+1)
    Te = Te_init.unsqueeze(0).repeat(nt + 1, 1).T
    ne = ne_init.unsqueeze(0).repeat(nt + 1, 1).T
    psi = psi_init.unsqueeze(0).repeat(nt + 1, 1).T
    
    # Flatten for optimization
    Ti = Ti.flatten().clone().detach().requires_grad_(True)
    Te = Te.flatten().clone().detach().requires_grad_(True)
    ne = ne.flatten().clone().detach().requires_grad_(True)
    psi = psi.flatten().clone().detach().requires_grad_(True)
    
    # Parameters
    params = {
        'grid': {'drho': dx, 'dt': dt, 'nrho': nrho, 'nt': nt},
        'initial_profiles': {
            'Ti': Ti_init,
            'Te': Te_init,
            'ne': ne_init,
            'psi': psi_init
        }
    }
    
    # Optimization
    optimizer = torch.optim.Adam([Ti, Te, ne, psi], lr=0.01)
    
    num_epochs = 100
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = compute_residual_loss(Ti, Te, ne, psi, params)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            l = loss.detach().item()
            print(f"epoch {epoch:06d}, loss {l:.6e}")
        
        losses.append(loss.detach().item())
    
    # Print final statistics
    Ti_final = Ti.detach().view(nrho, nt + 1)
    Te_final = Te.detach().view(nrho, nt + 1)
    ne_final = ne.detach().view(nrho, nt + 1)
    psi_final = psi.detach().view(nrho, nt + 1)
    
    print(f"\nFinal loss: {losses[-1]:.6e}")
    print(f"Ti range: {Ti_final.min().item():.3f} - {Ti_final.max().item():.3f}")
    print(f"Te range: {Te_final.min().item():.3f} - {Te_final.max().item():.3f}")
    print(f"ne range: {ne_final.min().item():.3f} - {ne_final.max().item():.3f}")
    print(f"psi range: {psi_final.min().item():.3f} - {psi_final.max().item():.3f}")
    
    # Plot results
    if show_plots and HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 8))
        
        # Create time and space grids
        t_grid = np.linspace(0, nt * dt, nt + 1)
        rho_grid = rho
        
        # Plot 1: Ion temperature
        plt.subplot(2, 3, 1)
        plt.contourf(t_grid, rho_grid, Ti_final.cpu().numpy(), levels=20)
        plt.colorbar(label='Ti (normalized)')
        plt.xlabel('Time')
        plt.ylabel('rho')
        plt.title('Ion Temperature')
        
        # Plot 2: Electron temperature
        plt.subplot(2, 3, 2)
        plt.contourf(t_grid, rho_grid, Te_final.cpu().numpy(), levels=20)
        plt.colorbar(label='Te (normalized)')
        plt.xlabel('Time')
        plt.ylabel('rho')
        plt.title('Electron Temperature')
        
        # Plot 3: Electron density
        plt.subplot(2, 3, 3)
        plt.contourf(t_grid, rho_grid, ne_final.cpu().numpy(), levels=20)
        plt.colorbar(label='ne (normalized)')
        plt.xlabel('Time')
        plt.ylabel('rho')
        plt.title('Electron Density')
        
        # Plot 4: Poloidal flux
        plt.subplot(2, 3, 4)
        plt.contourf(t_grid, rho_grid, psi_final.cpu().numpy(), levels=20)
        plt.colorbar(label='psi (normalized)')
        plt.xlabel('Time')
        plt.ylabel('rho')
        plt.title('Poloidal Flux')
        
        # Plot 5: Loss evolution
        plt.subplot(2, 3, 5)
        plt.semilogy(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        # Plot 6: Final profiles at different times
        plt.subplot(2, 3, 6)
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt]
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        for i, t_idx in enumerate(time_indices):
            plt.plot(rho_grid, Ti_final[:, t_idx].cpu().numpy(), 
                    color=colors[i], label=f't={t_grid[t_idx]:.2f}')
        plt.xlabel('rho')
        plt.ylabel('Ti (normalized)')
        plt.title('Ti Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('odil_normalized_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    elif show_plots and not HAS_MATPLOTLIB:
        print("Matplotlib not available. Cannot show plots.")


if __name__ == '__main__':
    main()
