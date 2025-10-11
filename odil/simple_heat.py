#!/usr/bin/env python3

import torch
import numpy as np

def simple_heat_equation_odil():
    """
    Simple 1D heat equation with ODIL approach: ∂u/∂t = D ∂²u/∂x²
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grid parameters
    nx = 16
    nt = 10
    dx = 1.0 / nx
    dt = 0.01
    D = 0.01  # Diffusion coefficient
    
    # Create grid
    x = torch.linspace(0, 1, nx, device=device)
    
    # Initial condition: u(x,0) = sin(πx)
    u_init = torch.sin(np.pi * x)
    
    # Create full solution array (nx × nt)
    u = u_init.unsqueeze(0).repeat(nt, 1).T  # Shape: (nx, nt)
    u = u.clone().detach().requires_grad_(True)
    
    # Optimization
    optimizer = torch.optim.Adam([u], lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Compute residuals for each time step
        loss = 0.0
        
        for t in range(1, nt):
            # Get current and previous solutions
            u_curr = u[:, t]
            u_prev = u[:, t-1]
            
            # Compute second derivative using finite differences
            d2u_dx2 = torch.zeros_like(u_curr)
            
            # Interior points
            d2u_dx2[1:-1] = (u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]) / (dx**2)
            
            # Boundary conditions: u(0) = u(1) = 0 (Dirichlet)
            d2u_dx2[0] = (u_curr[1] - 2*u_curr[0] + 0) / (dx**2)  # u(-1) = 0
            d2u_dx2[-1] = (0 - 2*u_curr[-1] + u_curr[-2]) / (dx**2)  # u(nx) = 0
            
            # Heat equation residual: ∂u/∂t - D ∂²u/∂x² = 0
            residual = (u_curr - u_prev) / dt - D * d2u_dx2
            
            # Boundary condition residuals
            bc_residual_0 = u_curr[0]  # u(0,t) = 0
            bc_residual_1 = u_curr[-1]  # u(1,t) = 0
            
            # Total loss
            loss += torch.mean(residual[1:-1]**2) + 100 * (bc_residual_0**2 + bc_residual_1**2)
        
        # Initial condition residual
        ic_residual = u[:, 0] - u_init
        loss += 100 * torch.mean(ic_residual**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    print(f"Final loss: {loss.item():.6f}")
    print(f"u range: {u.min().item():.3f} - {u.max().item():.3f}")
    
    return u.detach()

if __name__ == "__main__":
    u = simple_heat_equation_odil()
