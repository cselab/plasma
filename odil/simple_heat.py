#!/usr/bin/env python3

import torch
import numpy as np
from torch import sparse_coo_tensor

def create_sparse_laplacian_matrix(n, dx, device):
    """
    Create a sparse Laplacian matrix for finite difference second derivative.
    This is much more memory efficient than dense matrices.
    """
    # Create sparse matrix for second derivative: d²u/dx²
    # Interior points: (u[i+1] - 2*u[i] + u[i-1]) / dx²
    
    # Indices for sparse matrix
    indices = []
    values = []
    
    # Interior points (central differences)
    for i in range(1, n-1):
        # u[i-1] term
        indices.append([i, i-1])
        values.append(1.0 / (dx**2))
        
        # u[i] term
        indices.append([i, i])
        values.append(-2.0 / (dx**2))
        
        # u[i+1] term
        indices.append([i, i+1])
        values.append(1.0 / (dx**2))
    
    # Boundary points
    # At i=0: u[0] = 0 (Dirichlet), so we use forward difference
    # d²u/dx²[0] = (u[1] - 2*u[0] + 0) / dx²
    indices.append([0, 0])
    values.append(-2.0 / (dx**2))
    indices.append([0, 1])
    values.append(1.0 / (dx**2))
    
    # At i=n-1: u[n-1] = 0 (Dirichlet), so we use backward difference
    # d²u/dx²[n-1] = (0 - 2*u[n-1] + u[n-2]) / dx²
    indices.append([n-1, n-1])
    values.append(-2.0 / (dx**2))
    indices.append([n-1, n-2])
    values.append(1.0 / (dx**2))
    
    # Convert to tensors
    indices = torch.tensor(indices).T.to(device)
    values = torch.tensor(values, dtype=torch.float32).to(device)
    
    # Create sparse matrix
    sparse_matrix = sparse_coo_tensor(indices, values, size=(n, n), device=device)
    
    return sparse_matrix


def simple_heat_equation_odil():
    """
    Simple 1D heat equation with ODIL approach using sparse matrices: ∂u/∂t = D ∂²u/∂x²
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grid parameters
    nx = 32  # Larger grid to demonstrate sparse matrix benefits
    nt = 15
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
    
    # Create sparse Laplacian matrix once
    sparse_laplacian = create_sparse_laplacian_matrix(nx, dx, device)
    
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
            
            # Compute second derivative using sparse matrix multiplication
            # This is much more efficient than manual finite differences
            d2u_dx2 = torch.sparse.mm(sparse_laplacian, u_curr.unsqueeze(1)).squeeze(1)
            
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
    print(f"Grid size: {nx} points (sparse matrix benefits for larger grids)")
    
    return u.detach()

if __name__ == "__main__":
    u = simple_heat_equation_odil()
