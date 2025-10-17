import numpy as np
import matplotlib.pyplot as plt
import mmap
import sys

n = 25
dx = 1 / n
cell_centers = (np.arange(n) + 0.5) * dx
rc = 1 + 4 * n
sz = np.dtype(np.float64).itemsize
with open(sys.argv[1], "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    nt = len(mm) // (rc * sz)
    t_out = np.ndarray(shape=(nt, ),
                       dtype=np.float64,
                       buffer=mm,
                       offset=0,
                       strides=(rc * sz, ))
    states = np.ndarray(shape=(nt, 4, n),
                        dtype=np.float64,
                        buffer=mm,
                        offset=sz,
                        strides=(rc * sz, n * sz, sz))
    for j, var_name in enumerate(("T_i", "T_e", "psi", "n_e")):
        var = states[:, j]
        lo, hi = np.min(var), np.max(var)
        plt.figure()
        plt.title(var_name)
        plt.axis([None, None, lo, hi])
        for idx in [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]:
            plt.plot(cell_centers, var[idx], "o-", label=f"t={t_out[idx]:.3e}s")
        plt.legend()
        plt.xlabel("rho_norm")
        plt.ylabel(var_name)
        plt.savefig(f"{var_name}.png")
        plt.show()
