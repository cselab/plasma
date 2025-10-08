import copy
import jax
import torax
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from typing import Any

jax.config.update('jax_enable_x64', True)
_figure_counter = 1


def get_figure_filename():
    global _figure_counter
    filename = f"{_figure_counter:03d}.png"
    _figure_counter += 1
    return filename


_NBI_W_TO_MA = 1 / 16e6
W_to_Ne_ratio = 0
nbi_times = np.array([0, 99, 100])
nbi_powers = np.array([0, 0, 33e6])
nbi_cd = nbi_powers * _NBI_W_TO_MA
r_nbi = 0.25
w_nbi = 0.25
el_heat_fraction = 0.66
eccd_power = {0: 0, 99: 0, 100: 20.0e6}
CONFIG = {
    'plasma_composition': {
        'main_ion': {
            'D': 0.5,
            'T': 0.5
        },
        'impurity': {
            'Ne': 1 - W_to_Ne_ratio,
            'W': W_to_Ne_ratio
        },
        'Z_eff': {
            0.0: {
                0.0: 2.0,
                1.0: 2.0
            }
        },
    },
    'profile_conditions': {
        'Ip': {
            0: 3e6,
            100: 12.5e6
        },
        'T_i': {
            0.0: {
                0.0: 6.0,
                1.0: 0.2
            }
        },
        'T_i_right_bc': 0.2,
        'T_e': {
            0.0: {
                0.0: 6.0,
                1.0: 0.2
            }
        },
        'T_e_right_bc': 0.2,
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': {
            0: 0.35,
            100: 0.35
        },
        'nbar': 0.85,
        'n_e': {
            0: {
                0.0: 1.3,
                1.0: 1.0
            }
        },
        'normalize_n_e_to_nbar': True,
        'n_e_nbar_is_fGW': True,
        'initial_psi_from_j': True,
        'initial_j_is_total_current': True,
        'current_profile_nu': 2,
    },
    'numerics': {
        't_final': 150,
        'fixed_dt': 1,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'R_major': 6.2,
        'a_minor': 2.0,
        'B_0': 5.3,
    },
    'sources': {
        'ecrh': {
            'gaussian_width': 0.05,
            'gaussian_location': 0.35,
            'P_total': eccd_power,
        },
        'generic_heat': {
            'gaussian_location': r_nbi,
            'gaussian_width': w_nbi,
            'P_total': (nbi_times, nbi_powers),
            'electron_heat_fraction': el_heat_fraction,
        },
        'generic_current': {
            'use_absolute_current': True,
            'gaussian_width': w_nbi,
            'gaussian_location': r_nbi,
            'I_generic': (nbi_times, nbi_cd),
        },
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
        'cyclotron_radiation': {},
        'impurity_radiation': {
            'model_name': 'mavrin_fit',
            'radiation_multiplier': 0.0,
        },
    },
    'neoclassical': {
        'bootstrap_current': {
            'bootstrap_multiplier': 1.0,
        },
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'T_i_ped': {
            0: 0.5,
            100: 0.5,
            105: 3.0
        },
        'T_e_ped': {
            0: 0.5,
            100: 0.5,
            105: 3.0
        },
        'n_e_ped_is_fGW': True,
        'n_e_ped': 0.85,
        'rho_norm_ped_top': 0.95,
    },
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': True,
        'D_e_inner': 0.15,
        'V_e_inner': 0.0,
        'chi_i_inner': 0.3,
        'chi_e_inner': 0.3,
        'rho_inner': 0.1,
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.95,
        'chi_min': 0.05,
        'chi_max': 100,
        'D_e_min': 0.05,
        'D_e_max': 50,
        'V_e_min': -10,
        'V_e_max': 10,
        'smoothing_width': 0.1,
        'DV_effective': True,
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
        'avoid_big_negative_s': False,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': True,
        'n_corrector_steps': 10,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}


def detailed_plot_single_sim(dt: xr.DataTree, time: float | None = None):
    spr = dt.profiles.spr.sel(rho_norm=dt.rho_cell_norm)
    drho_norm = dt.scalars.drho_norm
    j_ohmic = dt.profiles.j_ohmic
    I_ohm = np.sum(spr * j_ohmic * drho_norm, axis=1)
    jnbi = dt.profiles.j_generic_current
    I_nbi = np.sum(spr * jnbi * drho_norm, axis=1)
    te_line_avg = np.mean(dt.profiles.T_e, axis=1)
    ti_line_avg = np.mean(dt.profiles.T_i, axis=1)
    if time is None:
        time_index = -1
    else:
        time_index = np.argmin(np.abs(dt.time.values - time))
    _, axes = plt.subplots(4, 5, figsize=(24, 12))
    fsize = 13
    fontreduction = 1
    plt.rcParams.update({'font.size': fsize})
    axes[0, 0].plot(dt.time,
                    dt.profiles.Ip_profile[:, -1] / 1e6,
                    'b-',
                    label=r'$I_p$')
    axes[0, 0].set_xlabel(r"Time [s]")
    axes[0, 0].set_ylabel(r"Plasma current [MA]")
    axes[0, 0].legend(fontsize=fsize - fontreduction)
    axes[0, 1].plot(dt.time,
                    dt.scalars.I_bootstrap / 1e6,
                    'b-',
                    label=r'$I_{bootstrap}$')
    axes[0, 1].plot(dt.time,
                    dt.scalars.I_ecrh / 1e6,
                    'r-',
                    label=r'$I_{ecrh}$')
    axes[0, 1].plot(dt.time, I_ohm / 1e6, 'm-', label=r'$I_{ohmic}$')
    axes[0, 1].plot(dt.time, I_nbi / 1e6, 'k-', label=r'$I_{nbi}$')
    axes[0, 1].set_xlabel(r"Time [s]")
    axes[0, 1].set_ylabel(r"Current [MA]")
    axes[0, 1].legend(fontsize=fsize - fontreduction)
    axes[0, 1].set_title(r"Total currents", fontsize=fsize - fontreduction)
    axes[0, 2].plot(dt.time[10:], dt.scalars.Q_fusion[10:], 'r-')
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel(r"Q")
    axes[0, 2].set_title(r"Fusion Q", fontsize=fsize - fontreduction)
    axes[0, 3].plot(dt.time[10:], dt.scalars.H20[10:], 'r-')
    axes[0, 3].set_xlabel("Time [s]")
    axes[0, 3].set_ylabel(r"H20")
    axes[0, 3].set_title(r"H20 confinement factor",
                         fontsize=fsize - fontreduction)
    axes[0, 4].plot(dt.time, te_line_avg, 'r-', label=r'$\langle T_e \rangle$')
    axes[0, 4].plot(dt.time, ti_line_avg, 'b-', label=r'$\langle T_i \rangle$')
    axes[0, 4].set_xlabel("Time [s]")
    axes[0, 4].set_ylabel(r"Temperature [keV]")
    axes[0, 4].set_title(r"Line averaged temperatures",
                         fontsize=fsize - fontreduction)
    axes[0, 4].legend(fontsize=fsize - fontreduction)
    axes[1, 0].plot(dt.time, dt.scalars.v_loop_lcfs, 'r-')
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel(r"$V_{loop}(LCFS)$")
    axes[1, 0].set_title(r"Loop voltage (at LCFS)",
                         fontsize=fsize - fontreduction)
    axes[1, 1].plot(dt.time,
                    dt.scalars.P_ecrh_e / 1e6,
                    'b-',
                    label=r'$P_{ECRH}$')
    axes[1, 1].plot(dt.time,
                    dt.scalars.P_aux_generic_total / 1e6,
                    'r-',
                    label=r'$P_{NBI}$')
    axes[1, 1].plot(dt.time,
                    dt.scalars.P_ohmic_e / 1e6,
                    'm-',
                    label=r'$P_{ohmic}$')
    axes[1, 1].plot(dt.time,
                    dt.scalars.P_alpha_total / 1e6 / 5,
                    'k-',
                    label=r'$P_{fusion}/5$')
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel(r"Heating powers $[MW]$")
    axes[1, 1].legend(fontsize=fsize - fontreduction)
    axes[1, 1].set_title(r"Total heating powers",
                         fontsize=fsize - fontreduction)
    axes[1, 2].plot(dt.time,
                    dt.scalars.P_cyclotron_e / 1e6,
                    'b-',
                    label=r'$P_{cyclotron}$')
    axes[1, 2].plot(dt.time,
                    dt.scalars.P_radiation_e / 1e6,
                    'r-',
                    label=r'$P_{radiation}+P_{brems}$')
    axes[1, 2].set_xlabel("Time [s]")
    axes[1, 2].set_ylabel(r"Heating sinks $[MW]$")
    axes[1, 2].legend(fontsize=fsize - fontreduction)
    axes[1, 2].set_title(r"Total sinks", fontsize=fsize - fontreduction)
    axes[1, 3].plot(dt.time, dt.scalars.q_min, 'b-', label=r'$q_{min}$')
    axes[1, 3].plot(dt.time, dt.profiles.q[:, 0], 'r-', label=r'$q_{0}$')
    axes[1, 3].plot(dt.time, dt.scalars.q95, 'm-', label=r'$q_{95}$')
    axes[1, 3].set_xlabel("Time [s]")
    axes[1, 3].set_ylabel(r"$q$")
    axes[1, 3].legend(fontsize=fsize - fontreduction)
    axes[1, 3].set_title(r"Safety factor (q) at various rho",
                         fontsize=fsize - fontreduction)
    axes[1, 4].plot(dt.time, dt.scalars.li3, 'r-')
    axes[1, 4].set_xlabel("Time [s]")
    axes[1, 4].set_ylabel(r"li(3)")
    axes[1, 4].set_title(r"Normalized internal inductance",
                         fontsize=fsize - fontreduction)
    axes[2, 0].plot(dt.rho_face_norm,
                    dt.profiles.chi_turb_e[time_index, :],
                    'b-',
                    label=r'$\chi_e$')
    axes[2, 0].plot(dt.rho_face_norm,
                    dt.profiles.chi_turb_i[time_index, :],
                    'r-',
                    label=r'$\chi_i$')
    axes[2, 0].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[2, 0].set_ylabel(r"Heat conductivity $[m^2/s]")
    axes[2, 0].legend(fontsize=fsize - fontreduction)
    axes[2, 0].set_title(r"Heat transport coefficients",
                         fontsize=fsize - fontreduction)
    axes[2, 1].plot(dt.rho_norm,
                    dt.profiles.T_e[time_index, :],
                    'b-',
                    label=r'$T_e$')
    axes[2, 1].plot(dt.rho_norm,
                    dt.profiles.T_i[time_index, :],
                    'r-',
                    label=r'$T_i$')
    axes[2, 1].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[2, 1].set_ylabel(r"Temperature [keV]")
    axes[2, 1].legend(fontsize=fsize - fontreduction)
    axes[2, 1].set_title(r"Temperature profiles",
                         fontsize=fsize - fontreduction)
    axes[2, 2].plot(dt.rho_norm,
                    dt.profiles.n_e[time_index, :],
                    'b-',
                    label=r'$n_e$')
    axes[2, 2].plot(dt.rho_norm,
                    dt.profiles.n_i[time_index, :],
                    'b--',
                    label=r'$n_i$')
    axes[2, 2].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[2, 2].set_ylabel(r"Density [$10^{20} m^{-3}$]")
    axes[2, 2].legend(fontsize=fsize - fontreduction)
    axes[2, 2].set_title(r"$n_e$, $n_i$", fontsize=fsize - fontreduction)
    axes[2, 3].plot(dt.rho_face_norm,
                    dt.profiles.q[time_index, :],
                    'b-',
                    label='q')
    axes[2, 3].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[2, 3].set_ylabel(r"q")
    axes[2, 3].legend(fontsize=fsize - fontreduction)
    axes[2, 3].set_title(r"Safety Factor ($q$)",
                         fontsize=fsize - fontreduction)
    axes[2, 4].plot(dt.rho_face_norm,
                    dt.profiles.magnetic_shear[time_index, :],
                    'b-',
                    label=r'$\hat{s}$')
    axes[2, 4].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[2, 4].set_ylabel(r"$\hat{s}$")
    axes[2, 4].legend(fontsize=fsize - fontreduction)
    axes[2, 4].set_title(r"Magnetic shear ($\hat{s}$)",
                         fontsize=fsize - fontreduction)
    psidot = dt.profiles.v_loop[time_index, :]
    ymin = min(min(psidot), 0) * 1.2
    ymax = max(max(psidot), 0) * 1.2
    axes[3, 0].plot(dt.rho_norm, psidot, 'b-')
    axes[3, 0].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 0].set_ylabel(r"Vloop [V]")
    axes[3, 0].set_ylim([ymin, ymax])
    axes[3, 0].set_title(r"Loop voltage profile",
                         fontsize=fsize - fontreduction)
    axes[3, 1].plot(dt.rho_norm,
                    dt.profiles.j_total[time_index, :] / 1e6,
                    'b-',
                    label=r'$j_{total}$')
    axes[3, 1].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 1].set_ylabel(r"Currents $[MA/m^2]$")
    axes[3, 1].legend(fontsize=fsize - fontreduction)
    axes[3, 1].set_title(r"Total current density",
                         fontsize=fsize - fontreduction)
    axes[3, 2].plot(dt.rho_cell_norm,
                    dt.profiles.j_ecrh[time_index, :] / 1e6,
                    'b-',
                    label=r'$j_{ecrh}$')
    axes[3, 2].plot(dt.rho_cell_norm,
                    dt.profiles.j_generic_current[time_index, :] / 1e6,
                    'r-',
                    label=r'$j_{nbi}$')
    axes[3, 2].plot(dt.rho_cell_norm,
                    dt.profiles.j_ohmic[time_index, :] / 1e6,
                    'm-',
                    label=r'$j_{ohmic}$')
    axes[3, 2].plot(dt.rho_norm,
                    dt.profiles.j_bootstrap[time_index, :] / 1e6,
                    'g-',
                    label=r'$j_{bootsrap}$')
    axes[3, 2].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 2].set_ylabel(r"Currents $[MA/m^2]$")
    axes[3, 2].legend(fontsize=fsize - fontreduction)
    axes[3, 2].set_title(r"Current source densities",
                         fontsize=fsize - fontreduction)
    axes[3, 3].plot(dt.rho_cell_norm,
                    dt.profiles.p_impurity_radiation_e[time_index, :] / 1e6,
                    'b-',
                    label=r'$P_{rad}$')
    axes[3, 3].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 3].set_ylabel(r"Heat sink density $[MW/m^3]$")
    axes[3, 3].legend(fontsize=fsize - fontreduction)
    axes[3, 3].set_title(r"Radiation heat sink",
                         fontsize=fsize - fontreduction)
    axes[3, 4].plot(dt.rho_cell_norm,
                    dt.profiles.p_ecrh_e[time_index, :] / 1e6,
                    'b-',
                    label=r'$Q_{ecrh}$')
    axes[3, 4].plot(dt.rho_cell_norm,
                    dt.profiles.p_generic_heat_i[time_index, :] / 1e6,
                    'r-',
                    label=r'$Q_{nbi_i}$')
    axes[3, 4].plot(dt.rho_cell_norm,
                    dt.profiles.p_generic_heat_e[time_index, :] / 1e6,
                    'm-',
                    label=r'$Q_{nbi_e}$')
    axes[3, 4].plot(dt.rho_cell_norm,
                    dt.profiles.p_alpha_i[time_index, :] / 1e6,
                    'g-',
                    label=r'$Q_{fus_i}$')
    axes[3, 4].plot(dt.rho_cell_norm,
                    dt.profiles.p_alpha_e[time_index, :] / 1e6,
                    'k-',
                    label=r'$Q_{fus_e}$')
    axes[3, 4].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[3, 4].set_ylabel(r"Heat source densities $[MW/m^3]$")
    axes[3, 4].legend(fontsize=fsize - fontreduction)
    axes[3, 4].set_title(r"Heat sources", fontsize=fsize - fontreduction)
    plt.tight_layout()
    plt.savefig(get_figure_filename())
    plt.close()


import seaborn as sns


def compare_timetraces(datatrees: list[xr.DataTree],
                       labels: list[str] | None = None):
    if labels is not None and len(datatrees) != len(labels):
        raise ValueError(
            "The number of labels must match the number of datatrees.")
    _, axes = plt.subplots(2, 4, figsize=(27, 9))
    fsize = 13
    fontreduction = 1
    plt.rcParams.update({'font.size': fsize})
    colors = sns.color_palette("Set1", n_colors=len(datatrees))
    for i, dt in enumerate(datatrees):
        spr = dt.profiles.spr.sel(rho_norm=dt.rho_cell_norm)
        drho_norm = dt.scalars.drho_norm
        j_ohmic = dt.profiles.j_ohmic
        I_ohm = np.sum(spr * j_ohmic * drho_norm, axis=1)
        color = colors[i]
        label = labels[i] if labels is not None else f"Sim {i+1}"
        axes[0, 0].plot(dt.time,
                        dt.profiles.Ip_profile[:, -1] / 1e6,
                        color=color,
                        label=label)
        axes[0, 1].plot(dt.time,
                        dt.scalars.q_min,
                        color=color,
                        linestyle='--',
                        label=f"{label} (q_min)")
        axes[0, 1].plot(dt.time,
                        dt.scalars.q95,
                        color=color,
                        linestyle='-',
                        label=f"{label} (q_95)")
        axes[0, 2].plot(dt.time, dt.scalars.li3, color=color, label=label)
        axes[0, 3].plot(dt.time,
                        dt.scalars.v_loop_lcfs,
                        color=color,
                        label=label)
        axes[1, 0].plot(dt.time[10:],
                        dt.scalars.Q_fusion[10:],
                        color=color,
                        label=label)
        te_line_avg = np.mean(dt.profiles.T_e, axis=1)
        ti_line_avg = np.mean(dt.profiles.T_i, axis=1)
        axes[1, 1].plot(dt.time,
                        te_line_avg,
                        color=color,
                        linestyle='--',
                        label=f"{label}" + r" $\langle T_e \rangle$")
        axes[1, 1].plot(dt.time,
                        ti_line_avg,
                        color=color,
                        linestyle='-',
                        label=f"{label}" + r" $\langle T_i \rangle$")
        P_ext = dt.scalars.P_aux_total
        axes[1, 2].plot(dt.time, P_ext / 1e6, color=color, label=label)
        axes[1, 3].plot(dt.time,
                        1 - I_ohm / dt.profiles.Ip_profile[:, -1],
                        color=color,
                        label=label)
    axes[0, 0].set_xlabel(r"Time [s]")
    axes[0, 0].set_ylabel(r"Plasma current [MA]")
    axes[0, 0].set_title(r"$I_p$")
    axes[0, 0].legend(fontsize=fsize - fontreduction)
    axes[0, 1].set_xlabel(r"Time [s]")
    axes[0, 1].set_ylabel(r"q")
    axes[0, 1].set_title(r"$q_{95}$ and $q_{min}$")
    axes[0, 1].legend(fontsize=fsize - fontreduction)
    axes[0, 2].set_xlabel(r"Time [s]")
    axes[0, 2].set_ylabel(r"li(3)")
    axes[0, 2].set_title(r"Normalized internal inductance")
    axes[0, 2].legend(fontsize=fsize - fontreduction)
    axes[0, 3].set_xlabel(r"Time [s]")
    axes[0, 3].set_ylabel(r"Vloop_LCFS")
    axes[0, 3].set_title(r"Vloop_LCFS")
    axes[0, 3].legend(fontsize=fsize - fontreduction)
    axes[1, 0].set_xlabel(r"Time [s]")
    axes[1, 0].set_ylabel(r"Q")
    axes[1, 0].set_title(r"Fusion Q")
    axes[1, 0].legend(fontsize=fsize - fontreduction)
    axes[1, 1].set_xlabel(r"Time [s]")
    axes[1, 1].set_ylabel(r"Temperature [keV]")
    axes[1, 1].set_title(r"Line averaged temperatures")
    axes[1, 1].legend(fontsize=fsize - fontreduction)
    axes[1, 2].set_xlabel("Time [s]")
    axes[1, 2].set_ylabel(r"Heating powers $[MW]$")
    axes[1, 2].set_title(r"Total External Heating Power")
    axes[1, 2].legend(fontsize=fsize - fontreduction)
    axes[1, 3].set_xlabel("Time [s]")
    axes[1, 3].set_ylabel(r"$f_{n_i}$")
    axes[1, 3].set_title(r"Non-inductive current fraction")
    axes[1, 3].legend(fontsize=fsize - fontreduction)
    plt.tight_layout()
    plt.savefig(get_figure_filename())
    plt.close()


import seaborn as sns


def compare_profiles(datatrees: list[xr.DataTree],
                     times: list[float],
                     labels: list[str] | None = None):
    if len(times) != len(datatrees):
        raise ValueError(
            "The number of times must match the number of datatrees.")
    if labels is not None and len(labels) != len(datatrees):
        raise ValueError(
            "The number of labels must match the number of datatrees.")
    num_sims = len(datatrees)
    _, axes = plt.subplots(2, 3, figsize=(18, 8))
    fsize = 13
    fontreduction = 1
    plt.rcParams.update({'font.size': fsize})
    colors = sns.color_palette("Set1", n_colors=len(datatrees))
    qmax = 0
    for i, (dt, time) in enumerate(zip(datatrees, times)):
        if labels is not None:
            label = labels[i] + f': t={time:.2f}'
        else:
            label = f"Sim{i+1}: t={time:.2f}"
        color = colors[i]
        time_index = np.argmin(np.abs(dt.time.values - time))
        axes[0, 0].plot(dt.rho_norm,
                        dt.profiles.v_loop.isel(time=time_index),
                        color=color,
                        label=label)
        axes[0, 1].plot(dt.rho_norm,
                        dt.profiles.T_e.isel(time=time_index),
                        color=color,
                        label=label)
        axes[0, 2].plot(dt.rho_norm,
                        dt.profiles.T_i.isel(time=time_index),
                        color=color,
                        label=label)
        axes[1, 0].plot(dt.rho_norm,
                        dt.profiles.j_total.isel(time=time_index) / 1e6,
                        color=color,
                        label=label)
        q = dt.profiles.q.isel(time=time_index)
        axes[1, 1].plot(dt.rho_face_norm, q, color=color, label=label)
        qmax = q[-1] if q[-1] > qmax else qmax
        axes[1, 2].plot(dt.rho_face_norm,
                        dt.profiles.magnetic_shear.isel(time=time_index),
                        color=color,
                        label=label)
    axes[1, 1].axhline(y=1.5, color='r', linestyle='--', label='q=1.5')
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', label='q=1.0')
    axes[0, 0].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[0, 0].set_ylabel(r"Vloop [V]")
    axes[0, 0].set_title(r"Loop voltage profile")
    axes[0, 0].legend()
    axes[0, 1].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[0, 1].set_ylabel(r"Temperature [keV]")
    axes[0, 1].set_title(r"Electron temperature")
    axes[0, 1].legend()
    axes[0, 2].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[0, 2].set_ylabel(r"Temperature [keV]")
    axes[0, 2].set_title(r"Ion Temperature")
    axes[0, 2].legend()
    axes[1, 0].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[1, 0].set_ylabel(r"Current Density $[MA/m^2]$")
    axes[1, 0].set_title(r"$j_{tot}$")
    axes[1, 0].legend()
    axes[1, 1].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[1, 1].set_ylabel(r"q")
    axes[1, 1].set_title(r"Safety Factor (q)")
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, qmax * 1.2])
    axes[1, 2].set_xlabel(r"Normalized Radius ($\hat{\rho}$)")
    axes[1, 2].set_ylabel(r"$\hat{s}$")
    axes[1, 2].set_title(r"Magnetic Shear ($\hat{s}$)")
    axes[1, 2].legend()
    plt.tight_layout()
    plt.savefig(get_figure_filename())
    plt.close()


def set_LH_transition_time(*, LH_transition_time: float) -> dict[str, Any]:
    config = copy.deepcopy(CONFIG)
    _validate_input(LH_transition_time, (20.0, 130.0))
    config['profile_conditions']['Ip'] = {0: 3e6, LH_transition_time: 12.5e6}
    config['sources']['ecrh']['P_total'] = {
        0: 0,
        LH_transition_time - 1: 0,
        LH_transition_time: 20.0e6
    }
    config['sources']['generic_heat']['P_total'] = {
        0: 0,
        LH_transition_time - 1: 0,
        LH_transition_time: 33.0e6
    }
    config['sources']['generic_current']['I_generic'] = {
        0: 0,
        LH_transition_time - 1: 0,
        LH_transition_time: 33.0e6 * _NBI_W_TO_MA
    }
    config['pedestal']['T_i_ped'] = {
        0: 0.5,
        LH_transition_time: 0.5,
        LH_transition_time + 5: 3.0
    }
    config['pedestal']['T_e_ped'] = {
        0: 0.5,
        LH_transition_time: 0.5,
        LH_transition_time + 5: 3.0
    }
    return config


def _validate_input(input_obj: Any, allowed_range: tuple):
    if not isinstance(input_obj, float) and not isinstance(input_obj, int):
        raise ValueError(f"Input must be a float or int for this exercise.")
    else:
        if not (allowed_range[0] <= input_obj <= allowed_range[1]):
            raise ValueError(
                f"Input value must be between {allowed_range[0]} and {allowed_range[1]}."
            )


def modify_config(*,
                  Ip=None,
                  nbi_power=None,
                  off_axis_ec_power=None,
                  off_axis_ec_location=None,
                  Z_eff=None,
                  W_to_Ne_ratio: float | None = None,
                  solver_type: str | None = None,
                  n_corrector_steps: int | None = None,
                  use_radiation: bool | None = None) -> dict[str, Any]:
    config = copy.deepcopy(CONFIG)
    if Ip is not None:
        _validate_input_dict(Ip, "Ip", (2.0e6, 18.0e6), "A")
        config['profile_conditions']['Ip'] = Ip
    if nbi_power is not None:
        _validate_input_dict(nbi_power, "nbi_power", (0.0, 33e6), "W")
        config['sources']['generic_heat']['P_total'] = nbi_power
        nbi_current = {
            time: value * _NBI_W_TO_MA
            for time, value in nbi_power.items()
        }
        config['sources']['generic_current']['I_generic'] = nbi_current
    if off_axis_ec_power is not None:
        _validate_input_dict(off_axis_ec_power, "off_axis_ec_power",
                             (0.0, 40e6), "W")
        config['sources']['ecrh']['P_total'] = off_axis_ec_power
    if off_axis_ec_location is not None:
        _validate_input_float(off_axis_ec_location, "off_axis_ec_location",
                              (0.1, 0.8))
        config['sources']['ecrh']['gaussian_location'] = off_axis_ec_location
    if Z_eff is not None:
        _validate_input_float(Z_eff, "Z_eff", (1.0, 4.0))
        config['plasma_composition']['Z_eff'] = {0.0: Z_eff, 1.0: Z_eff}
    if W_to_Ne_ratio is not None:
        _validate_input_float(W_to_Ne_ratio, "W_to_Ne_ratio", (0.0, 0.1))
        config['plasma_composition']['impurity'] = {
            'Ne': 1 - W_to_Ne_ratio,
            'W': W_to_Ne_ratio
        }
    if solver_type is not None:
        if not isinstance(solver_type, str):
            raise TypeError(
                f"Solver type must be a string. Received: {type(solver_type)}")
        if solver_type != 'newton_raphson' and solver_type != 'linear':
            raise ValueError(
                f"Solver type must be 'newton_raphson' or 'linear'. Received: {solver_type}"
            )
        config['solver']['solver_type'] = solver_type
    if n_corrector_steps is not None:
        if not isinstance(n_corrector_steps, int):
            raise TypeError(
                f"n_corrector_steps must be an int. Received: {type(n_corrector_steps)}"
            )
        if not 0 <= n_corrector_steps <= 100:
            raise ValueError(
                f"use_predictor_n_corrector_steps must be between 0 and 100. Received: {n_corrector_steps}"
            )
        config['solver']['n_corrector_steps'] = n_corrector_steps
    if use_radiation is not None:
        if not isinstance(use_radiation, bool):
            raise TypeError(
                f"use_radiation must be a bool. Received: {type(use_radiation)}"
            )
        if use_radiation:
            config['sources']['impurity_radiation'][
                'radiation_multiplier'] = 1.0
        else:
            config['sources']['impurity_radiation'][
                'radiation_multiplier'] = 0.0
    return config


def _validate_input_dict(input_obj: Any, variable_name: str,
                         allowed_range: tuple, units: str):
    if not isinstance(input_obj, dict):
        raise ValueError(
            f"Input {variable_name} must be a dict for this exercise.")
    else:
        for time, value in input_obj.items():
            if not isinstance(time, (int, float)):
                raise TypeError(
                    f"Time keys must be numbers. Received: {type(time)}")
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Value keys must be numbers. Received: {type(value)}")
            if not (allowed_range[0] <= value <= allowed_range[1]):
                raise ValueError(
                    f"{variable_name} values must be between {allowed_range[0]} and {allowed_range[1]} {units}."
                )
            if 0 < value < 1e3 and units == 'W':
                raise ValueError(
                    f'{variable_name} has suspiciously low non-zero input value {value}. Note that input units are W not MW'
                )


def _validate_input_float(input_obj: Any, variable_name: str,
                          allowed_range: tuple):
    if not isinstance(input_obj, float):
        raise ValueError(
            f"Input {variable_name} must be a float for this exercise.")
    else:
        if not (allowed_range[0] <= input_obj <= allowed_range[1]):
            raise ValueError(
                f"{variable_name} values must be between {allowed_range[0]} and {allowed_range[1]}."
            )


def run_sim(config: dict[str, Any]) -> xr.DataTree:
    torax_config = torax.ToraxConfig.from_dict(config)
    data_tree, _ = torax.run_simulation(torax_config, log_timestep_info=False)
    return data_tree


config0 = set_LH_transition_time(LH_transition_time=60)
config1 = set_LH_transition_time(LH_transition_time=80)
config2 = set_LH_transition_time(LH_transition_time=100)
config3 = set_LH_transition_time(LH_transition_time=120)
out0 = run_sim(config0)
out1 = run_sim(config1)
out2 = run_sim(config2)
out3 = run_sim(config3)
compare_timetraces(
    [out0, out1, out2, out3],
    labels=['t_LH = 60s', 't_LH = 80s', 't_LH = 100s', 't_LH = 120s'])
compare_profiles([out0, out3],
                 times=[60, 120],
                 labels=['t_LH = 60s', 't_LH = 120s'])
detailed_plot_single_sim(out0, time=60)
detailed_plot_single_sim(out3, time=120)
nbi_power0 = {0: 0.0, 99: 0.0, 100: 33e6}
nbi_power1 = {0: 0.0, 19: 0.0, 20: 16.5e6, 99: 16.5e6, 100: 33e6}
nbi_power2 = {0: 0.0, 49: 0.0, 50: 16.5e6, 99: 16.5e6, 100: 33e6}
nbi_power3 = {0: 0.0, 79: 0.0, 80: 16.5e6, 99: 16.5e6, 100: 33e6}
config0 = modify_config(nbi_power=nbi_power0)
config1 = modify_config(nbi_power=nbi_power1)
config2 = modify_config(nbi_power=nbi_power2)
config3 = modify_config(nbi_power=nbi_power3)
out0 = run_sim(config0)
out1 = run_sim(config1)
out2 = run_sim(config2)
out3 = run_sim(config3)
compare_timetraces(
    [out1, out2, out3, out0],
    labels=['t_NBI = 20s', 't_NBI = 50s', 't_NBI = 80s', 't_NBI = 100s'])
compare_profiles([out1, out0],
                 times=[80, 80],
                 labels=['t_NBI = 20s', 't_NBI = 100s'])
config_overrides0 = {
    'Ip': {
        0: 3.0e6,
        100: 12.5e6
    },
    'nbi_power': {
        0: 0.0,
        99: 0.0,
        100: 10e6
    },
    'off_axis_ec_power': {
        0: 0.0,
        99: 0.0,
        100: 40e6
    },
    'off_axis_ec_location': 0.4,
}
config_overrides1 = {
    'Ip': {
        0: 3.0e6,
        100: 12.5e6
    },
    'nbi_power': {
        0: 0.0,
        99: 0.0,
        100: 10e6
    },
    'off_axis_ec_power': {
        0: 0.0,
        49: 0.0,
        50: 40e6
    },
    'off_axis_ec_location': 0.2,
}
config_overrides2 = {
    'Ip': {
        0: 3.0e6,
        100: 12.5e6
    },
    'nbi_power': {
        0: 0.0,
        99: 0.0,
        100: 10e6
    },
    'off_axis_ec_power': {
        0: 0.0,
        49: 0.0,
        50: 40e6
    },
    'off_axis_ec_location': 0.4,
}
config_overrides3 = {
    'Ip': {
        0: 3.0e6,
        100: 12.5e6
    },
    'nbi_power': {
        0: 0.0,
        99: 0.0,
        100: 10e6
    },
    'off_axis_ec_power': {
        0: 0.0,
        49: 0.0,
        50: 40e6
    },
    'off_axis_ec_location': 0.6,
}
config_overrides4 = {
    'Ip': {
        0: 3.0e6,
        100: 10.25e6
    },
    'nbi_power': {
        0: 0.0,
        99: 0.0,
        100: 30e6
    },
    'off_axis_ec_power': {
        0: 0.0,
        49: 0.0,
        50: 20e6
    },
    'off_axis_ec_location': 0.4,
}
config0 = modify_config(**config_overrides0)
config1 = modify_config(**config_overrides1)
config2 = modify_config(**config_overrides2)
config3 = modify_config(**config_overrides3)
config4 = modify_config(**config_overrides4)
out0 = run_sim(config0)
out1 = run_sim(config1)
out2 = run_sim(config2)
out3 = run_sim(config3)
out4 = run_sim(config4)
labels = [
    '40MW ECRH at LH', '40MW early ECRH w=0.2', '40MW early ECRH w=0.4',
    '40MW early ECRH w=0.6'
]
compare_timetraces([out0, out1, out2, out3], labels=labels)
compare_profiles([out0, out1, out2, out3],
                 times=[120, 120, 120, 120],
                 labels=labels)
compare_profiles([out1, out1, out1, out1],
                 times=[48, 52, 60, 70],
                 labels=[
                     'Current hole formation', 'Current hole formation',
                     'Current hole formation', 'Current hole formation'
                 ])
compare_timetraces([out2, out4],
                   labels=[
                       '40MW early ECRH w=0.4, Ip-12.5MA',
                       '40MW early ECRH w=0.4, Ip-10.25MA'
                   ])
compare_profiles([out2, out4],
                 times=[150, 150],
                 labels=[
                     '40MW early ECRH w=0.4, Ip-12.5MA',
                     '20MW early ECRH w=0.4, Ip-10.25MA'
                 ])
config_overrides = {
    'Ip': {
        0: 3.0e6,
        80: 14.0e6,
        100: 11.5e6
    },
    'nbi_power': {
        0: 0.0,
        99: 0.0,
        100: 20e6
    },
    'off_axis_ec_power': {
        0: 5.0e6,
        99: 33.0e6,
        100: 33e6
    },
    'off_axis_ec_location': 0.4,
}
config0 = modify_config()
config1 = modify_config(**config_overrides)
out0 = run_sim(config0)
out1 = run_sim(config1)
labels = ['Non-optimized', 'Better-optimized']
compare_timetraces([out0, out1], labels=labels)
compare_profiles([out0, out1], times=[150, 150], labels=labels)
