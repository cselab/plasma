CONFIG = {
    'plasma_composition': {
        'main_ion': {
            'D': 0.5,
            'T': 0.5
        },  
        'impurity': 'Ne',
        'Z_eff': 1.6,  
    },
    'profile_conditions': {
        'Ip': {
            0: 3e6,
            80: 10.5e6
        },  
        'T_i': {
            0.0: {
                0.0: 6.0,
                1.0: 0.1
            }
        },
        'T_i_right_bc':
        0.1,  
        'T_e': {
            0.0: {
                0.0: 6.0,
                1.0: 0.1
            }
        },
        'T_e_right_bc': 0.1,  
        'n_e_right_bc_is_fGW': True,
        'n_e_right_bc': {
            0: 0.1,
            80: 0.3
        },
        'n_e_nbar_is_fGW': True,
        'nbar': 1,
        'n_e': {
            0: {
                0.0: 1.5,
                1.0: 1.0
            }
        },  
    },
    'numerics': {
        't_final': 80,  
        'fixed_dt': 2,
        'resistivity_multiplier': 1,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'max_dt': 0.5,
        'chi_timestep_prefactor': 30,
        'dt_reduction_factor': 3,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
        'R_major': 6.2,  
        'a_minor': 2.0,  
        'B_0': 5.3,  
    },
    'neoclassical': {
        'bootstrap_current': {
            'bootstrap_multiplier': 1.0,
        },
    },
    'sources': {
        'generic_current': {
            'fraction_of_total_current': 0.15,
            'gaussian_width': 0.075,
            'gaussian_location': 0.36,
        },
        'generic_particle': {
            'S_total': 0.0,
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'gas_puff': {
            'puff_decay_length': 0.3,
            'S_total': 0.0,
        },
        'ohmic': {},
        'pellet': {
            'S_total': 0.0e22,
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        'generic_heat': {
            'gaussian_location': 0.12741589640723575,
            'gaussian_width': 0.07280908366127758,
            'P_total': 20.0e6,
            'electron_heat_fraction': 1.0,
        },
        'fusion': {},
        'ei_exchange': {
            'Qei_multiplier': 1.0,
        },
    },
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'T_i_ped': 1.0,  
        'T_e_ped': 1.0,  
        'n_e_ped_is_fGW': True,
        'n_e_ped': {
            0: 0.3,
            80: 0.7
        },
        'rho_norm_ped_top': 0.9,  
    },
    'transport': {
        'model_name': 'qlknn',
        'apply_inner_patch': True,
        'D_e_inner': 0.25,
        'V_e_inner': 0.0,
        'chi_i_inner': 1.5,
        'chi_e_inner': 1.5,
        'rho_inner': 0.3,  
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.9,  
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
        'avoid_big_negative_s': True,
        'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },
    'solver': {
        'solver_type': 'newton_raphson',
        'use_predictor_corrector': True,
        'n_corrector_steps': 10,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
        'log_iterations': False,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
