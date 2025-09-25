CONFIG = {
    'profile_conditions': {},  # use default profile conditions
    'plasma_composition': {},  # use default plasma composition
    'numerics': {},  # use default numerics
    # circular geometry is only for testing and prototyping
    'geometry': {
        'geometry_type': 'circular',
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'fusion': {},
        'ei_exchange': {},
        'ohmic': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
