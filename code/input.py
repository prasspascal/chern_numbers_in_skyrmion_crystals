from constants import pi
import numpy as np

# ----------------------------------------------------------------------
# setup model
# ----------------------------------------------------------------------

params_thetas_kpm_list = [
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 400,
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- position in phase shift-magnetization phase diagram
        "shift": 0.0,
        "mag": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 400,
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- position in phase shift-magnetization phase diagram
        "shift": pi,
        "mag": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 400,
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- position in phase shift-magnetization phase diagram
        "shift": 0.0,
        "mag": 0.7,
    },
]

params_shifts_list = [
    {
       "texture": "skx",
       # -- settings of periodic boundary conditions
       "system_sizes": [137],
       # -- general model parameters
       "t": -1,  # hopping
       "m": 5,  # exchange
       # -- phase shift diagram
       "q_shift": 24/137,  # fixed q value for phase shift diagram
       "n_shift": 300,  # number of phaseshifts for which the spectra are computed
       "min_shift": 0.25*pi,  # lower bound for range of shifts (included)
       "max_shift": 0.45*pi,  # upper bound for range of shifts (included)
       # -- net magnetization (up to normalisation factor)
       "mag": 0.7,
       # -- eta for regularisation of normalisation
       "eta": 0.0,
    },
]

params_shifts_kpm_list = [
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 401,
        "q": 70/401,  # fixed q value for phase shift diagram
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 500,  # number of phaseshifts for which the spectra are computed
        "min_shift": 0,  # lower bound for range of shifts (included)
        "max_shift": pi,  # upper bound for range of shifts (included)
        "mag": 0.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(0/np.sqrt(3)),pi+np.arccos(0/np.sqrt(3))],
        # -- eta for regularisation of normalisation
        "eta": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 401,
        "q": 70/401,  # fixed q value for phase shift diagram
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 500,  # number of phaseshifts for which the spectra are computed
        "min_shift": 0,  # lower bound for range of shifts (included)
        "max_shift": pi,  # upper bound for range of shifts (included)
        "mag": 0.7,
        # -- origin in phase space
        "phi": [pi+np.arccos(0.7/np.sqrt(3)),pi+np.arccos(0.7/np.sqrt(3))],
        # -- eta for regularisation of normalisation
        "eta": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 401,
        "q": 70/401,  # fixed q value for phase shift diagram
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 500,  # number of phaseshifts for which the spectra are computed
        "min_shift": 0,  # lower bound for range of shifts (included)
        "max_shift": pi,  # upper bound for range of shifts (included)
        "mag": 0.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(0/np.sqrt(3)),pi+np.arccos(0/np.sqrt(3))],
        # -- eta for regularisation of normalisation
        "eta": 0.1,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 401,
        "q": 70/401,  # fixed q value for phase shift diagram
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 500,  # number of phaseshifts for which the spectra are computed
        "min_shift": 0,  # lower bound for range of shifts (included)
        "max_shift": pi,  # upper bound for range of shifts (included)
        "mag": 0.7,
        # -- origin in phase space
        "phi": [pi+np.arccos(0.7/np.sqrt(3)),pi+np.arccos(0.7/np.sqrt(3))],
        # -- eta for regularisation of normalisation
        "eta": 0.1,
    },
]

params_mags_kpm_list = [
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 401,
        "q": 70/401,  # fixed q value for phase shift diagram
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- positions in phase shift-magnetization phase diagram
        "shift": 0.0,
        "n_mag": 500,
        "min_mag": 0.0,
        "max_mag": 1.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(np.sqrt(3)*1/np.sqrt(3)),pi+np.arccos(np.sqrt(3)*1/np.sqrt(3))],
        # -- eta for regularisation of normalisation
        "eta": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 401,
        "q": 70/401,  # fixed q value for phase shift diagram
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,  # exchange
        # -- settings kpm
        "n_energies": 2048,
        "n_moments": 2048,
        "n_random_states": 10,
        # -- positions in phase shift-magnetization phase diagram
        "shift": 0.0,
        "n_mag": 500,
        "min_mag": 0.0,
        "max_mag": 1.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(np.sqrt(3)*1/np.sqrt(3)),pi+np.arccos(np.sqrt(3)*1/np.sqrt(3))],
        # -- eta for regularisation of normalisation
        "eta": 0.1,
    },
]

params_chern_list = [
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": [57 for _ in range(25)],
        "q": [10/57 for _ in range(25)], # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,   # exchange
        # -- position in phase shift-magnetization phase diagram
        "shift": [0.0 for _ in range(25)],
        "mag": [0.0 for _ in range(25)],
        # -- set of labels from whose subsets of even cardinality the Chern numbers are calculated
        "tau1": True,
        "tau2": True,
        "u1": True,
        "u2": True,
        "u3": False,
        # -- fermi energy
        "fermi": [-9.94, -9.42, -1.21, 0.27, 0.92, 1.48],
        # -- delta for difference quotient
        "delta": 10**-8,
    },
]

params_chern_fermis_list = [
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        "kbT": 0, # finite temperature
        # -- position in phase shift-magnetization phase diagram
        "shift": 0.0,
        "mag": 0.0,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- fermi energy
        "n_fermi": 2000,
        "min_fermi": -11.0,
        "max_fermi": 8.0,
        # -- delta for difference quotient
        "delta": 10**-8,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,   # exchange
        "kbT": 0, # finite temperature
        # -- position in phase shift-magnetization phase diagram
        "shift": pi,
        "mag": 0.0,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- fermi energy
        "n_fermi": 2000,
        "min_fermi": -11.0,
        "max_fermi": 8.0,
        # -- delta for difference quotient
        "delta": 10**-8,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,   # exchange
        "kbT": 0, # finite temperature
        # -- position in phase shift-magnetization phase diagram
        "shift": 0.0,
        "mag": 0.7,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- fermi energy
        "n_fermi": 2000,
        "min_fermi": -11.0,
        "max_fermi": 8.0,
        # -- delta for difference quotient
        "delta": 10**-8,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 53,
        "q": 20/53, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        "kbT": 0, # finite temperature
        # -- position in phase shift-magnetization phase diagram
        "shift": 0.0,
        "mag": 0.0,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- fermi energy
        "n_fermi": 2000,
        "min_fermi": -11.0,
        "max_fermi": 8.0,
        # -- delta for difference quotient
        "delta": 10**-8,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 53,
        "q": 20/53, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,   # exchange
        "kbT": 0, # finite temperature
        # -- position in phase shift-magnetization phase diagram
        "shift": pi,
        "mag": 0.0,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- fermi energy
        "n_fermi": 2000,
        "min_fermi": -11.0,
        "max_fermi": 8.0,
        # -- delta for difference quotient
        "delta": 10**-8,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 53,
        "q": 20/53, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,  # hopping
        "m": 5,   # exchange
        "kbT": 0, # finite temperature
        # -- position in phase shift-magnetization phase diagram
        "shift": 0.0,
        "mag": 0.7,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- fermi energy
        "n_fermi": 2000,
        "min_fermi": -11.0,
        "max_fermi": 8.0,
        # -- delta for difference quotient
        "delta": 10**-8,
    },
]

params_chern_shifts_ids_list = [
        {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 1000,
        "min_shift": 0.0,
        "max_shift": pi,
        "mag": 0.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(0/np.sqrt(3)),pi+np.arccos(0/np.sqrt(3))],
        # -- save spectrum for every point
        "save_spec": True,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- IDS of gap
        "ids": 1+(10/57)**2+0.5/57**2,
        # -- delta for difference quotient
        "delta": 10**-8,
        # -- eta for regularisation of normalisation
        "eta": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 1000,
        "min_shift": 0.0,
        "max_shift": pi,
        "mag": 0.7,
        # -- origin in phase space
        "phi": [pi+np.arccos(0.7/np.sqrt(3)),pi+np.arccos(0.7/np.sqrt(3))],
        # -- save spectrum for every point
        "save_spec": True,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- IDS of gap
        "ids": 1+(10/57)**2+0.5/57**2,
        # -- delta for difference quotient
        "delta": 10**-8,
        # -- eta for regularisation of normalisation
        "eta": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 1000,
        "min_shift": 0.0,
        "max_shift": pi,
        "mag": 0.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(0/np.sqrt(3)),pi+np.arccos(0/np.sqrt(3))],
        # -- save spectrum for every point
        "save_spec": True,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- IDS of gap
        "ids": 1+(10/57)**2+0.5/57**2,
        # -- delta for difference quotient
        "delta": 10**-8,
        # -- eta for regularisation of normalisation
        "eta": 0.1,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        # -- positions in phase shift-magnetization phase diagram
        "n_shift": 1000,
        "min_shift": 0.0,
        "max_shift": pi,
        "mag": 0.7,
        # -- origin in phase space
        "phi": [pi+np.arccos(0.7/np.sqrt(3)),pi+np.arccos(0.7/np.sqrt(3))],
        # -- save spectrum for every point
        "save_spec": True,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- IDS of gap
        "ids": 1+(10/57)**2+0.5/57**2,
        # -- delta for difference quotient
        "delta": 10**-8,
        # -- eta for regularisation of normalisation
        "eta": 0.1,
    },
]

params_chern_mags_ids_list = [
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        # -- positions in phase shift-magnetization phase diagram
        "shift": 0.0,
        "n_mag": 1000,
        "min_mag": 0.0,
        "max_mag": 1.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(np.sqrt(3)*1/np.sqrt(3)),pi+np.arccos(np.sqrt(3)*1/np.sqrt(3))],
        # -- save spectrum for every point
        "save_spec": True,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- IDS of gap
        "ids": 1+(10/57)**2+0.5/57**2,
        # -- delta for difference quotient
        "delta": 10**-8,
        # -- eta for regularisation of normalisation
        "eta": 0.0,
    },
    {
        "texture": "skx",
        # -- settings of periodic boundary conditions
        "system_sizes": 57,
        "q": 10/57, # fixed q value for Chern number calculation
        # -- general model parameters
        "t": -1,     # hopping
        "m": 5,      # exchange
        # -- positions in phase shift-magnetization phase diagram
        "shift": pi,
        "n_mag": 1000,
        "min_mag": 0.0,
        "max_mag": 1.0,
        # -- origin in phase space
        "phi": [pi+np.arccos(np.sqrt(3)*1/np.sqrt(3)),pi+np.arccos(np.sqrt(3)*1/np.sqrt(3))],
        # -- save spectrum for every point
        "save_spec": True,
        # -- set of Chern numbers to be calculated
        "tau1tau2": True,
        "u1u2": True,
        "tau1tau2u1u2": True,
        # -- IDS of gap
        "ids": 1+(10/57)**2+0.5/57**2,
        # -- delta for difference quotient
        "delta": 10**-8,
        # -- eta for regularisation of normalisation
        "eta": 0.1,
    },
]

params_chern_diagram_list = [
   {
       "texture": "skx",
       # -- settings of periodic boundary conditions
       "system_sizes": 57,
       "q": 10/57, # fixed q value for Chern number calculation
       # -- general model parameters
       "t": -1,     # hopping
       "m": 5,      # exchange
       # -- positions in phase shift-magnetization phase diagram
       "n_shift": 100,
       "min_shift": 0.0,
       "max_shift": pi,
       "n_mag": 100,
       "min_mag": 0.0,
       "max_mag": 2.0,
       # -- origin in phase space
       "phi": [0.0,0.0],
       # -- save spectrum for every point
       "save_spec": False,
       # -- set of Chern numbers to be calculated
       "tau1tau2": True,
       "u1u2": True,
       "tau1tau2u1u2": True,
       # -- IDS of gap
       "ids": 1+(10/57)**2+0.5/57**2,
       # -- delta for difference quotient
       "delta": 10**-8,
       # -- eta for regularisation of normalisation
       "eta": 0.0,
   },
]
