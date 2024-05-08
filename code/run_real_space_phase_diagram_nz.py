"""
    ./triangular_lattice/run
"""

from numba import njit
import numpy as np
from constants import *
from scipy.integrate import dblquad


@njit
def n(phi=np.array([0.0, 0.0]), shift=0.0, mag=0.0, eta=0.0):
    """
    Unraveling of the noncommutative torus onto the skyrmion lattice

    theta: wavenumber
    alpha: twist angle
    phi  : origin in phason (used for phason derivatives)
    shift: phase shift (applied to phase 1), relative phase degree of freedom moving the origin
    mag: parameter to tune net magnetization (proportional up to normalization)
    """

    # magnetization vector
    m = np.zeros(3, dtype=np.float64)

    # phase factors
    phase_1 = phi[0] 
    phase_2 = phi[1]
    phase_3 = shift - phi[0] - phi[1]

    # if texture == "skx":]
    n1 = np.dot(rot240, np.array([np.sin(phase_1), 0, np.cos(phase_1)]))
    n2 = np.array([np.sin(phase_2), 0, np.cos(phase_2)])  
    n3 = np.dot(rot120, np.array([np.sin(phase_3), 0, np.cos(phase_3)]))

    m = (n1 + n2 + n3) + np.array([0, 0, np.sqrt(3) * mag])
    m = m * (1 / (np.linalg.norm(m) + eta))

    return m



def net_magnetization(shift=0.0, mag=0.0, eta=0.01):

    def f(x, y):
        return n(np.array([x, y]), shift, mag, eta)[2] / (2*np.pi)**2
    
    return (dblquad(f, 0, 2*np.pi, 0, 2*np.pi))



if __name__ == '__main__':

    n_m = 200
    n_shifts = 200

    ms = np.linspace(-2,2, n_m)
    shifts = np.linspace(0, 2*np.pi, n_shifts)

    phase_diagram = np.zeros((n_m, n_shifts))

    for i in range(n_m):
        for j in range(n_shifts):
            phase_diagram[i, j] = net_magnetization(
                shift=shifts[j], mag=ms[i], eta=0.00)[0] 

        print(i+1, "/",  n_m)

    np.save("./real_space_phase_diagram_nz.npy", phase_diagram)
