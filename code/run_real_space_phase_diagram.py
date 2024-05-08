from numba import njit
import numpy as np
from constants import *
from scipy.integrate import dblquad
from scipy.optimize import minimize, basinhopping


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


@njit
def winding_density(phi=np.array([0.0, 0.0]), shift=0.0, mag=0.0, eta=0.0, dphi=1e-6):

    n0 = n(phi, shift, mag, eta)

    dn_dphi_1 = (n(phi+np.array([dphi, 0.0]), shift, mag, eta) -
                 n(phi+np.array([-dphi, 0.0]), shift, mag, eta))/(2*dphi)

    dn_dphi_2 = (n(phi+np.array([0.0, dphi]), shift, mag, eta) -
                 n(phi+np.array([0.0, -dphi]), shift, mag, eta))/(2*dphi)

    return np.dot(n0, np.cross(dn_dphi_1, dn_dphi_2)) / (4*np.pi)

    # return n0[2]


def winding_number(shift=0.0, mag=0.0, eta=0.01, dphi=1e-6):

    def f(x, y):
        return winding_density(np.array([x, y]), shift, mag, eta, dphi)

    return (dblquad(f, 0, 2*np.pi, 0, 2*np.pi))


def norm(shift=0.0, mag=0.0, eta=0.1):

    def f(x):
            
        return np.linalg.norm(n(x, shift, mag, eta))

    n_rand = 10
    vals = np.zeros(n_rand)
    for i in range(n_rand):
        x0 = 2*np.pi * np.random.rand(2)

        sol = minimize(f, x0, bounds=((0, 2*np.pi), (0, 2*np.pi))).x

        vals[i] = f(sol)
        

    return np.amin(vals)


if __name__ == '__main__':

    n_m = 200
    n_shifts = 200

    ms = np.linspace(-2,2, n_m)
    shifts = np.linspace(0, 2*np.pi, n_shifts)

    phase_diagram = np.zeros((n_m, n_shifts))

    for i in range(n_m):
        for j in range(n_shifts):
            phase_diagram[i, j] = winding_number(
                shift=shifts[j], mag=ms[i], eta=0.01, dphi=1e-6)[0] #/ (2*np.pi)**2
            
            # phase_diagram[i, j] = norm(
            #     shift=shifts[j], mag=ms[i], eta=0.01)

        print(i+1, "/",  n_m)

    np.save("./real_space_phase_diagram.npy", phase_diagram)
