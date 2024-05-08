import numpy as np
from constants import pi
from numba import jit

# triangular lattice
a = np.array([[1.0, 0], [1 / 2.0, np.sqrt(3) / 2.0]], dtype=np.float64)

# triangular reciprocal lattice
b = np.array(
    [[2 * pi, -2 * pi / np.sqrt(3)], [0, 4 * pi / np.sqrt(3)]], dtype=np.float64
)

@jit(nopython=True)
def R(i, j):
    """
    Real-space lattice
    """

    return i * a[0] + j * a[1]


@jit(nopython=True)
def G(i, j):
    """
    Reciprocal-space lattice
    """

    return i * b[0] + j * b[1]


@jit(nopython=True)
def Twist(alpha):
    """
    Rotation by alpha [rad] around z-axis
    """

    c = np.cos(alpha)
    s = np.sin(alpha)

    return np.array([[c, -s], [s, c]], dtype=np.float64)


@jit(nopython=True)
def Twist3D(alpha):
    """
    Rotation by alpha [rad] around z-axis
    """

    c = np.cos(alpha)
    s = np.sin(alpha)

    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


if __name__ == "__main__":

    alpha = 1

    print(Twist3D(alpha))
    print(Twist(alpha))
