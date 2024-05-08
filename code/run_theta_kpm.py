#import datetime
import json
import os
import sys
import time
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from math import floor
from input import params_thetas_kpm_list as params_list
from model import construct_hilbert_space, set_hamiltonian_kpm
from kpm import density_of_states
from mpicore import MPIControl

# ----------------------------------------------------------------------
# setup mpi
# ----------------------------------------------------------------------

mpiv = MPIControl()

for j,params in enumerate(params_list):
    mpiv.print("Current Set of Parameters: ", j+1)
    time.sleep(1)

    # ----------------------------------------------------------------------
    # generate output outdirectory
    # ----------------------------------------------------------------------

    outdir = ""
    if mpiv.is_root():

        # unique time stamp
        time.sleep(1)
        stamp = str(round(time.time()))

        # output outdirectory
        outdir = "./out/" + stamp
        os.mkdir(outdir)

        with open(outdir + "/params.json", "w") as f:
            json.dump(params, f)

    outdir = mpiv.comm.bcast(outdir, root=0)

    # ----------------------------------------------------------------------
    # calculate the spectrum
    # ----------------------------------------------------------------------

    # flux setting
    n = params["system_sizes"]
    qs = np.array([i/n for i in range(1,floor(n/2))])
    n_q = len(qs)

    # hilbert space
    labels, states = construct_hilbert_space(n, n)
    n_eigvals = 2 * (n * n)

    for i in range(n_q):
        if mpiv.my_turn(i):
            print("current index:", i + 1, n_q)
            sys.stdout.flush()

            # hamiltonian
            H = lil_matrix((n_eigvals, n_eigvals), dtype=complex)
            set_hamiltonian_kpm(
                H,
                states,
                n,
                n,
                qs[i],
                0.0,
                np.array([0,0]),
                params["shift"],
                params["mag"],
                params["t"],
                params["m"],
                params["texture"],
                "periodic",
            )
            H = csc_matrix(H)

            # spectrum
            _, dos = density_of_states(
                H,
                scale=12,
                n_moments=params["n_moments"],
                n_energies=params["n_energies"],
                n_random_states=params["n_random_states"]
            )

            key = str(i).zfill(4)
            np.save(outdir + "/dos_" + key + ".npy", dos)

    # ---------------------------------------------------------------------
    # finalize
    # ----------------------------------------------------------------------

    mpiv.barrier()
    mpiv.print("Done!")
    mpiv.stop_clock()

    if mpiv.is_root():

        walltime = mpiv.get_time()

        mpiv.print("Walltime: ", walltime)
        mpiv.print("outdir: ", outdir)
        np.savetxt(outdir + "/walltime.txt", [walltime])

mpiv.finalize()
