#import datetime
import json
import os
import sys
import time
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from input import params_mags_kpm_list as params_list
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
    q = params["q"]
    n_mag = params["n_mag"]
    phi = params["phi"]

    # generate list of shifts from 0 to pi to evaluate spectra at
    mags = np.linspace(params["min_mag"], params["max_mag"], n_mag)

    if mpiv.is_root():

        np.save(outdir + "/shifts.npy", mags)

    # hilbert space
    labels, states = construct_hilbert_space(n, n)
    n_eigvals = 2 * (n * n)

    for i in range(n_mag):
        if mpiv.my_turn(i):
            print("current index:", i + 1, n_mag)
            sys.stdout.flush()

            # hamiltonian
            H = lil_matrix((n_eigvals, n_eigvals), dtype=complex)
            set_hamiltonian_kpm(
                H,
                states,
                n,
                n,
                q,
                0.0,
                np.array(phi),
                params["shift"],
                mags[i],
                params["t"],
                params["m"],
                params["texture"],
                "periodic",
                params["eta"]
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
