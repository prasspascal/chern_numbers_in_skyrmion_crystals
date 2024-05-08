import json
import os
import time
import sys

import numpy as np
from constants import pi
from input import params_chern_fermis_list as params_list
from model import construct_hilbert_space, spectrumV, fermi_projection, phason_derivative_DQ_fermi, translation_derivative, cyclic_cocycle
from mpicore import MPIControl

# ----------------------------------------------------------------------
# setup mpi
# ----------------------------------------------------------------------

mpiv = MPIControl()

for params in params_list:
    mpiv.start_clock()

    # size of unit cell and texture parameter
    n = params["system_sizes"]
    q = params["q"]
    shift = params["shift"]
    mag = params["mag"]

    fermis = np.linspace(params["min_fermi"],params["max_fermi"],params["n_fermi"])

    # set of labels for whose even subsets the chern numbers are computed
    tau1tau2 = params["tau1tau2"]
    u1u2 = params["u1u2"]
    tau1tau2u1u2 = params["tau1tau2u1u2"]

    if tau1tau2:
        tau1tau2_partial = np.zeros(0,dtype=np.complex128)
    if u1u2:
        u1u2_partial = np.zeros(0,dtype=np.complex128)
    if tau1tau2u1u2:
        tau1tau2u1u2_partial = np.zeros(0,dtype=np.complex128)
    partial = np.zeros(0)
    
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


    # ----------------------------------------------------------------------
    # calculate Chern numbers
    # ----------------------------------------------------------------------

    # calculate the spectrum
    labels, states = construct_hilbert_space(n, n)

    eigenvals , eigenvecs = spectrumV(
                            states,
                            n,
                            n,
                            q,
                            0.0,
                            np.array([0,0]),
                            shift,
                            mag,
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            )

    for i,fermi in enumerate(fermis):
        if mpiv.my_turn(i):
            print(i,"fermi: ",fermi)
            sys.stdout.flush()

            # list for partial derivatives of the spectral operator
            dP_J = np.zeros((4,2*n*n,2*n*n),dtype=np.complex128)

            if tau1tau2 or u1u2 or tau1tau2u1u2:
                # calculate the spectral projection operator
                P = fermi_projection(eigenvals,eigenvecs,fermi,params["kbT"])

            if tau1tau2 or tau1tau2u1u2:
                dP_J[0] = translation_derivative(P, 0, states, n, n)

                dP_J[1] = translation_derivative(P, 1, states, n, n)

            if u1u2 or tau1tau2u1u2:
                dP_J[2] = phason_derivative_DQ_fermi(
                            P,
                            np.array([params["delta"],0]),
                            states,
                            n,
                            n,
                            q,
                            0.0,
                            np.array([0,0]),
                            shift,
                            mag,
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            fermi,
                            params["kbT"]
                            )

                dP_J[3] = phason_derivative_DQ_fermi(
                            P,
                            np.array([0,params["delta"]]),
                            states,
                            n,
                            n,
                            q,
                            0.0,
                            np.array([0,0]),
                            shift,
                            mag,
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            fermi,
                            params["kbT"]
                            )

            if tau1tau2:
                tau1tau2_partial = np.append(tau1tau2_partial,[(2*pi*1j) * cyclic_cocycle(n*n,P,dP_J,[0,1])])

            if u1u2:
                u1u2_partial = np.append(u1u2_partial,[(2*pi*1j) * cyclic_cocycle(n*n,P,dP_J,[2,3])])

            if tau1tau2u1u2:
                tau1tau2u1u2_partial = np.append(tau1tau2u1u2_partial,[(2*pi*1j)**2 / 2 * cyclic_cocycle(n*n,P,dP_J,[0,1,2,3])])

            partial = np.append(partial,i)

    mpiv.barrier()
    mpiv.print("Done!")
    mpiv.stop_clock()

    # gather and flatten
    index = mpiv.gather(partial)

    if mpiv.is_root():
        index = np.array([item for sublist in index for item in sublist])
        #mpiv.print(index)
        #mpiv.print(index[np.argsort(index)])

    # gather lists of lists chern numbers
    if tau1tau2:
        tau1tau2_list = mpiv.gather(tau1tau2_partial)
    if u1u2:
        u1u2_list = mpiv.gather(u1u2_partial)
    if tau1tau2u1u2:
        tau1tau2u1u2_list = mpiv.gather(tau1tau2u1u2_partial)

    if mpiv.is_root():
        np.save(outdir + "/spectrum.npy", eigenvals)
        # flatten list of lists to numpy array, sort numpy array by index, and save numpy array
        if tau1tau2:
            tau1tau2_list = np.array([item for sublist in tau1tau2_list for item in sublist])
            tau1tau2_list = tau1tau2_list[np.argsort(index)]
            np.save(outdir + "/ch_tau1tau2.npy", tau1tau2_list)
        if u1u2:
            u1u2_list = np.array([item for sublist in u1u2_list for item in sublist])
            u1u2_list = u1u2_list[np.argsort(index)]
            np.save(outdir + "/ch_u1u2.npy", u1u2_list)
        if tau1tau2u1u2:
            tau1tau2u1u2_list = np.array([item for sublist in tau1tau2u1u2_list for item in sublist])
            tau1tau2u1u2_list = tau1tau2u1u2_list[np.argsort(index)]
            np.save(outdir + "/ch_tau1tau2u1u2.npy", tau1tau2u1u2_list)


    if mpiv.is_root():

        walltime = mpiv.get_time()

        mpiv.print("Walltime: ", walltime)
        mpiv.print("outdir: ", outdir)
        np.savetxt(outdir + "/walltime.txt", [walltime])

mpiv.finalize()
