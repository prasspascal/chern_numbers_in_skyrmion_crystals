import json
import os
import time
import sys

import numpy as np
from constants import pi
from input import params_chern_mags_ids_list as params_list
from model import construct_hilbert_space, spectrumV, spec_projection, phason_derivative_DQ_IDS, translation_derivative, cyclic_cocycle
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
    mags = np.linspace(params["min_mag"],params["max_mag"],params["n_mag"])
    phi = params["phi"]

    ids = params["ids"]

    # set of labels for whose even subsets the chern numbers are computed
    save_spec = params["save_spec"]
    tau1tau2 = params["tau1tau2"]
    u1u2 = params["u1u2"]
    tau1tau2u1u2 = params["tau1tau2u1u2"]

    fermi_partial = np.zeros(0,dtype=np.complex128)
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
            tmp =  {key: value for key, value in params.items()}
            json.dump(tmp, f)

    outdir = mpiv.comm.bcast(outdir, root=0)
    # ----------------------------------------------------------------------
    # calculate Chern numbers
    # ----------------------------------------------------------------------

    # calculate the spectrum
    labels, states = construct_hilbert_space(n, n)

    for i,mag in enumerate(mags):
        if mpiv.my_turn(i):
            print(i,"mag: ",mag)
            sys.stdout.flush()

            eigenvals , eigenvecs = spectrumV(
                        states,
                        n,
                        n,
                        q,
                        0.0,
                        np.array(phi),
                        shift,
                        mag,
                        params["t"],
                        params["m"],
                        params["texture"],
                        "periodic",
                        params["eta"]
                        )

            # save spectrum in file
            if save_spec:
                key = str(i).zfill(4)
                np.save(outdir + "/spec_" + key + ".npy", eigenvals)

            # list for partial derivatives of the spectral operator
            dP_J = np.zeros((4,2*n*n,2*n*n),dtype=np.complex128)
            
            # calculate the spectral projection operator
            P = spec_projection(n,n,eigenvecs,ids)

            if tau1tau2 or tau1tau2u1u2:
                dP_J[0] = translation_derivative(P, 0, states, n, n)

                dP_J[1] = translation_derivative(P, 1, states, n, n)

            if u1u2 or tau1tau2u1u2:
                dP_J[2] = phason_derivative_DQ_IDS(
                            P,
                            np.array([params["delta"],0]),
                            states,
                            n,
                            n,
                            q,
                            0.0,
                            np.array(phi),
                            shift,
                            mag,
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            ids,
                            params["eta"]
                            )

                dP_J[3] = phason_derivative_DQ_IDS(
                            P,
                            np.array([0,params["delta"]]),
                            states,
                            n,
                            n,
                            q,
                            0.0,
                            np.array(phi),
                            shift,
                            mag,
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            ids,
                            params["eta"]
                            )

            # fermi energy to given IDS (in the middle of the corresponding gap)
            fermi = (eigenvals[int(ids*n*n)]+eigenvals[int(ids*n*n)-1])/2
            fermi_partial = np.append(fermi_partial,[fermi])

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
    fermi_list = mpiv.gather(fermi_partial)
    if tau1tau2:
        tau1tau2_list = mpiv.gather(tau1tau2_partial)
    if u1u2:
        u1u2_list = mpiv.gather(u1u2_partial)
    if tau1tau2u1u2:
        tau1tau2u1u2_list = mpiv.gather(tau1tau2u1u2_partial)

    if mpiv.is_root():
        # flatten list of lists to numpy array, sort numpy array by index, and save numpy array
        fermi_list = np.array([item for sublist in fermi_list for item in sublist])
        fermi_list = fermi_list[np.argsort(index)]
        np.save(outdir + "/fermi.npy", fermi_list)
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
