import json
import os
import sys
import time

import numpy as np
# from constants import deg
from input import params_chern_list as params_list
from model import construct_hilbert_space, spectrumV, fermi_projection, phason_derivative_DQ_fermi, translation_derivative, chern_character
from mpicore import MPIControl

# ----------------------------------------------------------------------
# setup mpi
# ----------------------------------------------------------------------

mpiv = MPIControl()

for j,params in enumerate(params_list):
    mpiv.print("Current Set of Parameters: ", j+1)
    time.sleep(1)

    # size of unit cell and texture parameter
    ns = params["system_sizes"]
    qs = params["q"]
    fermis = params["fermi"]
    shift = params["shift"]
    mag = params["mag"]
    n_q = len(qs)

    # set of labels for whose even subsets the chern numbers are computed
    chern_labels = np.array(["tau1","tau2","u1","u2","u3"])
    J = np.array([params[label] for label in chern_labels])

    for i in range(n_q):
        if mpiv.my_turn(i):
            print("current index:", i + 1, n_q)
            sys.stdout.flush()

            # ----------------------------------------------------------------------
            # generate output outdirectory
            # ----------------------------------------------------------------------
            outdir = ""
            # unique time stamp
            time.sleep(2*i)
            stamp = str(round(time.time()))

            # output outdirectory
            outdir = "./out/" + stamp
            os.mkdir(outdir)

            with open(outdir + "/params.json", "w") as f:
                tmp =  {key: value for key, value in params.items()}
                tmp["system_sizes"]=[ns[i]]
                tmp["q"]=qs[i]
                tmp["fermi"]=fermis[i]
                tmp["shift"]=shift[i]
                tmp["mag"]=mag[i]
                json.dump(tmp, f)

            t0 = time.time()

            # ----------------------------------------------------------------------
            # calculate Chern numbers
            # ----------------------------------------------------------------------

            # list for partial derivatives of the spectral operator
            dP_J = np.zeros((J.size,2*ns[i]*ns[i],2*ns[i]*ns[i]),dtype=np.complex128)

            # calculate the spectrum
            labels, states = construct_hilbert_space(ns[i], ns[i])

            eigenvals , eigenvecs = spectrumV(
                                    states,
                                    ns[i],
                                    ns[i],
                                    qs[i],
                                    0.0,
                                    np.array([0,0]),
                                    shift[i],
                                    mag[i],
                                    params["t"],
                                    params["m"],
                                    params["texture"],
                                    "periodic",
                                    )

            # calculate the spectral projection operator
            P = fermi_projection(eigenvals,eigenvecs,fermis[i])
            #np.save(outdir + "/spec_projection.npy", P)
            print("current index:", i + 1, n_q,"Spectral Projection: ",  time.time()-t0, np.linalg.norm(P-np.transpose(np.conj(P))) )
            sys.stdout.flush()

            # Calculate the derivatives of the spectral projection operator for all labels which are True in J
            if J[0]:
                #dP_tau1
                dP_J[0] = translation_derivative(P, 0, states, ns[i], ns[i])
                #np.save(outdir + "/projection_derivation_"+chern_labels[0]+".npy", dP_J[0])
                print("current index:", i + 1, n_q,"Translation derivative 1: ",  time.time()-t0, np.linalg.norm(dP_J[0]-np.transpose(np.conj(dP_J[0]))) )
                sys.stdout.flush()

            if J[1]:
                #dP_tau2
                dP_J[1] = translation_derivative(P, 1, states, ns[i], ns[i])
                #np.save(outdir + "/projection_derivation_"+chern_labels[1]+".npy", dP_J[1])
                print("current index:", i + 1, n_q,"Translation derivative 2: ",  time.time()-t0, np.linalg.norm(dP_J[1]-np.transpose(np.conj(dP_J[1]))) )
                sys.stdout.flush()

            if J[2]:
                #dP_phi1
                # dP_J[2] = phason_derivative_alg(
                #             0,
                #             eigenvals,
                #             eigenvecs,
                #             states,
                #             ns[i],
                #             ns[i],
                #             qs[i],
                #             shift[i],
                #             mag[i],
                #             params["m"],
                #             params["texture"],
                #             fermis[i]
                #             )
                dP_J[2] = phason_derivative_DQ_fermi(
                            P,
                            np.array([params["delta"],0]),
                            states,
                            ns[i],
                            ns[i],
                            qs[i],
                            0.0,
                            np.array([0,0]),
                            shift[i],
                            mag[i],
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            fermis[i]
                            )
                #np.save(outdir + "/projection_derivation_"+chern_labels[2]+".npy", dP_J[2])
                print("current index:", i + 1, n_q,"Phason derivative 1: ",  time.time()-t0, np.linalg.norm(dP_J[2]-np.transpose(np.conj(dP_J[2]))) )
                sys.stdout.flush()

            if J[3]:
                #dP_phi2
                # dP_J[3] = phason_derivative_alg(
                #             1,
                #             eigenvals,
                #             eigenvecs,
                #             states,
                #             ns[i],
                #             ns[i],
                #             qs[i],
                #             shift[i],
                #             mag[i],
                #             params["m"],
                #             params["texture"],
                #             fermis[i]
                #             )
                dP_J[3] = phason_derivative_DQ_fermi(
                            P,
                            np.array([0,params["delta"]]),
                            states,
                            ns[i],
                            ns[i],
                            qs[i],
                            0.0,
                            np.array([0,0]),
                            shift[i],
                            mag[i],
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            fermis[i]
                            )
                #np.save(outdir + "/projection_derivation_"+chern_labels[3]+".npy", dP_J[3])
                print("current index:", i + 1, n_q,"Phason derivative 2: ",  time.time()-t0, np.linalg.norm(dP_J[3]-np.transpose(np.conj(dP_J[3]))) )
                sys.stdout.flush()

            if J[4]:
                #dP_phi3
                # dP_J[4] = phason_derivative_alg(
                #             2,
                #             eigenvals,
                #             eigenvecs,
                #             states,
                #             ns[i],
                #             ns[i],
                #             qs[i],
                #             shift[i],
                #             mag[i],
                #             params["m"],
                #             params["texture"],
                #             fermis[i]
                #             )
                dP_J[4] = phason_derivative_DQ_fermi(
                            P,
                            np.array([-params["delta"]/2,-params["delta"]/2]),
                            states,
                            ns[i],
                            ns[i],
                            qs[i],
                            0.0,
                            np.array([0,0]),
                            shift[i],
                            mag[i],
                            params["t"],
                            params["m"],
                            params["texture"],
                            "periodic",
                            fermis[i]
                            )
                #np.save(outdir + "/projection_derivation_"+chern_labels[4]+".npy", dP_J[4])
                print("current index:", i + 1, n_q,"Phason derivative 3: ",  time.time()-t0, np.linalg.norm(dP_J[4]-np.transpose(np.conj(dP_J[4]))) )
                sys.stdout.flush()

            # Compute the Chern numbers for all even subsets of the labels which are True in J
            chern_numbers = chern_character(J,P,dP_J,ns[i],ns[i])

            key = ''.join(chern_labels[J])
            np.save(outdir + "/chern_numbers_"+key+".npy", dict(chern_numbers)) # numba does not return an ordinary dictionary but a numba.typed.Dict

            walltime = time.time()-t0
            print("current index:", i + 1, n_q,"Walltime: ", walltime)
            sys.stdout.flush()
            np.savetxt(outdir + "/walltime.txt", [walltime])

    # ---------------------------------------------------------------------
    # finalize
    # ----------------------------------------------------------------------

    mpiv.barrier()
    mpiv.print("Done!")
    mpiv.stop_clock()

    if mpiv.is_root():

        walltime = mpiv.get_time()
        # cputime = mpiv.size * mpiv.get_time() / 3600.0

        mpiv.print("Walltime: ", walltime)
        # mpiv.print("CPUs: ", mpiv.size * mpiv.get_time())
        mpiv.print("outdir: ", outdir)
        # np.savetxt(outdir + "/cputime.txt", [cputime])

mpiv.finalize()
