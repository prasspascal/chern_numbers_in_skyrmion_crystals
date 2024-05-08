# Python code

The python scripts `./code/run_*` generate the data used in our work from the input parameters stored in `./code/input.py`. The output is sent to the directory `./code/out/`, which needs to be created first.
The parameters for each generated data set are recorded in the respective `*/params.json` files.

The python files `./code/model.py`, `./code/lattice.py`, and `./code/constants.py` contains all methods used to set up the spin Hamiltonian, construct the Fermi projections, and evaluate the Chern numbers.

The python file `./code/kpm.py` contains the methods for the Kernel Polynomial Method. See DOI: [10.1103/RevModPhys.78.275](https://doi.org/10.1103/RevModPhys.78.275) for a detailed description of the method itself.

The python file `./code/mpicore.py` sets up MPI for Python to provide Python bindings for the Message Passing Interface (MPI) standard, allowing Python applications to exploit multiple processors on workstations, clusters, and supercomputers.