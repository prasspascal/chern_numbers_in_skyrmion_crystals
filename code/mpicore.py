import sys
import time

import numpy as np
from mpi4py import MPI


class MPIControl:

    # -- initialize -----------------------------------------------------

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.root = 0
        self.err = 0

        self._t0 = 0.0
        self._t1 = 0.0
        self._total = 0.0

    # -- finalize -------------------------------------------------------

    def finalize(self):
        MPI.Finalize()

    # -- barrier --------------------------------------------------------

    def barrier(self):
        self.comm.Barrier()

    # -- timing ---------------------------------------------------------

    def start_clock(self):
        if self.rank == self.root:
            self._t0 = time.time()

    def stop_clock(self):
        if self.rank == self.root:
            self._t1 = time.time()
            self._total = self._t1 - self._t0

    def get_time(self):
        if self.rank == self.root:
            return self._total

    # -- printing -------------------------------------------------------

    def print(self, *args):
        if self.rank == self.root:
            print(*args)
            sys.stdout.flush()

    # -- rank identification --------------------------------------------

    def is_root(self):
        return self.rank == self.root

    def my_turn(self, i):
        return (i % self.size) == self.rank

    def random_assignment(self, i):
        # ISO/IEC 9899 LCG
        a = 1103515245  # multiplier
        c = 12345  # increment
        m = 2**32  # modulus

        return ((a * i + c) % m) % self.size == self.rank

    # -- communication --------------------------------------------------

    def reduce_sum(self, arr, arr_red):

        # extract data type
        if arr.dtype == np.dtype(np.float64):
            type = MPI.DOUBLE
        elif arr.dtype == np.dtype(np.int32):
            type = MPI.INT
        else:
            self.print("MPI ERROR: unknown dtype")
            self.err = 1
            exit(-1)

        self.comm.Reduce([arr, type], [arr_red, type], op=MPI.SUM, root=self.root)

    def broadcast(self, arr):
        return self.comm.bcast(arr, root=self.root)

    def gather(self, arr):
        return self.comm.gather(arr, root=self.root)

    def Gatherv(self, send, recv):
        return self.comm.Gatherv(send, recv, root=self.root)

    # -- dynamic load balancing -----------------------------------------

    def assign_work(self, i):
        # find available worker unit
        worker_unit = self.comm.recv(source=MPI.ANY_SOURCE)
        # send some work to it
        self.comm.send(i, dest=worker_unit)

    def stop_working_units(self):
        for i in range(self.size - 1):
            worker_unit = self.comm.recv(source=MPI.ANY_SOURCE)
            self.comm.send(-1, dest=worker_unit)


if __name__ == "__main__":

    mpiv = MPIControl()

    print("A warm hello from: ", mpiv.rank + 1, "/", mpiv.size)
