"""
SciPy "kernel".  This kernel is not Numba-compatible, and will never be
selected as the default.  It primarily exists for ease in testing and
benchmarking CSR operations.
"""

import numpy as np
from graphblas import Matrix, Vector
from csr import CSR

max_nnz = np.iinfo('i8').max


def to_handle(csr: CSR):
    values = csr.values
    if values is None:
        values = 1.0
    return Matrix.from_csr(csr.rowptrs, csr.colinds, values, nrows=csr.nrows, ncols=csr.ncols)


def from_handle(h: Matrix):
    rps, cis, vs = h.to_csr()
    return CSR(h.nrows, h.ncols, h.nvals, rps, cis, vs)


def order_columns(h):
    pass


def release_handle(h):
    pass


def mult_ab(A, B):
    C = Matrix(nrows=A.nrows, ncols=B.ncols)
    C << A @ B
    return C


def mult_abt(A, B):
    C = Matrix(nrows=A.nrows, ncols=B.nrows)
    C << A @ B.T
    return C


def mult_vec(A, v):
    r = Vector(size=A.nrows)
    vm = Vector.from_dense(v)
    r << A @ vm
    return r.to_dense(fill_value=0)
