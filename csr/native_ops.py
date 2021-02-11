"""
Backend implementations of Numba operations.
"""

import logging
import numpy as np
from numba import njit, prange

from .csr import CSR, _row_extent

_log = logging.getLogger(__name__)


@njit
def _swap(a, i, j):
    a[i], a[j] = a[j], a[i]


@njit
def make_empty(nrows, ncols):
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    colinds = np.zeros(0, dtype=np.intc)
    values = np.zeros(0)
    return CSR(nrows, ncols, 0, rowptrs, colinds, True, values)


@njit
def make_structure(nrows, ncols, nnz, rowptrs, colinds):
    return CSR(nrows, ncols, nnz, rowptrs, colinds, False, np.zeros(0))


@njit
def make_complete(nrows, ncols, nnz, rowptrs, colinds, values):
    return CSR(nrows, ncols, nnz, rowptrs, colinds, True, values)


@njit
def make_unintialized(nrows, ncols, sizes):
    nnz = np.sum(sizes)
    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    for i in range(nrows):
        rowptrs[i+1] = rowptrs[i] + sizes[i]
    colinds = np.full(nnz, -1, dtype=np.intc)
    values = np.full(nnz, np.nan)
    return CSR(nrows, ncols, nnz, rowptrs, colinds, True, values)


row_extent = njit(_row_extent)


@njit
def row(csr, row):
    "Get a row as a dense vector."
    v = np.zeros(csr.ncols)
    if csr.nnz == 0:
        return v

    sp, ep = row_extent(csr, row)
    cols = csr.colinds[sp:ep]
    if csr.has_values > 0:
        v[cols] = csr.values[sp:ep]
    else:
        v[cols] = 1

    return v


@njit
def row_cs(csr, row):
    "Get the column indices for a row."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]

    return csr.colinds[sp:ep]


@njit
def row_vs(csr, row):
    "Get the nonzero values for a row."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]

    if csr.has_values:
        return csr.values[sp:ep]
    else:
        return np.full(ep - sp, 1.0)


@njit
def rowinds(csr):
    "Get the row indices for the nonzero values in a matrix."
    ris = np.zeros(csr.nnz, np.intc)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        ris[sp:ep] = i
    return ris


@njit
def subset_rows(csr, begin, end):
    "Take a subset of the rows of a CSR."
    st = csr.rowptrs[begin]
    ed = csr.rowptrs[end]
    rps = csr.rowptrs[begin:(end+1)] - st

    cis = csr.colinds[st:ed]
    if csr.has_values:
        vs = csr.values[st:ed]
    else:
        vs = None
    return CSR(end - begin, csr.ncols, ed - st, rps, cis, csr.has_values, vs)


@njit(nogil=True)
def center_rows(csr):
    "Mean-center the nonzero values of each row of a CSR."
    means = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        if sp == ep:
            continue  # empty row
        vs = row_vs(csr, i)
        m = np.mean(vs)
        means[i] = m
        csr.values[sp:ep] -= m

    return means


@njit(nogil=True)
def unit_rows(csr):
    "Normalize the rows of a CSR to unit vectors."
    norms = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        if sp == ep:
            continue  # empty row
        vs = row_vs(csr, i)
        m = np.linalg.norm(vs)
        norms[i] = m
        csr.values[sp:ep] /= m

    return norms


@njit(nogil=True)
def transpose(csr, include_values):
    "Transpose a CSR."
    brp = np.zeros(csr.ncols + 1, csr.rowptrs.dtype)
    bci = np.zeros(csr.nnz, np.int32)
    if include_values and csr.has_values:
        bvs = np.zeros(csr.nnz, np.float64)
    else:
        bvs = np.zeros(0)

    # count elements
    for i in range(csr.nrows):
        ars, are = row_extent(csr, i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            brp[j+1] += 1

    # convert to pointers
    for j in range(csr.ncols):
        brp[j+1] = brp[j] + brp[j+1]

    # construct results
    for i in range(csr.nrows):
        ars, are = row_extent(csr, i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            bci[brp[j]] = i
            if include_values and csr.has_values:
                bvs[brp[j]] = csr.values[jj]
            brp[j] += 1

    # restore pointers
    for i in range(csr.ncols - 1, 0, -1):
        brp[i] = brp[i-1]
    brp[0] = 0

    if not include_values or not csr.has_values:
        return make_structure(csr.ncols, csr.nrows, csr.nnz, brp, bci)
    else:
        return make_complete(csr.ncols, csr.nrows, csr.nnz, brp, bci, bvs)


@njit(nogil=True)
def sort_rows(csr):
    "Sort the rows of a CSR by increasing column index"
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        # bubble-sort so it's super-fast on sorted arrays
        swapped = True
        while swapped:
            swapped = False
            for j in range(sp, ep - 1):
                if csr.colinds[j] > csr.colinds[j+1]:
                    _swap(csr.colinds, j, j+1)
                    if csr.has_values:
                        _swap(csr.values, j, j+1)
                    swapped = True


@njit(nogil=True)
def _csr_align(rowinds, nrows, rowptrs, align):
    rcts = np.zeros(nrows, dtype=rowptrs.dtype)
    for r in rowinds:
        rcts[r] += 1

    rowptrs[1:] = np.cumsum(rcts)
    rpos = rowptrs[:-1].copy()

    for i in range(len(rowinds)):
        row = rowinds[i]
        pos = rpos[row]
        align[pos] = i
        rpos[row] += 1
