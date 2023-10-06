#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT __declspec( dllexport )
#else
#define EXPORT
#endif

#define H(p) (p)
#define LK_CSP(p) (p)

#include <cs.h>
#define LK_CS_TYPE cs_di
#include "cs_ops.h"

void check_return(const char *call, sparse_status_t rc)
{
    const char *message = "unknown";
    switch(rc) {
    case SPARSE_STATUS_SUCCESS:
        return;
    case SPARSE_STATUS_NOT_INITIALIZED:
        message = "not-initialized";
        break;
    case SPARSE_STATUS_ALLOC_FAILED:
        message = "alloc-failed";
        break;
    case SPARSE_STATUS_INVALID_VALUE:
        message = "invalid-value";
        break;
    case SPARSE_STATUS_EXECUTION_FAILED:
        message = "execution-failed";
        break;
    case SPARSE_STATUS_INTERNAL_ERROR:
        message = "internal-error";
        break;
    case SPARSE_STATUS_NOT_SUPPORTED:
        message = "not-supported";
        break;
    }
    fprintf(stderr, "MKL call %s failed with code %d (%s)\n", call, rc, message);
    abort();
}

EXPORT lk_cs_h
lk_cs_spcreate(int nrows, int ncols, int *rowptrs, int *colinds, double *values)
{
    int nnz = rowptrs[nrows];
    cs_di *mat = NULL;
    lk_cs_h h = malloc(sizeof(lk_cs_t));
    if (!h) abort();

    if (nnz > 0) {
        mat = cs_calloc(1, sizeof(cs_di));

        mat->m = ncols;
        mat->n = nrows;
        mat->nzmax = nnz;
        mat->nz = -1;
        mat->p = rowptrs;
        mat->i = colinds;
        mat->x = values;
        h->cs_di = mat;
    }

    h->owner = 0;

#ifdef LK_TRACE
    fprintf(stderr, "allocated 0x%8lx (%dx%d, %d nnz)\n", matrix, nrows, ncols, rowptrs[nrows]);
#endif
    return H(h);
}

EXPORT lk_cs_h
lk_cs_spsubset(int rsp, int rep, int ncols, int *rowptrs, int *colinds, double *values)
{
    sparse_matrix_t matrix = NULL;
    sparse_status_t rv;
    int nrows = rep - rsp;

    rv = mkl_sparse_d_create_csr(&matrix, SPARSE_INDEX_BASE_ZERO, nrows, ncols,
                                 rowptrs + rsp, rowptrs + rsp + 1, colinds, values);
    check_return("mkl_sparse_d_create_csr", rv);

#ifdef LK_TRACE
    fprintf(stderr, "allocated 0x%8lx (%d:%d)x%d\n", matrix, rsp, rep, ncols);
#endif
    return H(matrix);
}

EXPORT void
lk_cs_spfree(lk_cs_h matrix)
{
#ifdef LK_TRACE
    fprintf(stderr, "destroying 0x%8lx\n", matrix);
#endif
    if (matrix->owner) {
        cs_di_spfree(matrix->cs_di);
    } else {
        cs_free(matrix->cs_di);
    }

    free(matrix);
}

EXPORT int
lk_cs_spmv(double alpha, lk_cs_h matrix, double *x, double beta, double *y)
{
    struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL, 0, 0
    };
    sparse_status_t rv;
    rv = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, MP(matrix), descr, x, beta, y);
    check_return("mkl_sparse_d_mv", rv);
    return rv;
}

/**
 * Compute A * B
 */
EXPORT lk_cs_h
lk_cs_spmab(lk_cs_h a, lk_cs_h b)
{
    sparse_matrix_t c = NULL;
    sparse_status_t rv;

#ifdef LK_TRACE
    fprintf(stderr, "multiplying 0x%8lx x 0x%8lx", a, b);
#endif
    rv = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, MP(a), MP(b), &c);
#ifdef LK_TRACE
    fprintf(stderr, " -> 0x%8lx\n", c);
#endif
    check_return("mkl_sparse_spmm", rv);

    return H(c);
}

/**
 * Compute A * B^T
 */
EXPORT lk_cs_h
lk_cs_spmabt(lk_cs_h a, lk_cs_h b)
{
    sparse_matrix_t c = NULL;
    sparse_status_t rv;
    struct matrix_descr descr = {
        SPARSE_MATRIX_TYPE_GENERAL, 0, 0
    };

    rv = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, MP(a),
                         SPARSE_OPERATION_TRANSPOSE, descr, MP(b),
                         SPARSE_STAGE_FULL_MULT, &c);
#ifdef LK_TRACE
    fprintf(stderr, "mult 0x%8lx x 0x%8lx^T -> 0x%8lx\n", a, b, c);
#endif
    check_return("mkl_sparse_sp2m", rv);

    return H(c);
}

EXPORT struct lk_csr
lk_cs_spexport(lk_cs_h matrix)
{
    struct lk_csr csr;
    sparse_status_t rv;
    sparse_index_base_t idx;

#ifdef LK_TRACE
    fprintf(stderr, "export 0x%8lx\n", matrix);
#endif
    rv = mkl_sparse_d_export_csr(MP(matrix), &idx, &csr.nrows, &csr.ncols,
                                 &csr.row_sp, &csr.row_ep, &csr.colinds, &csr.values);

    check_return("mkl_sparse_d_export_csr", rv);

    return csr;
}

/* Pointer-based export interface for Numba. */
EXPORT void* lk_cs_spexport_p(lk_cs_h matrix)
{
    struct lk_csr *ep = malloc(sizeof(struct lk_csr));
    if (!ep) return NULL;

    *ep = lk_cs_spexport(matrix);
    return ep;
}

EXPORT void lk_cs_spe_free(void* ep)
{
    free(ep);
}

EXPORT int lk_cs_spe_nrows(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->nrows;
}
EXPORT int lk_cs_spe_ncols(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->ncols;
}
EXPORT int* lk_cs_spe_row_sp(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->row_sp;
}
EXPORT int* lk_cs_spe_row_ep(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->row_ep;
}
EXPORT int* lk_cs_spe_colinds(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->colinds;
}
EXPORT double* lk_cs_spe_values(void* ep)
{
    struct lk_csr *csr = (struct lk_csr*) ep;
    return csr->values;
}
