#ifndef LK_CS_TYPE
#define LK_CS_TYPE void
#endif
#ifndef EXPORT
#define EXPORT
#endif

typedef struct lk_cs_w {
    LK_CS_TYPE *cs;
    int owner;
} lk_cs_t;
typedef lk_cs_t* lk_cs_h;

typedef struct lk_csr {
    int nrows;
    int ncols;
    int nnz;
    int *rowptrs;
    int *colinds;
    double *values;
};

EXPORT lk_cs_h lk_cs_spcreate(int nrows, int ncols, int *rowptrs, int *colinds, double *values);
EXPORT lk_cs_h lk_cs_spsubset(int rsp, int rep, int ncols, int *rowptrs, int *colinds, double *values);
EXPORT void lk_cs_spfree(lk_cs_h matrix);
EXPORT struct lk_csr lk_cs_export(lk_cs_h matrix);

EXPORT int lk_cs_spmv(double alpha, lk_cs_h matrix, double *x, double beta, double *y);
EXPORT lk_cs_h lk_cs_spmab(lk_cs_h a, lk_cs_h b);
EXPORT lk_cs_h lk_cs_spmabt(lk_cs_h a, lk_cs_h b);
