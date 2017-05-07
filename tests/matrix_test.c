
#include "../src/matrix.h"
#include "../src/thd_info.h"
#include "ctest/ctest.h"
#include "splatt_test.h"
#include <math.h>

#define NMATS 4

CTEST_DATA(matrix)
{
  idx_t nthreads;
  matrix_t * mats[NMATS];
};

CTEST_SETUP(matrix)
{
  data->nthreads = 7;
  splatt_omp_set_num_threads(data->nthreads);

  data->mats[0] = mat_rand(100, 3);
  data->mats[1] = mat_rand(3, 100);
  data->mats[2] = mat_rand(100, 100);
  data->mats[3] = mat_rand(100, 1);
}

CTEST_TEARDOWN(matrix)
{
  for(idx_t m=0; m < NMATS; ++m) {
    mat_free(data->mats[m]);
  }
}



CTEST2(matrix, matmul)
{
  for(idx_t m=0; m < NMATS; ++m) {
    matrix_t const * const A = data->mats[m];

    idx_t const K = A->J;
    matrix_t * B = mat_rand(K, 19);

    /* perform matrix multiplication */
    matrix_t * C = mat_alloc(B->J, A->I);
    mat_matmul(A, B, C);

    idx_t const I = A->I;
    idx_t const J = B->J;

    /* check dimensions */
    ASSERT_EQUAL(I, C->I);
    ASSERT_EQUAL(J, C->J);


    matrix_t * gold = mat_alloc(A->I, B->J);
    memset(gold->vals, 0, gold->I * gold->J * sizeof(*gold->vals));

    val_t const * const av = A->vals;
    val_t const * const bv = B->vals;
    val_t * gv = gold->vals;

    /* compute gold */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        for(idx_t k=0; k < K; ++k) {
          gv[j + (i*J)] += av[k+(i*K)] * bv[j + (k*J)];
        }
      }
    }

    /* compare */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
#if SPLATT_VAL_TYPEWIDTH == 32
        ASSERT_DBL_NEAR_TOL(gv[j+(i*J)], C->vals[j+(i*J)], 1e-4);
#else
        ASSERT_DBL_NEAR_TOL(gv[j+(i*J)], C->vals[j+(i*J)], 1e-12);
#endif
      }
    }

    mat_free(B);
    mat_free(C);
    mat_free(gold);
  }
}

CTEST2(matrix, mat_zero)
{
  idx_t const nrows = 10;
  idx_t const ncols = 10;

  matrix_t * mat = mat_zero(nrows, ncols);

  ASSERT_NOT_NULL(mat);
  ASSERT_NOT_NULL(mat->vals);
  ASSERT_EQUAL(1, mat->rowmajor);
  ASSERT_EQUAL(nrows, mat->I);
  ASSERT_EQUAL(ncols, mat->J);

  for(idx_t x=0; x < nrows * ncols; ++x) {
    ASSERT_DBL_NEAR_TOL(0., mat->vals[x], 0.);
  }

  mat_free(mat);
}
