

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "util.h"
#include "timer.h"
#include "splatt_lapack.h"
#include <math.h>


#ifdef SPLATT_USE_MPI
#include <mpi.h>
#else
/* define MPI_Comm to make life easier without MPI */
typedef int MPI_Comm;
#endif


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/


/**
* @brief Normalize each column of a and store the column l_2 norms in 'lambda'.
*        If SPLATt_USE_MPI is defined, it will aggregate the norms over MPI
*        communicator 'comm'. 'comm' is not touched if SPLATT_USE_MPI is not
*        defined.
*
* @param[out] A The matrix whose columns we normalze.
* @param[out] lambda The column norms.
* @param comm The MPI communicator.
*/
static void p_mat_2norm(
  matrix_t * const A,
  val_t * const restrict lambda,
  MPI_Comm comm)
{
  idx_t const I = A->I;
  idx_t const J = A->J;
  val_t * const restrict vals = A->vals;

  #pragma omp parallel
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * restrict mylambda = splatt_malloc(J * sizeof(*mylambda));
    for(idx_t j=0; j < J; ++j) {
      mylambda[j] = 0;
    }

    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        mylambda[j] += vals[j + (i*J)] * vals[j + (i*J)];
      }
    }

    /* do reduction on partial sums */
    thread_allreduce(mylambda, J, SPLATT_REDUCE_SUM);

    #pragma omp master
    {
#ifdef SPLATT_USE_MPI
      /* now do an MPI reduction to get the global lambda */
      timer_start(&timers[TIMER_MPI_NORM]);
      timer_start(&timers[TIMER_MPI_COMM]);
      MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_SUM, comm);
      timer_stop(&timers[TIMER_MPI_COMM]);
      timer_stop(&timers[TIMER_MPI_NORM]);
#else
      for(idx_t j=0; j < J; ++j) {
        lambda[j] = mylambda[j];
      }
#endif

      /* compute the final norms */
      for(idx_t j=0; j < J; ++j) {
        lambda[j] = sqrt(lambda[j]);
      }
    }
    #pragma omp barrier

    /* do the normalization */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < I; ++i) {
      for(idx_t j=0; j < J; ++j) {
        vals[j+(i*J)] /= lambda[j];
      }
    }

    splatt_free(mylambda);
  } /* end omp parallel */
}


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void mat_cholesky(
  matrix_t const * const A)
{
  timer_start(&timers[TIMER_CHOLESKY]);
  /* check dimensions */
  assert(A->I == A->J);

  /* Cholesky factorization */
  splatt_blas_int N = A->I;
  val_t * const restrict neqs = A->vals;
  char uplo = 'L';
  splatt_blas_int order = N;
  splatt_blas_int lda = N;
  splatt_blas_int info;
  LAPACK_DPOTRF(&uplo, &order, neqs, &lda, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRF returned %d\n", info);
  }

  timer_stop(&timers[TIMER_CHOLESKY]);
}


void mat_solve_cholesky(
    matrix_t * const cholesky,
    matrix_t * const rhs)
{
  /* Chunked AO-ADMM will call this from a parallel region. */
  if(!splatt_omp_in_parallel()) {
    timer_start(&timers[TIMER_BACKSOLVE]);
  }
  splatt_blas_int N = cholesky->I;

  /* Solve against rhs */
  char tri = 'L';
  splatt_blas_int lda = N;
  splatt_blas_int info;
  splatt_blas_int nrhs = rhs->I;
  splatt_blas_int ldb = N;
  LAPACK_DPOTRS(&tri, &N, &nrhs, cholesky->vals, &lda, rhs->vals, &ldb, &info);
  if(info) {
    fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
  }

  if(!splatt_omp_in_parallel()) {
    timer_stop(&timers[TIMER_BACKSOLVE]);
  }
}


val_t mat_trace(
    matrix_t const * const A)
{
  assert(A->I == A->J);

  idx_t const N = A->I;
  val_t const * const restrict vals = A->vals;

  val_t trace = 0.;
  for(idx_t i=0; i < N; ++i) {
    trace += vals[i + (i*N)];
  }

  return trace;
}


void mat_aTa(
  matrix_t const * const A,
  matrix_t * const ret)
{
  timer_start(&timers[TIMER_ATA]);

  /* check matrix dimensions */
  assert(ret->I == ret->J);
  assert(ret->I == A->J);
  assert(ret->vals != NULL);
  assert(A->rowmajor);
  assert(ret->rowmajor);

  idx_t const I = A->I;
  idx_t const F = A->J;

  char uplo = 'L';
  char trans = 'N'; /* actually do A * A' due to row-major ordering */
  splatt_blas_int N = (splatt_blas_int) F;
  splatt_blas_int K = (splatt_blas_int) I;
  splatt_blas_int lda = N;
  splatt_blas_int ldc = N;
  val_t alpha = 1.;
  val_t beta = 0.;

  SPLATT_BLAS(syrk)(
      &uplo, &trans,
      &N, &K,
      &alpha,
      A->vals, &lda,
      &beta,
      ret->vals, &ldc);

  timer_stop(&timers[TIMER_ATA]);
}


#ifdef SPLATT_USE_MPI
void mat_aTa_mpi(
  matrix_t const * const A,
  matrix_t * const ret,
  MPI_Comm comm)
{
  /* local matrix multiplication */
  mat_aTa(A, ret);

  /* aggregate results */
  idx_t const F = A->J;

  timer_start(&timers[TIMER_ATA]);
  timer_start(&timers[TIMER_MPI_ATA]);
  timer_start(&timers[TIMER_MPI_COMM]);
  MPI_Allreduce(MPI_IN_PLACE, ret->vals, F * F, SPLATT_MPI_VAL, MPI_SUM, comm);
  timer_stop(&timers[TIMER_MPI_COMM]);
  timer_stop(&timers[TIMER_MPI_ATA]);
  timer_stop(&timers[TIMER_ATA]);
}

#endif

void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C)
{
  timer_start(&timers[TIMER_MATMUL]);

  C->I = A->I;
  C->J = B->J;

  /* check dimensions */
  assert(A->J == B->I);
  assert(C->I * C->J <= A->I * B->J);

  /* set dimensions */
  C->I = A->I;
  C->J = B->J;

  /* This calls column-major BLAS by instead computing: C^T = B^T * A^T. */
  char transA = 'N';
  char transB = 'N';
  val_t * a_vals = B->vals;
  val_t * b_vals = A->vals;
  val_t * c_vals = C->vals;
  splatt_blas_int M = B->J;
  splatt_blas_int N = A->I;
  splatt_blas_int K = A->J;
  splatt_blas_int lda = M;
  splatt_blas_int ldb = K;
  splatt_blas_int ldc = M;
  val_t alpha = 1.;
  val_t beta  = 0.;

  SPLATT_BLAS(gemm)(
      &transA, &transB,
      &M, &N, &K,
      &alpha,
      a_vals, &lda,
      b_vals, &ldb,
      &beta,
      c_vals, &ldc);

  timer_stop(&timers[TIMER_MATMUL]);
}


void mat_normalize(
  matrix_t * const A,
  val_t * const restrict lambda)
{
  timer_start(&timers[TIMER_MATNORM]);

#ifdef SPLATT_USE_MPI
  /* passing comm=0 will break things in MPI mode */
  fprintf(stderr, "SPLATT: mat_normalize() is invalid in MPI mode. ");
  fprintf(stderr, "Use mat_normalize_mpi() instead.\n");
  return;
#endif

	p_mat_2norm(A, lambda, 0);

  timer_stop(&timers[TIMER_MATNORM]);
}


#ifdef SPLATT_USE_MPI
void mat_normalize_mpi(
  matrix_t * const A,
  val_t * const restrict lambda,
  MPI_Comm comm)
{
  timer_start(&timers[TIMER_MATNORM]);

	p_mat_2norm(A, lambda, comm);

  timer_stop(&timers[TIMER_MATNORM]);
}
#endif


void mat_form_gram(
    matrix_t * * aTa,
    matrix_t * out_mat,
    idx_t nmodes,
    idx_t mode)
{
  idx_t const N = aTa[mode]->J;
  val_t * const restrict gram = out_mat->vals;

  #pragma omp parallel
  {
    /* first initialize */
    #pragma omp for schedule(static, 1)
    for(idx_t i=0; i < N; ++i) {
      for(idx_t j=i; j < N; ++j) {
        gram[j+(i*N)] = 1.;
      }
    }

    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }

			/* only work with upper triangular */
      val_t const * const restrict mat = aTa[m]->vals;
      #pragma omp for schedule(static, 1) nowait
      for(idx_t i=0; i < N; ++i) {
        for(idx_t j=i; j < N; ++j) {
          gram[j+(i*N)] *= mat[j+(i*N)];
        }
      }
    }
  } /* omp parallel */
}


void mat_add_diag(
    matrix_t * const A,
    val_t const scalar)
{
  idx_t const rank = A->J;
  val_t * const restrict vals = A->vals;

  for(idx_t i=0; i < rank; ++i) {
    vals[i + (i*rank)] += scalar;
  }
}


matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->vals = (val_t *) splatt_malloc(nrows * ncols * sizeof(val_t));
  mat->rowmajor = 1;
  return mat;
}

matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);
  val_t * const vals = mat->vals;

  fill_rand(vals, nrows * ncols);

  return mat;
}


matrix_t * mat_zero(
  idx_t const nrows,
  idx_t const ncols)
{
  matrix_t * mat = mat_alloc(nrows, ncols);

  /* Initialize in parallel in case system is NUMA. This may bring a small
   * improvement. */
  #pragma omp parallel for schedule(static)
  for(idx_t i=0; i < nrows; ++i) {
    for(idx_t j=0; j < ncols; ++j) {
      mat->vals[j + (i*ncols)] = 0.;
    }
  }

  return mat;
}


matrix_t * mat_mkptr(
    val_t * const data,
    idx_t rows,
    idx_t cols,
    int rowmajor)
{
  matrix_t * mat = splatt_malloc(sizeof(*mat));

  mat_fillptr(mat, data, rows, cols, rowmajor);

  return mat;
}


void mat_fillptr(
    matrix_t * ptr,
    val_t * const data,
    idx_t rows,
    idx_t cols,
    int rowmajor)
{
  ptr->I = rows;
  ptr->J = cols;
  ptr->rowmajor = rowmajor;
  ptr->vals = data;
}


void mat_free(
  matrix_t * mat)
{
  if(mat == NULL) {
    return;
  }
  splatt_free(mat->vals);
  splatt_free(mat);
}


matrix_t * mat_mkrow(
  matrix_t const * const mat)
{
  assert(mat->rowmajor == 0);

  idx_t const I = mat->I;
  idx_t const J = mat->J;

  matrix_t * row = mat_alloc(I, J);
  val_t       * const restrict rowv = row->vals;
  val_t const * const restrict colv = mat->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      rowv[j + (i*J)] = colv[i + (j*I)];
    }
  }

  return row;
}

matrix_t * mat_mkcol(
  matrix_t const * const mat)
{
  assert(mat->rowmajor == 1);
  idx_t const I = mat->I;
  idx_t const J = mat->J;

  matrix_t * col = mat_alloc(I, J);
  val_t       * const restrict colv = col->vals;
  val_t const * const restrict rowv = mat->vals;

  for(idx_t i=0; i < I; ++i) {
    for(idx_t j=0; j < J; ++j) {
      colv[i + (j*I)] = rowv[j + (i*J)];
    }
  }

  col->rowmajor = 0;

  return col;
}


spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz)
{
  spmatrix_t * mat = (spmatrix_t*) splatt_malloc(sizeof(spmatrix_t));
  mat->I = nrows;
  mat->J = ncols;
  mat->nnz = nnz;
  mat->rowptr = (idx_t*) splatt_malloc((nrows+1) * sizeof(idx_t));
  mat->colind = (idx_t*) splatt_malloc(nnz * sizeof(idx_t));
  mat->vals   = (val_t*) splatt_malloc(nnz * sizeof(val_t));
  return mat;
}

void spmat_free(
  spmatrix_t * mat)
{
  free(mat->rowptr);
  free(mat->colind);
  free(mat->vals);
  free(mat);
}

