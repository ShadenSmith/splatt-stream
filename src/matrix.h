#ifndef SPLATT_MATRIX_H
#define SPLATT_MATRIX_H

#include "base.h"


/******************************************************************************
 * STRUCTURES
 *****************************************************************************/
typedef struct
{
  idx_t I;
  idx_t J;
  val_t *vals;
  int rowmajor;
} matrix_t;

typedef struct
{
  idx_t I;
  idx_t J;
  idx_t nnz;
  idx_t * rowptr;
  idx_t * colind;
  val_t * vals;
} spmatrix_t;


/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "splatt_mpi.h"
#include "thd_info.h"



/******************************************************************************
 * LAPACK PROTOTYPES
 *****************************************************************************/

#if   SPLATT_VAL_TYPEWIDTH == 32
  void spotrf_(char *, int *, float *, int *, int *);
  void spotrs_(char *, int *, int *, float *, int *, float *, int *, int *);
  void ssyrk_(char *, char *, int *, int *, float *, float *, int *, float *, float *, int *);
  void sgbtrf_(int *, int *, int *, int *, float *, int *, int *, int *);
  void sgbtrs_(char *, int *, int *, int *, int *, float *, int *, int *, float *, int *, int *);

  #define LAPACK_DPOTRF spotrf_
  #define LAPACK_DPOTRS spotrs_
  #define LAPACK_DSYRK  ssyrk_
  #define LAPACK_DGBTRF sgbtrf_
  #define LAPACK_DGBTRS sgbtrs_
#else
  void dpotrf_(char *, int *, double *, int *, int *);
  void dpotrs_(char *, int *, int *, double *, int *, double *, int *, int *);
  void dsyrk_(char *, char *, int *, int *, double *, double *, int *, double *, double *, int *);
  void dgbtrf_(int *, int *, int *, int *, double *, int *, int *, int *);
  void dgbtrs_(char *, int *, int *, int *, int *, double *, int *, int *, double *, int *, int *);

  #define LAPACK_DPOTRF dpotrf_
  #define LAPACK_DPOTRS dpotrs_
  #define LAPACK_DSYRK  dsyrk_
  #define LAPACK_DGBTRF dgbtrf_
  #define LAPACK_DGBTRS dgbtrs_
#endif




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/



#define mat_normalize splatt_mat_normalize
/**
* @brief Normalize the columns of A using l_2 norm and return the norms in
*        lambda.
*
* @param A The matrix to normalize.
* @param lambda The vector of column norms.
*/
void mat_normalize(
  matrix_t * const A,
  val_t * const restrict lambda);



#ifdef SPLATT_USE_MPI
#define mat_normalize_mpi splatt_mat_normalize_mpi
/**
* @brief Normalize the columns of A and return the norms in lambda.
*
* @param A The matrix to normalize.
* @param which Which norm to use.
* @param comm MPI communicator to use.
*/
void mat_normalize_mpi(
  matrix_t * const A,
  val_t * const restrict lambda,
  MPI_Comm comm);
#endif



#define mat_fillptr splatt_mat_fillptr
/**
* @brief Fill a matrix pointer 'ptr' with a SHALLOW COPY of the existing data.
*
* @param[out] ptr The structure to fill.
* @param data The data to copy into the matrix structure.
* @param I The number of rows in the matrix.
* @param J The number of cols in the matrix.
* @param rowmajor If the matrix is row major.
*/
void mat_fillptr(
    matrix_t * ptr,
    val_t * const data,
    idx_t rows,
    idx_t cols,
    int rowmajor);


#define mat_mkptr splatt_mat_mkptr
/**
* @brief Fill a dense matrix with a SHALLOW COPY of the existing data. The
*        struct is allocated for you.
*
* @param data The data to copy into the matrix structure.
* @param I The number of rows in the matrix.
* @param J The number of cols in the matrix.
* @param rowmajor If the matrix is row major.
*
* @return A pointer to the created struct.
*/
matrix_t * mat_mkptr(
    val_t * const data,
    idx_t rows,
    idx_t cols,
    int rowmajor);


#define mat_cholesky splatt_mat_cholesky
/**
* @brief Compute the Cholesky factorization of A.
*
* @param A The SPD matrix A.
*/
void mat_cholesky(
    matrix_t const * const A);


void mat_solve_cholesky(
    matrix_t * const cholesky,
    matrix_t * const rhs);

val_t mat_trace(
    matrix_t const * const A);

void mat_add_diag(
    matrix_t * const A,
    val_t const scalar);


#define mat_matmul splatt_mat_matmul
/**
* @brief Dense matrix-matrix multiplication, C = AB + C.
*
* @param A The left multiplication parameter.
* @param B The right multiplication parameter.
* @param C The result matrix. NOTE: C is not zeroed before multiplication!
*/
void mat_matmul(
  matrix_t const * const A,
  matrix_t const * const B,
  matrix_t  * const C);


#define mat_aTa splatt_mat_aTa
/**
* @brief Compute A^T * A with a nice row-major pattern.
*
* @param A The input matrix.
* @param[out] ret The output matrix, A^T * A.
*/
void mat_aTa(
  matrix_t const * const A,
  matrix_t * const ret);


#ifdef SPLATT_USE_MPI
/**
* @brief Perform `mat_ata()` when the matrix is row-distributed among MPI
* ranks. The result will be stored in 'ret' on all ranks (via
* `MPI_Allreduce()`).
*
* @param A The distributed matrix.
* @param[out] ret The output matrix, A^T * A.
* @param comm The MPI communicator.
*/
void mat_aTa_mpi(
  matrix_t const * const A,
  matrix_t * const ret,
  MPI_Comm comm);
#endif



void mat_form_gram(
    matrix_t * * aTa,
    matrix_t * out_mat,
    idx_t nmodes,
    idx_t mode);


#define mat_rand splatt_mat_rand
/**
* @brief Return a randomly initialized matrix (from util's rand_val()).
*
* @param nrows The number of rows in the matrix.
* @param ncols The number of columns in the matrix.
*
* @return The random matrix.
*/
matrix_t * mat_rand(
  idx_t const nrows,
  idx_t const ncols);


#define mat_alloc splatt_mat_alloc
/**
* @brief Allocate a dense matrix. The values will not be initialized. This
*        matrix must be freed with mat_free().
*
* @param nrows The number of rows in the matrix.
* @param ncols The number of columns in the matrix.
*
* @return The allocated matrix.
*/
matrix_t * mat_alloc(
  idx_t const nrows,
  idx_t const ncols);


#define mat_zero splatt_mat_zero
/**
* @brief Allocate a dense matrix and initialize with zero.  Matrix must be
*        freed with mat_free().
*
* @param nrows The number of rows in the matrix.
* @param ncols The number of columns in the matrix.
*
* @return The allocated zero matrix.
*/
matrix_t * mat_zero(
  idx_t const nrows,
  idx_t const ncols);


#define mat_free splatt_mat_free
/**
* @brief Free a matrix allocated with mat_alloc(). This also frees the matrix
*        pointer!
*
* @param mat The matrix to be freed.
*/
void mat_free(
  matrix_t * mat);


#define spmat_alloc splatt_spmat_alloc
/**
* @brief Allocate a sparse matrix in CSR format. The values will not be
*        initialized. This matrix must be freed with spmat_free().
*
* @param nrows The number of rows in the sparse matrix.
* @param ncols The number of columns in the sparse matrix.
* @param nnz The number of nonzero values in the sparse matrix.
*
* @return The allocated CSR matrix.
*/
spmatrix_t * spmat_alloc(
  idx_t const nrows,
  idx_t const ncols,
  idx_t const nnz);


#define spmat_free splatt_spmat_free
/**
* @brief Free a sparse matrix allocated with spmat_alloc(). This also frees the
*        matrix pointer!
*
* @param mat The sparse matrix to be freed.
*/
void spmat_free(
  spmatrix_t * mat);


#define mat_mkrow splatt_mat_mkrow
/**
* @brief Copies a column-major matrix and returns a row-major version.
*
* @param mat The column-major matrix to copy.
*
* @return A row-major copy of mat.
*/
matrix_t * mat_mkrow(
  matrix_t const * const mat);


#define mat_mkcol splatt_mat_mkcol
/**
* @brief Copies a row-major matrix and returns a column-major version.
*
* @param mat The row-major matrix to copy.
*
* @return A column-major copy of mat.
*/
matrix_t * mat_mkcol(
  matrix_t const * const mat);

#endif
