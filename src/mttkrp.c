
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode)
{
  matrix_t       * const M = mats[ft->dim_perms[mode][0]];
  matrix_t const * const A = mats[ft->dim_perms[mode][1]];
  matrix_t const * const B = mats[ft->dim_perms[mode][2]];
  idx_t const nslices = M->I;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, nslices * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;
  printf("grabbing M: %u x %u\n", M->I, M->J);
  printf("grabbing A: %u x %u\n", A->I, A->J);
  printf("grabbing B: %u x %u\n", B->I, B->J);

  idx_t const * const restrict sptr = ft->sptr[mode];
  idx_t const * const restrict fptr = ft->fptr[mode];
  idx_t const * const restrict fids = ft->fids[mode];
  idx_t const * const restrict inds = ft->inds[mode];
  val_t const * const restrict vals = ft->vals[mode];

  val_t * restrict accumF = (val_t *) malloc(rank * sizeof(val_t));

  for(idx_t s=0; s < nslices; ++s) {
    val_t * const restrict mv = mvals + (s * rank);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      val_t const * const restrict av = avals  + (fids[f] * rank);
      for(idx_t r=0; r < rank; ++r) {
        mv[r] += accumF[r] * av[r];
      }
    }
  }

  free(accumF);
}


void mttkrp_giga(
  spmatrix_t const * const spmat,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[mode];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  printf("grabbing M: %u x %u\n", M->I, M->J);
  printf("grabbing A: %u x %u\n", A->I, A->J);
  printf("grabbing B: %u x %u\n", B->I, B->J);

  idx_t const I = M->I;
  idx_t const rank = M->J;

  idx_t const * const restrict rowptr = spmat->rowptr;
  idx_t const * const restrict colind = spmat->colind;
  val_t const * const restrict vals   = spmat->vals;

  for(idx_t r=0; r < rank; ++r) {
#if 0
    val_t       * const restrict mv =  M->vals + (r * M->I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);
#endif

    /* Joined Hadamard products of X, C, and B */
    for(idx_t i=0; i < I; ++i) {
      for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
        idx_t const a = colind[y] / B->I;
        idx_t const b = colind[y] % B->I;
        printf("a: %u b: %u\n", a, b);
        //scratch[y] = vals[y] * av[a] * bv[b];
        scratch[y] = vals[y] * A->vals[r + (a*rank)] * B->vals[r + (b*rank)];
      }
    }

    /* now accumulate rows into column of M1 */
    for(idx_t i=0; i < I; ++i) {
      val_t sum = 0;
      for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
        sum += scratch[y];
      }
      //mv[i] = sum;
      M->vals[r + (i*rank)] = sum;
    }
  }
}


void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[mode];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];
  idx_t const rank = M->J;

  memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  printf("grabbing M: %u x %u\n", M->I, M->J);
  printf("grabbing A: %u x %u\n", A->I, A->J);
  printf("grabbing B: %u x %u\n", B->I, B->J);


  val_t * const restrict m1vals = M->vals;

  idx_t const nnz = tt->nnz;
  idx_t const * const restrict indM = tt->ind[mode];
  idx_t const * const restrict indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
  idx_t const * const restrict indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];

  val_t const * const restrict vals = tt->vals;

  for(idx_t r=0; r < rank; ++r) {
#if 0
    val_t       * const restrict mv =  M->vals + (r * M->I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);
#endif

    /* stretch out columns of A and B */
    for(idx_t x=0; x < nnz; ++x) {
      //scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
      scratch[x] = vals[x] * A->vals[r + (rank*indA[x])] * B->vals[r + (rank*indB[x])];
      printf("a: %u b: %u\n", indA[x], indB[x]);
    }

    /* now accumulate into m1 */
    for(idx_t x=0; x < nnz; ++x) {
      //mv[indM[x]] += scratch[x];
      M->vals[r + (rank * indM[x])] += scratch[x];
    }
  }
}



