

/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "base.h"
#include "thd_info.h"
#include "util.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
val_t rand_val(void)
{
  /* TODO: modify this to work based on the size of idx_t */
  val_t v =  3.0 * ((val_t) rand() / (val_t) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}


idx_t rand_idx(void)
{
  /* TODO: modify this to work based on the size of idx_t */
  return (idx_t) (rand() << 16) | rand();
}


void fill_rand(
  val_t * const restrict vals,
  idx_t const nelems)
{
  for(idx_t i=0; i < nelems; ++i) {
    vals[i] = rand_val();
  }
}


char * bytes_str(
  size_t const bytes)
{
  double size = (double)bytes;
  int suff = 0;
  const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
  while(size > 1024 && suff < 5) {
    size /= 1024.;
    ++suff;
  }
  char * ret = splatt_malloc(512 * sizeof(*ret));
  sprintf(ret, "%0.2f%s", size, suffix[suff]);
  return ret;
}



idx_t argmax_elem(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = 0;
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] > arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}


idx_t argmin_elem(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = 0;
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] < arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}


int * get_primes(
  int N,
  int * nprimes)
{
  /* silly base case */
  if(N == 0) {
    *nprimes = 0;
    return NULL;
  }

  int size = 10;
  int * p = malloc(size * sizeof(*p));
  int np = 0;

  while(N != 1) {
    int i;
    for(i=2; i <= N; ++i) {
      if(N % i == 0) {
        /* found the next prime */
        break;
      }
    }

    /* realloc if necessary */
    if(size == np) {
      p = realloc(p, size * 2 * sizeof(*p));
    }

    p[np++] = i;
    N /= i;
  }

  *nprimes = np;
  return p;
}



void par_memcpy(
    void * const restrict dst,
    void const * const restrict src,
    size_t const bytes)
{
  #pragma omp parallel
  {
    int nthreads = splatt_omp_get_num_threads();
    int tid = splatt_omp_get_thread_num();

    size_t n_per_thread = (bytes + nthreads - 1)/nthreads;
    size_t n_begin = SS_MIN(n_per_thread * tid, bytes);
    size_t n_end = SS_MIN(n_begin + n_per_thread, bytes);

    memcpy((char *)dst + n_begin, (char *)src + n_begin, n_end - n_begin);
  }
}


