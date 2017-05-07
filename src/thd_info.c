

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "thd_info.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Perform a parallel SUM reduction.
*
* @param thds The data we are reducing (one array for each thread).
* @param buffer thread-local buffer.
* @param nelems How many elements in the scratch array.
*/
static void p_reduce_sum(
    val_t * * reduce_ptrs,
    val_t * buffer,
    idx_t const nelems)
{
  int const tid = splatt_omp_get_thread_num();
  int const nthreads = splatt_omp_get_num_threads();

  int half = nthreads / 2;
  while(half > 0) {
    if(tid < half && tid + half < nthreads) {
      val_t const * const target = reduce_ptrs[tid+half];
      for(idx_t i=0; i < nelems; ++i) {
        buffer[i] += target[i];
      }
    }

    #pragma omp barrier

    /* check for odd number */
    #pragma omp master
    if(half > 1 && half % 2 == 1) {
        val_t const * const last = reduce_ptrs[half-1];
        for(idx_t i=0; i < nelems; ++i) {
          buffer[i] += last[i];
        }
    }

    /* next iteration */
    half /= 2;
  }

  /* account for odd thread at end */
  #pragma omp master
  {
    if(nthreads % 2 == 1) {
      val_t const * const last = reduce_ptrs[nthreads-1];
      for(idx_t i=0; i < nelems; ++i) {
        buffer[i] += last[i];
      }
    }
  }
}


/**
* @brief Perform a parallel MAX reduction.
*
* @param thds The data we are reducing (one array for each thread).
* @param buffer thread-local buffer.
* @param nelems How many elements in the scratch array.
*/
static void p_reduce_max(
    val_t * * reduce_ptrs,
    val_t * buffer,
    idx_t const nelems)
{
  int const tid = splatt_omp_get_thread_num();
  int const nthreads = splatt_omp_get_num_threads();

  int half = nthreads / 2;
  while(half > 0) {
    if(tid < half && tid + half < nthreads) {
      val_t const * const target = reduce_ptrs[tid+half];
      for(idx_t i=0; i < nelems; ++i) {
        buffer[i] = SS_MAX(buffer[i], target[i]);
      }
    }

    #pragma omp barrier

    /* check for odd number */
    #pragma omp master
    if(half > 1 && half % 2 == 1) {
        val_t const * const last = reduce_ptrs[half-1];
        for(idx_t i=0; i < nelems; ++i) {
          buffer[i] = SS_MAX(buffer[i], last[i]);
        }
    }

    /* next iteration */
    half /= 2;
  }

  /* account for odd thread at end */
  #pragma omp master
  {
    if(nthreads % 2 == 1) {
      val_t const * const last = reduce_ptrs[nthreads-1];
      for(idx_t i=0; i < nelems; ++i) {
        buffer[i] = SS_MAX(buffer[i], last[i]);
      }
    }
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void thread_allreduce(
    val_t * const buffer,
    idx_t const nelems,
    splatt_reduce_type const which)
{
  int const tid = splatt_omp_get_thread_num();
  int const nthreads = splatt_omp_get_num_threads();

  /* used to get coherent all-to-all access to reduction data. */
  static val_t ** reduce_ptrs;

  if(nthreads == 1) {
    return;
  }

  /* get access to all thread pointers */
  #pragma omp master
  reduce_ptrs = splatt_malloc(nthreads * sizeof(*reduce_ptrs));
  #pragma omp barrier

  reduce_ptrs[tid] = buffer;
  #pragma omp barrier

  /* do the reduction */
  switch(which) {
  case SPLATT_REDUCE_SUM:
    p_reduce_sum(reduce_ptrs, buffer, nelems);
    break;
  case SPLATT_REDUCE_MAX:
    p_reduce_max(reduce_ptrs, buffer, nelems);
    break;
  default:
    fprintf(stderr, "SPLATT: thread_allreduce type '%d' not recognized.\n",
        which);
  }

  #pragma omp barrier

  /* now each thread grabs master values */
  for(idx_t i=0; i < nelems; ++i) {
    buffer[i] = reduce_ptrs[0][i];
  }
  #pragma omp barrier

  #pragma omp master
  splatt_free(reduce_ptrs);
}


thd_info * thd_init(
  idx_t const nthreads,
  idx_t const nscratch,
  ...)
{
  thd_info * thds = (thd_info *) splatt_malloc(nthreads * sizeof(thd_info));

  for(idx_t t=0; t < nthreads; ++t) {
    timer_reset(&thds[t].ttime);
    thds[t].nscratch = nscratch;
    thds[t].scratch = (void **) splatt_malloc(nscratch * sizeof(void*));
  }

  va_list args;
  va_start(args, nscratch);
  for(idx_t s=0; s < nscratch; ++s) {
    idx_t const bytes = va_arg(args, idx_t);
    for(idx_t t=0; t < nthreads; ++t) {
      thds[t].scratch[s] = (void *) splatt_malloc(bytes);
      memset(thds[t].scratch[s], 0, bytes);
    }
  }
  va_end(args);

  return thds;
}

void thd_times(
  thd_info * thds,
  idx_t const nthreads)
{
  for(idx_t t=0; t < nthreads; ++t) {
    printf("  thread: %"SPLATT_PF_IDX" %0.3fs\n", t, thds[t].ttime.seconds);
  }
}


void thd_time_stats(
  thd_info * thds,
  idx_t const nthreads)
{
  double max_time = 0.;
  double avg_time = 0.;
  for(idx_t t=0; t < nthreads; ++t) {
    avg_time += thds[t].ttime.seconds;
    max_time = SS_MAX(max_time, thds[t].ttime.seconds);
  }
  avg_time /= nthreads;

  double const imbal = (max_time - avg_time) / max_time;
  printf("  avg: %0.3fs max: %0.3fs (%0.1f%% imbalance)\n",
      avg_time, max_time, 100. * imbal);
}



void thd_reset(
  thd_info * thds,
  idx_t const nthreads)
{
  for(idx_t t=0; t < nthreads; ++t) {
    timer_reset(&thds[t].ttime);
  }
}

void thd_free(
  thd_info * thds,
  idx_t const nthreads)
{
  for(idx_t t=0; t < nthreads; ++t) {
    for(idx_t s=0; s < thds[t].nscratch; ++s) {
      free(thds[t].scratch[s]);
    }
    free(thds[t].scratch);
  }
  free(thds);
}

