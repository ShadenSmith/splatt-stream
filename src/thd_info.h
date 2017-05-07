#ifndef SPLATT_THDINFO_H
#define SPLATT_THDINFO_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdarg.h>


/******************************************************************************
 * PUBLIC STRUCTURES
 *****************************************************************************/

/**
* @brief A general structure for data structures that need to be thread-local.
*/
typedef struct
{
  idx_t nscratch;
  void ** scratch;
  sp_timer_t ttime;
} thd_info;


/**
* @brief The types of parallel reductions supported.
*/
typedef enum
{
  SPLATT_REDUCE_SUM,
  SPLATT_REDUCE_MAX
} splatt_reduce_type;



/******************************************************************************
 * OPENMP WRAPPER FUNCTIONS
 *****************************************************************************/

#ifdef _OPENMP
static inline void splatt_omp_set_num_threads(
    int num_threads)
{
  omp_set_num_threads(num_threads);
}

static inline int splatt_omp_in_parallel()
{
  return omp_in_parallel();
}

static inline int splatt_omp_get_thread_num()
{
  return omp_get_thread_num();
}

static inline int splatt_omp_get_max_threads()
{
  return omp_get_max_threads();
}

static inline int splatt_omp_get_num_threads()
{
  return omp_get_num_threads();
}

#else
static inline void splatt_omp_set_num_threads(
    int num_threads)
{
  /* do nothing */
}

static inline int splatt_omp_get_thread_num()
{
  return 0;
}

static inline int splatt_omp_in_parallel()
{
  return 0;
}

static inline int splatt_omp_get_max_threads()
{
  return 1;
}

static inline int splatt_omp_get_num_threads()
{
  return 1;
}
#endif




/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define thread_allreduce splatt_thread_allreduce
/**
* @brief Do a parallel all-to-all reduction with each threads' buffer.
*
* @param buffer A thread's private buffer.
* @param nelems The length of the buffer.
* @param which Which type of reduction.
*/
void thread_allreduce(
    val_t * const buffer,
    idx_t const nelems,
    splatt_reduce_type const which);


#define thd_times splatt_thd_times
/**
* @brief Output a list of all thread timers.
*
* @param thds The array on thd_info structs.
* @param nthreads The number of timers to print.
*/
void thd_times(
  thd_info * thds,
  idx_t const nthreads);


#define thd_time_stats splatt_thd_time_stats
/**
* @brief Output a summary to STDOUT of thread timers.
*
* @param thds The array on thd_info structs.
* @param nthreads The number of timers to print.
*/
void thd_time_stats(
  thd_info * thds,
  idx_t const nthreads);


#define thd_reset splatt_thd_reset
/**
* @brief Reset an array of thread timers.
*
* @param thds The array of thd_info structs.
* @param nthreads The number of times to reset.
*/
void thd_reset(
  thd_info * thds,
  idx_t const nthreads);

#define thd_init splatt_thd_init
/**
* @brief Allocate and initialize a number thd_info structs.
*
* @param nthreads The number of threads to allocate for.
* @param nscratch The number of scratch arrays to use.
* @param ...      The number of bytes to allocate for each scratch array.
*
* @return A pointer to an array of thd_info.
*/
thd_info * thd_init(
  idx_t const nthreads,
  idx_t const nscratch,
  ...);


#define thd_free splatt_thd_free
/**
* @brief Free the memory allocated by thd_init.
*
* @param thds The array of thd_info structs.
* @param nthreads The number of threads to free.
*/
void thd_free(
  thd_info * thds,
  idx_t const nthreads);

#endif
