
#include "base.h"
#include "thd_info.h"
#include <stdlib.h>


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
double * splatt_default_opts(void)
{
  double * opts = (double *) splatt_malloc(SPLATT_OPTION_NOPTIONS * \
      sizeof(*opts));
  for(int i=0; i < SPLATT_OPTION_NOPTIONS; ++i) {
    opts[i] = SPLATT_VAL_OFF;
  }

  opts[SPLATT_OPTION_TOLERANCE]  = DEFAULT_TOL;
  opts[SPLATT_OPTION_REGULARIZE] = 0.;
  opts[SPLATT_OPTION_NITER]      = DEFAULT_ITS;
  opts[SPLATT_OPTION_VERBOSITY]  = SPLATT_VERBOSITY_LOW;

  opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_TWOMODE;
  opts[SPLATT_OPTION_TILE]      = SPLATT_NOTILE;

  opts[SPLATT_OPTION_PRIVTHRESH] = 0.02;

  /* Tile one level by default. */
  opts[SPLATT_OPTION_TILELEVEL] = 1;

  opts[SPLATT_OPTION_DECOMP] = SPLATT_DECOMP_MEDIUM;
  opts[SPLATT_OPTION_COMM]   = SPLATT_COMM_ALL2ALL;

  opts[SPLATT_OPTION_RANDSEED] = time(NULL);

  if(splatt_omp_in_parallel()) {
    opts[SPLATT_OPTION_NTHREADS]  = 1;
  } else {
    opts[SPLATT_OPTION_NTHREADS]  = splatt_omp_get_max_threads();
  }

  return opts;
}

void splatt_free_opts(
  double * opts)
{
  free(opts);
}





splatt_global_opts * splatt_alloc_global_opts(void)
{
  splatt_global_opts * gopts = splatt_malloc(sizeof(*gopts));

  gopts->num_threads = 1;
  if(!splatt_omp_in_parallel()) {
    gopts->num_threads = splatt_omp_get_max_threads();
  }

  gopts->random_seed = time(NULL);

#if 0
  /* XXX */
  gopts->verbosity = SPLATT_VERBOSITY_HIGH;
#else
  gopts->verbosity = SPLATT_VERBOSITY_NONE;
#endif

  return gopts;
}


void splatt_free_global_opts(
    splatt_global_opts * opts)
{
  splatt_free(opts);
}

