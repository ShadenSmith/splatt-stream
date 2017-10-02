


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
extern "C" {
#include "../stream.h"
}

#include "StreamParserSimple.hxx"




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void p_print_stream_stats(
    char const * const filename,
    splatt_idx_t rank,
    splatt_idx_t const stream_mode,
    double const forget,
    splatt_cpd_opts const * const cpd_options,
    splatt_global_opts const * const global_options)
{
  printf("Streaming"
         "------------------------------------------------------\n");
  printf("STREAM-FILE=%s\n", filename);
  printf("STREAM-MODE=%" SPLATT_PF_IDX " ", stream_mode+1);
  printf("STREAM-FORGET=%f ", forget);
  printf("RANK=%" SPLATT_PF_IDX " ", rank);
  printf("MAXITS=%" SPLATT_PF_IDX " ", cpd_options->max_iterations);
  printf("TOL=%0.1e ", cpd_options->tolerance);
  printf("SEED=%d ", global_options->random_seed);
  printf("THREADS=%d\n", global_options->num_threads);

}




/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

splatt_error_type splatt_cpd_stream(
    char const * const filename,
    splatt_idx_t rank,
    splatt_idx_t const stream_mode,
    double const forget,
    splatt_cpd_opts const * const cpd_options,
    splatt_global_opts const * const global_options)
{
  p_print_stream_stats(filename, rank, stream_mode, forget,
      cpd_options, global_options);

  StreamParserSimple parser(filename, stream_mode);


  return SPLATT_SUCCESS;
}
