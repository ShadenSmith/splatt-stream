
#include "StreamCPD.hxx"
#include "StreamMatrix.hxx"

extern "C" {
#include "../../mttkrp.h"
#include "../../timer.h"
#include "../../io.h"
}





StreamCPD::StreamCPD(
    ParserBase * source
) :
    _source(source)
{
}

StreamCPD::~StreamCPD()
{
}


void StreamCPD::compute(
    splatt_idx_t const rank,
    double const forget,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const stream_mode = _source->stream_mode();
  idx_t const num_modes = _source->num_modes();

  StreamMatrix * stream_mats[SPLATT_MAX_NMODES];
  for(idx_t m=0; m < num_modes; ++m) {
    stream_mats[m] = new StreamMatrix(rank);
  }


  matrix_t * mat_ptrs[SPLATT_MAX_NMODES+1];
  mat_ptrs[stream_mode] = mat_alloc(1, rank);

  /* Gram matrices */
  matrix_t * gram     = mat_zero(rank, rank);
  matrix_t * new_gram = mat_zero(rank, rank); /* new time slice: c^T * c */
  matrix_t * aTa[SPLATT_MAX_NMODES+1];
  for(idx_t m=0; m < num_modes; ++m) {
    aTa[m] = mat_zero(rank, rank);
    if(m == stream_mode) {
      memset(aTa[m]->vals, 0, rank * rank * sizeof(val_t));
    }
  }


  StreamMatrix mttkrp_buf(rank);

  /* Stream */
  idx_t it = 0;
  timer_start(&timers[TIMER_CPD]);
  sptensor_t * batch = _source->next_batch();
  while(batch != NULL) {
    sp_timer_t batch_time;
    timer_fstart(&batch_time);

    /*
     * Grow factor matrices and update Gram matrices
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m != stream_mode) {
        stream_mats[m]->grow(batch->dims[m]);
        mat_ptrs[m] = stream_mats[m]->mat();
        mat_aTa(stream_mats[m]->mat(), aTa[m]);
      }
    }


    /*
     * Compute new time slice.
     */
    timer_start(&timers[TIMER_MTTKRP]);
    mat_ptrs[SPLATT_MAX_NMODES] = mat_ptrs[stream_mode];
    mttkrp_stream(batch, mat_ptrs, stream_mode);
    timer_stop(&timers[TIMER_MTTKRP]);
    mat_form_gram(aTa, gram, num_modes, stream_mode);
    mat_cholesky(gram);
    mat_solve_cholesky(gram, mat_ptrs[stream_mode]);

    /* this slice's Gram */
    mat_aTa(mat_ptrs[stream_mode], new_gram);
    //mat_write(new_gram, NULL);


    /*
     * Update the remaining modes
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m == stream_mode) {
        continue;
      }

      /* MTTKRP */
      mttkrp_buf.grow(batch->dims[m]);
      timer_start(&timers[TIMER_MTTKRP]);
      mat_ptrs[SPLATT_MAX_NMODES] = mttkrp_buf.mat();
      mat_ptrs[SPLATT_MAX_NMODES]->I = batch->dims[m];
      mttkrp_stream(batch, mat_ptrs, m);
      timer_stop(&timers[TIMER_MTTKRP]);
    }


    /* accumulate new time vector into temporal Gram matrix */
    timer_start(&timers[TIMER_ATA]);
    val_t       * const restrict rvals = aTa[stream_mode]->vals;
    val_t const * const restrict nvals = new_gram->vals;
    #pragma omp parallel for schedule(static)
    for(idx_t r=0; r < rank; ++r) {
      rvals[r] += nvals[r];
    }
    timer_stop(&timers[TIMER_ATA]);

  
    /*
     * Batch stats
     */
    timer_stop(&batch_time);
    printf("batch %5lu: %lu nnz (%0.3fs) (%0.3e NNZ/s)\n",
        it+1, batch->nnz, batch_time.seconds,
        (double) batch->nnz / batch_time.seconds);


    /* prepare for next batch */
    tt_free(batch);
    batch = _source->next_batch();
    ++it;
  } /* while batch != NULL */
  timer_stop(&timers[TIMER_CPD]);


  mat_free(gram);
  mat_free(new_gram);
  for(idx_t m=0; m < num_modes; ++m) {
    delete stream_mats[m];
    mat_free(aTa[m]);
  }
  mat_free(mat_ptrs[stream_mode]);
}




