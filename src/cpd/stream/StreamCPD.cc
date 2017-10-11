
#include "StreamCPD.hxx"
#include "StreamMatrix.hxx"

extern "C" {
#include "../admm.h"
#include "../../mttkrp.h"
#include "../../timer.h"
#include "../../io.h" /* XXX debug */
#include "../../util.h"

#include <cblas.h>
}


static void p_copy_upper_tri(
    matrix_t * const M)
{
  timer_start(&timers[TIMER_ATA]);
  idx_t const I = M->I;
  idx_t const J = M->J;
  val_t * const restrict vals = M->vals;

  #pragma omp parallel for schedule(static, 1) if(I > 50)
  for(idx_t i=1; i < I; ++i) {
    for(idx_t j=0; j < i; ++j) {
      vals[j + (i*J)] = vals[i + (j*J)];
    }
  }
  timer_stop(&timers[TIMER_ATA]);
}



splatt_kruskal * StreamCPD::get_kruskal()
{
  /* store output */
  splatt_kruskal * cpd = (splatt_kruskal *) splatt_malloc(sizeof(*cpd));
  cpd->nmodes = _nmodes;
  cpd->lambda = (val_t *) splatt_malloc(_rank * sizeof(*cpd->lambda));
  cpd->rank = _rank;
  for(idx_t r=0; r < _rank; ++r) {
    cpd->lambda[r] = 1.;
  }
  for(idx_t m=0; m < _nmodes; ++m) {
    if(m == _stream_mode) {
      idx_t const nrows = _global_time->num_rows();
      cpd->dims[m] = nrows;
      cpd->factors[m] = (val_t *)
          splatt_malloc(nrows * _rank * sizeof(val_t));
      par_memcpy(cpd->factors[m], _global_time->vals(), nrows * _rank * sizeof(val_t));

    } else {
      idx_t const nrows = _stream_mats_new[m]->num_rows();
      cpd->dims[m] = nrows;

      cpd->factors[m] = (val_t *) splatt_malloc(nrows * _rank * sizeof(val_t));
      /* permute rows */
      #pragma omp parallel for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        idx_t const new_id = i;
        memcpy(&(cpd->factors[m][i*_rank]),
               &(_stream_mats_new[m]->vals()[new_id * _rank]),
               _rank * sizeof(val_t));
      }
    }
  }

  return cpd;
}


void StreamCPD::grow_mats(
    idx_t const * const new_dims)
{
  for(idx_t m=0; m < _nmodes; ++m) {
    if(m != _stream_mode) {
      idx_t const new_rows = new_dims[m];

      _stream_mats_new[m]->grow_rand(new_rows);
      _mat_ptrs[m] = _stream_mats_new[m]->mat();
      mat_aTa(_mat_ptrs[m], _cpd_ws->aTa[m]);

      _stream_mats_old[m]->grow_zero(new_rows);
      _stream_duals[m]->grow_zero(new_rows);

      _stream_auxil->grow_zero(new_rows);
      _stream_init->grow_zero(new_rows);
      _mttkrp_buf->grow_zero(new_rows);
    } else {
      _stream_duals[m]->grow_zero(1); /* we only need 1 row for time dual */
      _global_time->grow_zero(_global_time->num_rows() + 1);
    }

    _cpd_ws->duals[m] = _stream_duals[m]->mat();
  }
  _mat_ptrs[SPLATT_MAX_NMODES] = _mttkrp_buf->mat();
  _cpd_ws->mttkrp_buf = _mttkrp_buf->mat();
  _cpd_ws->auxil = _stream_auxil->mat();
  _cpd_ws->mat_init = _stream_init->mat();
}


void StreamCPD::add_historical(
    idx_t const mode)
{
  matrix_t * ata_buf = _cpd_ws->aTa_buf;

  /* 
   * Construct Gram matrix.
   */

  /* Time info -- make sure to copy upper triangular to lower */
  par_memcpy(ata_buf->vals, _old_gram->vals,
      _rank * _rank * sizeof(*ata_buf->vals));
  p_copy_upper_tri(ata_buf);

  matrix_t * _historical = mat_zero(_rank, _rank);

  /* other factors: old^T * new */
  /* TODO: timer */
  for(idx_t m=0; m < _nmodes; ++m) {
    if((m == mode) || (m == _stream_mode)) {
      continue;
    }
    assert(_stream_mats_new[m]->num_rows() == _stream_mats_old[m]->num_rows());

    splatt_blas_int const M = _rank;
    splatt_blas_int const N = _rank;
    splatt_blas_int const K = _stream_mats_new[m]->num_rows();
    splatt_blas_int const LDA = _rank;
    splatt_blas_int const LDB = _rank;
    splatt_blas_int const LDC = _rank;

    /* TODO: precision */
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        M, N, K,
        1.,
        _stream_mats_old[m]->vals(), LDA,
        _stream_mats_new[m]->vals(), LDB,
        0.,
        _historical->vals, _rank);

    /* incorporate into Gram */
    #pragma omp parallel for schedule(static)
    for(idx_t x=0; x < _rank * _rank; ++x) {
      ata_buf->vals[x] *= _historical->vals[x];
    }
  }


  /* 
   * mttkrp += old * aTa_buf
   */
  {
    splatt_blas_int const M = _stream_mats_new[mode]->num_rows();
    splatt_blas_int const N = _rank;
    splatt_blas_int const K = _rank;
    splatt_blas_int const LDA = _rank;
    splatt_blas_int const LDB = _rank;
    splatt_blas_int const LDC = _rank;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.,
        _stream_mats_old[mode]->vals(), LDA,
        ata_buf->vals, LDB,
        1.,
        _mttkrp_buf->vals(), _rank);
  }
  
  mat_free(_historical);
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


splatt_kruskal *  StreamCPD::compute(
    splatt_idx_t const rank,
    double const forget,
    splatt_cpd_opts * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const stream_mode = _source->stream_mode();
  idx_t const num_modes = _source->num_modes();

  /* TODO fix constructor */
  _stream_mode = stream_mode;
  _rank = rank;
  _nmodes = num_modes;

  /* register constraints */
  for(idx_t m=0; m < num_modes; ++m) {
    if(m != stream_mode) {
      splatt_register_maxcolnorm(cpd_opts, &m, 1);
    }
  }

  _cpd_ws = cpd_alloc_ws_empty(_nmodes, _rank, cpd_opts, global_opts);


  _global_time = new StreamMatrix(rank);
  _mttkrp_buf = new StreamMatrix(rank);
  _stream_auxil = new StreamMatrix(rank);
  _stream_init = new StreamMatrix(rank);
  for(idx_t m=0; m < num_modes; ++m) {
    _stream_mats_new[m] = new StreamMatrix(rank);
    _stream_mats_old[m] = new StreamMatrix(rank);
    _stream_duals[m] = new StreamMatrix(rank);
  }

  _mat_ptrs[stream_mode] = mat_zero(1, rank);

  /* Only previous info -- just used for add_historical() */
  _old_gram = mat_zero(rank, rank);

  printf("\n");

  /*
   * Stream
   */
  idx_t it = 0;
  timer_start(&timers[TIMER_CPD]);
  sptensor_t * batch = _source->next_batch();
  while(batch != NULL) {
    sp_timer_t batch_time;
    timer_fstart(&batch_time);

    grow_mats(batch->dims);
    /* normalize factors on the first batch */
    if(it == 0) {
      val_t * tmp = (val_t *) splatt_malloc(_rank * sizeof(*tmp));
      for(idx_t m=0; m < num_modes; ++m) {
        if(m == stream_mode) {
          continue;
        }
        mat_normalize(_mat_ptrs[m], tmp);
      }
      splatt_free(tmp);
    }

    /*
     * Compute new time slice.
     */
    timer_start(&timers[TIMER_MTTKRP]);
    _mat_ptrs[SPLATT_MAX_NMODES]->I = 1;
    mttkrp_stream(batch, _mat_ptrs, stream_mode);

    timer_stop(&timers[TIMER_MTTKRP]);
    admm(_stream_mode, _mat_ptrs, NULL, _cpd_ws, cpd_opts, global_opts);

    /* Accumulate new time slice into temporal Gram matrix */
    val_t       * const restrict ata_vals = _cpd_ws->aTa[stream_mode]->vals;
    val_t const * const restrict new_slice = _mat_ptrs[stream_mode]->vals;
    p_copy_upper_tri(_cpd_ws->aTa[stream_mode]);
    /* save old Gram matrix */
    par_memcpy(_old_gram->vals, ata_vals, rank * rank * sizeof(*ata_vals));
    timer_start(&timers[TIMER_ATA]);
    #pragma omp parallel for schedule(static) if(rank > 50)
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=0; j < rank; ++j) {
        ata_vals[j + (i*rank)] += new_slice[i] * new_slice[j];
      }
    }
    timer_stop(&timers[TIMER_ATA]);


    /*
     * Update the remaining modes
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m == stream_mode) {
        continue;
      }

      /* MTTKRP */
      timer_start(&timers[TIMER_MTTKRP]);
      _mat_ptrs[SPLATT_MAX_NMODES]->I = batch->dims[m];
      mttkrp_stream(batch, _mat_ptrs, m);
      timer_stop(&timers[TIMER_MTTKRP]);

      /* add historical data to MTTKRP */
      add_historical(m);

      /* Reset dual matrix. TODO: necessary? */
      memset(_cpd_ws->duals[m]->vals, 0,
          _cpd_ws->duals[m]->I * _rank * sizeof(val_t));
      admm(m, _mat_ptrs, NULL, _cpd_ws, cpd_opts, global_opts);

      /* lastly, update aTa */
      mat_aTa(_stream_mats_new[m]->mat(), _cpd_ws->aTa[m]);
    } /* foreach mode */

    /* Incorporate forgetting factor */
    for(idx_t x=0; x < _rank * _rank; ++x) {
      _cpd_ws->aTa[_stream_mode]->vals[x] *= forget;
    }

    /*
     * Copy new factors into old
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m == stream_mode) {
        continue;
      }
      par_memcpy(_stream_mats_old[m]->vals(), _stream_mats_new[m]->vals(),
          _stream_mats_new[m]->num_rows() * rank * sizeof(val_t));
    }

    /* save time vector */
    par_memcpy(&(_global_time->vals()[it*rank]),
      _mat_ptrs[stream_mode]->vals, rank * sizeof(val_t));

    /*
     * Batch stats
     */
    timer_stop(&batch_time);
#if 0
    printf("batch %5lu: %lu nnz (%0.3fs) (%0.3e NNZ/s)\n",
        it+1, batch->nnz, batch_time.seconds,
        (double) batch->nnz / batch_time.seconds);
#endif

#if 1
    splatt_kruskal * batch_cpd = get_kruskal();
    sptensor_t * until = _source->stream_until(it+1);
    double const global_err = cpd_error(until, batch_cpd);
    printf("  global-err: %0.5f\n", global_err * global_err);
    tt_free(until);
    splatt_free_cpd(batch_cpd);
#endif

    /* prepare for next batch */
    tt_free(batch);
    batch = _source->next_batch();
    ++it;
    /* XXX */
  } /* while batch != NULL */
  timer_stop(&timers[TIMER_CPD]);


  /* compute quality assessment */
  splatt_kruskal * cpd = get_kruskal();
  double const final_err = cpd_error(_source->full_stream(), cpd);
  printf("\n");
  printf("final-err: %0.5f\n",  final_err * final_err);

  mat_free(_old_gram);
  for(idx_t m=0; m < num_modes; ++m) {
    delete _stream_mats_new[m];
    delete _stream_mats_old[m];
  }
  mat_free(_mat_ptrs[stream_mode]);
  delete _mttkrp_buf;
  delete _stream_init;

  /* XXX */
  //splatt_cpd_free_ws(_cpd_ws);

  return cpd;
}




