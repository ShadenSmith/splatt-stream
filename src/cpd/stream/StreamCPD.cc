
#include "StreamCPD.hxx"
#include "StreamMatrix.hxx"

extern "C" {
#include "../admm.h"
#include "../../mttkrp.h"
#include "../../timer.h"
#include "../../util.h"
#include "../../stats.h"
}

#include <math.h>


#ifdef SPLATT_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif


/* How often to check global factorization error. This is expensive! */
#ifndef CHECK_ERR_INTERVAL
#define CHECK_ERR_INTERVAL 100
#endif

#ifndef USE_CSF
#define USE_CSF 0
#endif


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

void StreamCPD::track_row(
    idx_t mode,
    idx_t orig_index,
    char * name)
{
  idx_t const num_rows = _stream_mats_new[mode]->num_rows();

  /* lookup stocks */
  idx_t const row_id = _source->lookup_ind(mode, orig_index);
  printf("tracking %s: %lu of %lu\n", name, row_id, num_rows);
  if(num_rows >= row_id) {
    val_t const * const track_row = &(_stream_mats_new[mode]->vals()[_rank * row_id]);
    val_t const * const time_row = _mat_ptrs[_stream_mode]->vals;
    val_t track_norm = 0.;
    val_t time_norm  = 0.;
    val_t inner = 0.;
    for(idx_t f=0; f < _rank; ++f) {
      inner += track_row[f] * time_row[f];

      track_norm += track_row[f] * track_row[f];
      time_norm += time_row[f] * time_row[f];
    }
    printf("%s: %e (track: %e time: %e)\n", name, inner, track_norm, time_norm);
  }
}


double StreamCPD::compute_errorsq(
    idx_t num_previous)
{
  splatt_kruskal * cpd = get_prev_kruskal(num_previous);
  sptensor_t * prev_tensor = _source->stream_prev(num_previous);
  double const err = cpd_error(prev_tensor, cpd);
  tt_free(prev_tensor);
  splatt_free_cpd(cpd);
  return err * err;
}


double StreamCPD::compute_cpd_errorsq(
    idx_t num_previous)
{
  sptensor_t * prev_tensor = _source->stream_prev(num_previous);

  double * csf_opts = splatt_default_opts();
  splatt_csf * csf = splatt_csf_alloc(prev_tensor, csf_opts);
  splatt_free(prev_tensor);

  splatt_kruskal * newcpd = splatt_alloc_cpd(csf, _rank);
  splatt_cpd(csf, _rank, NULL, NULL, newcpd);
  splatt_free_csf(csf, csf_opts);

  double const err = 1 - newcpd->fit;

  splatt_free_cpd(newcpd);
  splatt_free_opts(csf_opts);
  return err;
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


splatt_kruskal * StreamCPD::get_prev_kruskal(idx_t previous)
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
      idx_t const nrows = SS_MIN(previous, _global_time->num_rows());
      idx_t const startrow = _global_time->num_rows() - nrows;

      cpd->dims[m] = nrows;
      cpd->factors[m] = (val_t *)
          splatt_malloc(nrows * _rank * sizeof(val_t));
      par_memcpy(cpd->factors[m], &(_global_time->vals()[startrow * _rank]),
          nrows * _rank * sizeof(val_t));

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
  timer_start(&timers[TIMER_MATMUL]);

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
  timer_stop(&timers[TIMER_MATMUL]);
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
      /* convert NTF to norm-constrained NTF */
      if(strcmp(cpd_opts->constraints[m]->description, "NON-NEGATIVE") == 0) {
        splatt_register_maxcolnorm_nonneg(cpd_opts, &m, 1);

      /* just column norm constraints */
      } else if(strcmp(cpd_opts->constraints[m]->description, "UNCONSTRAINED") == 0) {
        splatt_register_maxcolnorm(cpd_opts, &m, 1);
      }
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

#if USE_CSF == 1
  double * csf_opts = splatt_default_opts();
  csf_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ONEMODE;
  csf_opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
  csf_opts[SPLATT_OPTION_VERBOSITY] = SPLATT_VERBOSITY_NONE;
#endif

  cpd_opts->tolerance = 5e-2;
  cpd_opts->max_inner_iterations = 25;
  cpd_opts->inner_tolerance = 7e-2;
  cpd_stats2(_rank, _source->num_modes(), cpd_opts, global_opts);

  /*
   * Stream
   */
  sp_timer_t stream_time;
  timer_reset(&stream_time);
  idx_t it = 0;
  sptensor_t * batch = _source->next_batch();
  while(batch != NULL) {
    sp_timer_t batch_time;
    timer_start(&stream_time);
    timer_fstart(&batch_time);

#if USE_CSF == 1
    sp_timer_t csf_timer;
    timer_fstart(&csf_timer);
    splatt_csf * csf = splatt_csf_alloc(batch, csf_opts);
    splatt_mttkrp_ws * mttkrp_ws = splatt_mttkrp_alloc_ws(csf, _rank, csf_opts);
    timer_stop(&csf_timer);
    printf("    csf-alloc: %0.3fs\n", csf_timer.seconds);
#endif

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

    val_t prev_delta = 0.;
    for(idx_t outer=0; outer < cpd_opts->max_iterations; ++outer) {
      val_t delta = 0.;

      /*
       * Compute new time slice.
       */
      timer_start(&timers[TIMER_MTTKRP]);
      _mat_ptrs[SPLATT_MAX_NMODES]->I = 1;
#if USE_CSF == 1
      mttkrp_csf(csf, _mat_ptrs, stream_mode, _cpd_ws->thds, mttkrp_ws, global_opts);
#else
      mttkrp_stream(batch, _mat_ptrs, stream_mode);
#endif
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
#if USE_CSF == 1
        mttkrp_csf(csf, _mat_ptrs, m, _cpd_ws->thds, mttkrp_ws, global_opts);
#else
        mttkrp_stream(batch, _mat_ptrs, m);
#endif
        timer_stop(&timers[TIMER_MTTKRP]);

        /* add historical data to MTTKRP */
        add_historical(m);

        /* Reset dual matrix. TODO: necessary? */
        memset(_cpd_ws->duals[m]->vals, 0,
            _cpd_ws->duals[m]->I * _rank * sizeof(val_t));
        admm(m, _mat_ptrs, NULL, _cpd_ws, cpd_opts, global_opts);

        /* lastly, update aTa */
        mat_aTa(_stream_mats_new[m]->mat(), _cpd_ws->aTa[m]);

        delta +=
            mat_norm_diff(_stream_mats_old[m]->mat(), _stream_mats_new[m]->mat())
                / mat_norm(_stream_mats_new[m]->mat());
      } /* foreach mode */

      printf("  delta: %e prev_delta: %e (%e diff)\n", delta, prev_delta, fabs(delta - prev_delta));

      /* check convergence */
      if(outer > 0 && fabs(delta - prev_delta) < cpd_opts->tolerance) {
        printf("  converged in: %lu\n", outer+1);
        prev_delta = 0.;
        break;
      }
      prev_delta = delta;
    } /* foreach outer */

    /* Optional: track rows for data analysis. */
#if 0
    track_row(2, 1920, "stocks");
    track_row(3, 17825, "obama");
#endif

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
    timer_stop(&stream_time);
    ++it;


    double local_err   = compute_errorsq(1);
    double global_err  = -1.;
    double local10_err = -1.;
    double cpd_err     = -1.;
    if((it > 0) && ((it % CHECK_ERR_INTERVAL == 0) || _source->last_batch())) {
      global_err  = compute_errorsq(it);
      local10_err = compute_errorsq(10);
      cpd_err     = compute_cpd_errorsq(it);
      if(isnan(cpd_err)) {
        cpd_err = -1.;
      }
    }

    printf("batch %5lu: %7lu nnz (%0.5fs) (%0.3e NNZ/s) "
           "cpd: %+0.5f global: %+0.5f local-1: %+0.5f local-10: %+0.5f\n",
        it, batch->nnz, batch_time.seconds,
        (double) batch->nnz / batch_time.seconds,
        cpd_err, global_err, local_err, local10_err);

    /* prepare for next batch */
    tt_free(batch);
#if USE_CSF == 1
    splatt_free_csf(csf, csf_opts);
    splatt_mttkrp_free_ws(mttkrp_ws);
#endif
    batch = _source->next_batch();
    /* XXX */
  } /* while batch != NULL */

  printf("stream-time: %0.3fs\n", stream_time.seconds);

#if USE_CSF == 1
  splatt_free_opts(csf_opts);
#endif

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


