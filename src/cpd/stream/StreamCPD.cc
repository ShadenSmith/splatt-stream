
#include "StreamCPD.hxx"
#include "StreamMatrix.hxx"

extern "C" {
#include "../../mttkrp.h"
#include "../../timer.h"
#include "../../io.h" /* XXX debug */
#include "../../util.h"

#include <cblas.h>
}

static void p_setup_stream_RHS(
    matrix_t * mttkrp_buf, /* output */
    StreamMatrix * * stream_mats_new,
    StreamMatrix * * stream_mats_old,
    idx_t stream_mode,
    idx_t curr_mode,
    idx_t nmodes,
    matrix_t * rTr)
{
  assert(rTr->I == rTr->J);
  idx_t const rank = rTr->I;

  matrix_t * ata_buf = mat_alloc(rank, rank); /* accumulate into */
  matrix_t * qtb_buf = mat_alloc(rank, rank); /* Q^T * B, etc. */

  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < rank * rank; ++x) {
    ata_buf->vals[x] = 1.;
  }

  /* build RHS */
  for(idx_t m=0; m < nmodes; ++m) {
    if((m == stream_mode) || (m == curr_mode)) {
      continue;
    }
    assert(stream_mats_new[m]->num_rows() == stream_mats_old[m]->num_rows());

    splatt_blas_int const M = rank;
    splatt_blas_int const N = rank;
    splatt_blas_int const K = stream_mats_new[m]->num_rows();
    splatt_blas_int const LDA = rank;
    splatt_blas_int const LDB = rank;
    splatt_blas_int const LDC = rank;

    /* Q^T * B */
    cblas_dgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        M, N, K,
        1.,
        stream_mats_old[m]->vals(), LDA,
        stream_mats_new[m]->vals(), LDB,
        0.,
        qtb_buf->vals, rank);

    #pragma omp parallel for schedule(static)
    for(idx_t x=0; x < rank * rank; ++x) {
      ata_buf->vals[x] *= qtb_buf->vals[x];
    }
  }

  /* multiply rTr into aTa_buf, and reflect upper to lower triangular*/
  for(idx_t i=0; i < rank; ++i) {
    for(idx_t j=0; j < i; ++j) {
      ata_buf->vals[j + (i*rank)] *= rTr->vals[i + (j*rank)];
    }
    for(idx_t j=i; j < rank; ++j) {
      ata_buf->vals[j + (i*rank)] *= rTr->vals[j + (i*rank)];
    }
  }

  /* mttkrp += P * aTa_buf */
  {
    splatt_blas_int const M = mttkrp_buf->I;
    splatt_blas_int const N = rank;
    splatt_blas_int const K = rank;
    splatt_blas_int const LDA = K;
    splatt_blas_int const LDB = N;
    splatt_blas_int const LDC = N;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1.,
        stream_mats_new[curr_mode]->vals(), LDA,
        ata_buf->vals, LDB,
        1.,
        mttkrp_buf->vals, rank);
  }

  mat_free(ata_buf);
  mat_free(qtb_buf);
}

static void p_setup_gram(
    matrix_t * * aTa,
    matrix_t * new_gram,
    matrix_t * out_mat, 
    idx_t num_modes,
    idx_t stream_mode,
    idx_t curr_mode)
{
  idx_t const N = aTa[curr_mode]->J;
  val_t * const restrict gram = out_mat->vals;

  val_t const * const restrict ngram = new_gram->vals;
  val_t const * const restrict ogram = aTa[stream_mode]->vals;

  #pragma omp parallel
  {
    /* first initialize with time data */
    #pragma omp for schedule(static, 1)
    for(idx_t i=0; i < N; ++i) {
      for(idx_t j=i; j < N; ++j) {
        gram[j+(i*N)] = ngram[j+(i*N)] + ogram[j+(i*N)];
      }
    }

    for(idx_t m=0; m < num_modes; ++m) {
      if((m == stream_mode) || (m == curr_mode)) {
        continue;
      }

			/* only work with upper triangular */
      val_t const * const restrict mat = aTa[m]->vals;
      #pragma omp for schedule(static, 1) nowait
      for(idx_t i=0; i < N; ++i) {
        for(idx_t j=i; j < N; ++j) {
          gram[j+(i*N)] *= mat[j+(i*N)];
        }
      }
    }
  } /* omp parallel */
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
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const stream_mode = _source->stream_mode();
  idx_t const num_modes = _source->num_modes();

  StreamMatrix time_mat(rank);
  StreamMatrix * stream_mats_new[SPLATT_MAX_NMODES];
  StreamMatrix * stream_mats_old[SPLATT_MAX_NMODES];
  for(idx_t m=0; m < num_modes; ++m) {
    stream_mats_new[m] = new StreamMatrix(rank);
    stream_mats_old[m] = new StreamMatrix(rank);
  }

  matrix_t * mat_ptrs[SPLATT_MAX_NMODES+1];
  mat_ptrs[stream_mode] = mat_zero(1, rank);

  /* Gram matrices */
  matrix_t * gram     = mat_zero(rank, rank); /* old time slices: R^T * R */
  matrix_t * new_gram = mat_zero(rank, rank); /* new time slice:  c^T * c */
  matrix_t * aTa[SPLATT_MAX_NMODES];
  for(idx_t m=0; m < num_modes; ++m) {
    aTa[m] = mat_zero(rank, rank);

    /* initialize streaming aTa with identity */
    if(m == stream_mode) {
      for(idx_t x=0; x < rank * rank; ++x) {
        aTa[m]->vals[x] = 1.;
      }
    }
  }

  StreamMatrix mttkrp_buf(rank);

  /*
   * Stream
   */
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
        /* XXX which ?*/
#if 1
        stream_mats_old[m]->grow_zero(batch->dims[m]);
#else
        stream_mats_old[m]->grow_rand(batch->dims[m]);
#endif
        stream_mats_new[m]->grow_rand(batch->dims[m]);
        mat_ptrs[m] = stream_mats_new[m]->mat();
        mat_aTa(stream_mats_new[m]->mat(), aTa[m]);
      }
    }

    /* inner ALS iterations */
    for(idx_t inner_it=0; inner_it < 1; ++inner_it) {
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
      mat_aTa(mat_ptrs[stream_mode], new_gram); /* this slice's Gram */

      /*
       * Update the remaining modes
       */
      for(idx_t m=0; m < num_modes; ++m) {
        if(m == stream_mode) {
          continue;
        }

        /* MTTKRP */
        mttkrp_buf.grow_zero(batch->dims[m]);
        timer_start(&timers[TIMER_MTTKRP]);
        mat_ptrs[SPLATT_MAX_NMODES] = mttkrp_buf.mat();
        mat_ptrs[SPLATT_MAX_NMODES]->I = batch->dims[m];
        mttkrp_stream(batch, mat_ptrs, m);
        timer_stop(&timers[TIMER_MTTKRP]);

        /* aTa setup */
        p_setup_stream_RHS(
            mttkrp_buf.mat(),
            stream_mats_new, stream_mats_old,
            stream_mode, m, num_modes,
            aTa[stream_mode]);

        /* Gram setup */
        p_setup_gram(aTa, new_gram, gram, num_modes, stream_mode, m);
#if 0
        /* regularize */
        val_t const norm = mat_norm(gram);
        mat_add_diag(gram, norm * norm / rank);
#endif
        mat_cholesky(gram);
        mat_solve_cholesky(gram, mttkrp_buf.mat());

        /* Copy output to factor matrix. NOTE: mttkrp_buf.num_rows() may be
         * larger than stream_mats_new[m]->num_rows() due to other modes,
         * so use the latter.*/
        assert(stream_mats_new[m]->num_rows() <= mttkrp_buf.num_rows());
        par_memcpy(stream_mats_new[m]->vals(), mttkrp_buf.vals(),
            stream_mats_new[m]->num_rows() * rank * sizeof(val_t));

        /* lastly, update aTa */
        mat_aTa(stream_mats_new[m]->mat(), aTa[m]);
      }
    } /* inner its */


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
     * Copy new factors into old
     */
    for(idx_t m=0; m < num_modes; ++m) {
      if(m == stream_mode) {
        continue;
      }
      par_memcpy(stream_mats_old[m]->vals(), stream_mats_new[m]->vals(),
          stream_mats_new[m]->num_rows() * rank * sizeof(val_t));
    }

    /* save time vector */
    time_mat.grow_zero(it+1);
    par_memcpy(&(time_mat.vals()[it*rank]),
      mat_ptrs[stream_mode]->vals, rank * sizeof(val_t));

    /*
     * Batch stats
     */
    timer_stop(&batch_time);
#if 1
    printf("batch %5lu: %lu nnz (%0.3fs) (%0.3e NNZ/s)\n",
        it+1, batch->nnz, batch_time.seconds,
        (double) batch->nnz / batch_time.seconds);
#endif


    /* prepare for next batch */
    tt_free(batch);
    batch = _source->next_batch();
    ++it;
    /* XXX */
  } /* while batch != NULL */
  timer_stop(&timers[TIMER_CPD]);


  /* store output */
  splatt_kruskal * cpd = (splatt_kruskal *) splatt_malloc(sizeof(*cpd));
  cpd->nmodes = num_modes;
  cpd->lambda = (val_t *) splatt_malloc(rank * sizeof(*cpd->lambda));
  cpd->rank = rank;
  for(idx_t r=0; r < rank; ++r) {
    cpd->lambda[r] = 1.;
  }
  for(idx_t m=0; m < num_modes; ++m) {

    if(m == stream_mode) {
      cpd->dims[m] = it;
      cpd->factors[m] = (val_t *)
          splatt_malloc(it * rank * sizeof(val_t));

      par_memcpy(cpd->factors[m], time_mat.vals(), it * rank * sizeof(val_t));
    } else {
      idx_t const nrows = stream_mats_new[m]->num_rows();
      cpd->dims[m] = nrows;

      cpd->factors[m] = (val_t *) splatt_malloc(nrows * rank * sizeof(val_t));
      /* permute rows */
      #pragma omp parallel for schedule(static)
      for(idx_t i=0; i < nrows; ++i) {
        idx_t const new_id = _source->lookup_ind(m, i);
        memcpy(&(cpd->factors[m][i*rank]), 
               &(stream_mats_new[m]->vals()[new_id * rank]),
               rank * sizeof(val_t));
      }
    }
  }

  /* compute quality assessment */
  val_t const Xnorm = tt_normsq(_source->full_stream());
  val_t const mynorm = kruskal_norm(cpd);

  printf("Xnorm: %e mynorm: %e\n", Xnorm, mynorm);

  mat_free(gram);
  mat_free(new_gram);
  for(idx_t m=0; m < num_modes; ++m) {
    delete stream_mats_new[m];
    delete stream_mats_old[m];
    mat_free(aTa[m]);
  }
  mat_free(mat_ptrs[stream_mode]);

  return cpd;
}




