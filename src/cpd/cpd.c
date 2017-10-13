/**
* @file cpd.c
* @brief Tensor factorization with the CPD model using AO-ADMM.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-14
*/




/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include <math.h>

#include "cpd.h"
#include "admm.h"

#include "../csf.h"
#include "../sptensor.h"
#include "../mttkrp.h"
#include "../timer.h"
#include "../util.h"




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/





/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/


splatt_error_type splatt_cpd(
    splatt_csf const * const tensor,
    splatt_idx_t rank,
    splatt_cpd_opts const * const cpd_options,
    splatt_global_opts const * const global_options,
    splatt_kruskal * factored)
{
  /* allocate default options if they were not supplied */
  splatt_global_opts * global_opts = (splatt_global_opts *) global_options;
  if(global_options == NULL) {
    global_opts = splatt_alloc_global_opts();
  }
  splatt_cpd_opts * cpd_opts = (splatt_cpd_opts *) cpd_options;
  if(cpd_options == NULL) {
    cpd_opts = splatt_alloc_cpd_opts();
  }

  splatt_omp_set_num_threads(global_opts->num_threads);

  /* allocate workspace */
  cpd_ws * ws = cpd_alloc_ws(tensor, rank, cpd_opts, global_opts);

  /* perform the factorization! */
  cpd_iterate(tensor, rank, ws, cpd_opts, global_opts, factored);

  /* clean up workspace */
  cpd_free_ws(ws);

  /* free options if we had to allocate them */
  if(global_options == NULL) {
    splatt_free_global_opts(global_opts);
  }
  if(cpd_options == NULL) {
    splatt_free_cpd_opts(cpd_opts);
  }

  return SPLATT_SUCCESS;
}



splatt_cpd_opts * splatt_alloc_cpd_opts(void)
{
  splatt_cpd_opts * opts = splatt_malloc(sizeof(*opts));

  /* defaults */
  opts->tolerance = 1e-5;
  opts->max_iterations = 200;

  opts->inner_tolerance = 1e-2;
  opts->max_inner_iterations = 50;

  for(idx_t m=0; m < MAX_NMODES; ++m) {
    opts->chunk_sizes[m] = 50;
    opts->constraints[m] = splatt_alloc_constraint(SPLATT_CON_CLOSEDFORM);
  }

  return opts;
}


void splatt_free_cpd_opts(
    splatt_cpd_opts * opts)
{
  /* free constraints */
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    splatt_free_constraint(opts->constraints[m]);
  }

  /* free options pointer */
  splatt_free(opts);
}


splatt_kruskal * splatt_alloc_cpd(
    splatt_csf const * const csf,
    splatt_idx_t rank)
{
  splatt_kruskal * cpd = splatt_malloc(sizeof(*cpd));

  cpd->nmodes = csf->nmodes;
  cpd->rank = rank;

  cpd->lambda = splatt_malloc(rank * sizeof(*cpd->lambda));
  for(idx_t m=0; m < csf->nmodes; ++m) {
    cpd->dims[m] = csf->dims[m];
    cpd->factors[m] = splatt_malloc(csf->dims[m] * rank *
        sizeof(**cpd->factors));

    /* TODO: allow custom initialization including NUMA aware */
    fill_rand(cpd->factors[m], csf->dims[m] * rank);
  }

  /* initialize lambda in case it is not modified */
  for(idx_t r=0; r < rank; ++r) {
    cpd->lambda[r] = 1.;
  }

  return cpd;
}


void splatt_free_cpd(
    splatt_kruskal * factored)
{
  splatt_free(factored->lambda);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    splatt_free(factored->factors[m]);
  }
  splatt_free(factored);
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


double cpd_iterate(
    splatt_csf const * const tensor,
    idx_t const rank,
    cpd_ws * const ws,
    splatt_cpd_opts * const cpd_opts,
    splatt_global_opts const * const global_opts,
    splatt_kruskal * factored)
{
  idx_t const nmodes = tensor->nmodes;

  /* XXX: fix MTTKRP interface */
  /*
   * The matrices used for MTTKRP. When using MPI, these may be larger than
   * the mats[:] matrices due to non-local indices. If the sizes are the same,
   * these are just aliases for mats[:].
   */
  matrix_t * mats[MAX_NMODES+1];
  matrix_t * mttkrp_mats[MAX_NMODES+1];
  for(idx_t m=0; m < tensor->nmodes; ++m) {
    mats[m] = mat_mkptr(factored->factors[m], tensor->dims[m], rank, 1);
#ifdef SPLATT_USE_MPI
    /* setup local MTTKRP matrices */
#else
    mttkrp_mats[m] = mats[m];
#endif

    mat_normalize(mats[m], factored->lambda);
  }
  mats[MAX_NMODES] = ws->mttkrp_buf;
  mttkrp_mats[MAX_NMODES] = ws->mttkrp_buf;

  /* allow constraints to initialize */
  cpd_init_constraints(cpd_opts, mats, nmodes);

  /* reset column weights */
  val_t * const restrict norms = factored->lambda;
  for(idx_t r=0; r < rank; ++r) {
    norms[r] = 1.;
  }

  /* initialite aTa values */
  for(idx_t m=1; m < nmodes; ++m) {
    mat_aTa(mats[m], ws->aTa[m]);
  }

  /* XXX TODO: CSF opts */
  double * opts = splatt_default_opts();

  /* MTTKRP ws */
  splatt_mttkrp_ws * mttkrp_ws = splatt_mttkrp_alloc_ws(tensor, rank, opts);

  /* for tracking convergence */
  double olderr = 1.;
  double err = 0.;
  double const ttnormsq = csf_frobsq(tensor);

  /* timers */
  sp_timer_t itertime;
  sp_timer_t modetime[MAX_NMODES];
  timer_start(&timers[TIMER_CPD]);

  val_t inner_its[MAX_NMODES];

  /* foreach outer iteration */
  for(idx_t it=0; it < cpd_opts->max_iterations; ++it) {
    timer_fstart(&itertime);
    /* foreach AO step */
    for(idx_t m=0; m < nmodes; ++m) {
      timer_fstart(&modetime[m]);
      mttkrp_csf(tensor, mttkrp_mats, m, ws->thds, mttkrp_ws, global_opts);
#ifdef SPLATT_USE_MPI
      /* exchange partial MTTKRP results */
#endif

      /* ADMM solve for constraints */
      inner_its[m] = admm(m, mats, norms, ws, cpd_opts, global_opts);
#ifdef SPLATT_USE_MPI
      /* exchange updated factor rows */
#endif

      /* prepare aTa for next mode */
#ifdef SPLATT_USE_MPI
      /* XXX use real comm */
      mat_aTa_mpi(mats[m], ws->aTa[m], MPI_COMM_WORLD);
#else
      mat_aTa(mats[m], ws->aTa[m]);
#endif

      timer_stop(&modetime[m]);
    } /* foreach mode */

    /* calculate outer convergence */
    double const norm = cpd_norm(ws, norms);
    double const inner = cpd_innerprod(nmodes-1, ws, mats, norms);
    double const residual = ttnormsq + norm - (2 * inner);
    err = residual / ttnormsq;

    assert(err <= olderr);
    timer_stop(&itertime);

    /* print progress */
    if(global_opts->verbosity > SPLATT_VERBOSITY_NONE) {
      printf("  its = %4"SPLATT_PF_IDX" (%0.3"SPLATT_PF_VAL"s)  "
             "rel-errsq = %0.5"SPLATT_PF_VAL"  delta = %+0.4e\n",
          it+1, itertime.seconds, err, err - olderr);
      if(global_opts->verbosity > SPLATT_VERBOSITY_LOW) {
        for(idx_t m=0; m < nmodes; ++m) {
          printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs)",
              m+1, modetime[m].seconds);
          if(inner_its[m] > 0) {
            printf(" [%4.1"SPLATT_PF_VAL" ADMM its per row]", inner_its[m]);
          }
          printf("\n");
        }
      }
    }

    /* terminate if converged */
    if(it > 0 && fabs(olderr - err) < cpd_opts->tolerance) {
      break;
    }
    olderr = err;
  }

  /* absorb into lambda if no constraints/regularizations */
  if(ws->unconstrained) {
    cpd_post_process(mats, norms, ws, cpd_opts, global_opts);
  } else {
    cpd_finalize_constraints(cpd_opts, mats, nmodes);
  }

  splatt_free(opts);
  for(idx_t m=0; m < tensor->nmodes; ++m) {
    /* free matrix memory if not an alias */
    if(mttkrp_mats[m] != mats[m]) {
      mat_free(mttkrp_mats[m]);
    }

    /* only free ptr */
    splatt_free(mats[m]);
  }

  splatt_mttkrp_free_ws(mttkrp_ws);

  timer_stop(&timers[TIMER_CPD]);

  factored->fit = 1 - err;
  return err;
}



void cpd_post_process(
    matrix_t * * mats,
    val_t * const column_weights,
    cpd_ws * const ws,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const rank = mats[0]->J;
  val_t * tmp =  splatt_malloc(rank * sizeof(*tmp));

  /* normalize each matrix and adjust lambda */
  for(idx_t m=0; m < ws->nmodes; ++m) {
    mat_normalize(mats[m], tmp);
    for(idx_t f=0; f < rank; ++f) {
      column_weights[f] *= tmp[f];
    }
  }

  splatt_free(tmp);
}



cpd_ws * cpd_alloc_ws(
    splatt_csf const * const tensor,
    idx_t rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  idx_t const nmodes = tensor->nmodes;

  cpd_ws * ws = splatt_malloc(sizeof(*ws));

  ws->nmodes = nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    ws->aTa[m] = mat_alloc(rank, rank);
  }
  ws->aTa_buf = mat_alloc(rank, rank);
  ws->gram = mat_alloc(rank, rank);

  ws->nthreads = global_opts->num_threads;
  ws->thds =  thd_init(ws->nthreads, 3,
    (rank * rank * sizeof(val_t)) + 64,
    0,
    (nmodes * rank * sizeof(val_t)) + 64);

  /* MTTKRP space */
  idx_t const maxdim = tensor->dims[argmax_elem(tensor->dims, nmodes)];
  ws->mttkrp_buf = mat_alloc(maxdim, rank);

  /* Setup structures needed for constraints. */
  ws->unconstrained = true;
  for(idx_t m=0; m < nmodes; ++m) {
    /* allocate duals if we need to perform ADMM */
    if(cpd_opts->constraints[m]->solve_type != SPLATT_CON_CLOSEDFORM) {
      ws->duals[m] = mat_zero(tensor->dims[m], rank);
    } else {
      ws->duals[m] = NULL;
    }

    if(strcmp(cpd_opts->constraints[m]->description, "UNCONSTRAINED") != 0) {
      ws->unconstrained = false;
    }
  }

  if(ws->unconstrained) {
    ws->auxil    = NULL;
    ws->mat_init = NULL;
  } else {
    ws->auxil    = mat_alloc(maxdim, rank);
    ws->mat_init = mat_alloc(maxdim, rank);
  }

  return ws;
}



cpd_ws * cpd_alloc_ws_empty(
    idx_t const nmodes,
    idx_t const rank,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  cpd_ws * ws = splatt_malloc(sizeof(*ws));

  ws->nmodes = nmodes;
  for(idx_t m=0; m < nmodes; ++m) {
    ws->aTa[m] = mat_zero(rank, rank);
  }
  ws->aTa_buf = mat_zero(rank, rank);
  ws->gram = mat_zero(rank, rank);

  ws->nthreads = global_opts->num_threads;
  ws->thds =  thd_init(ws->nthreads, 3,
    (rank * rank * sizeof(val_t)) + 64,
    0,
    (nmodes * rank * sizeof(val_t)) + 64);

  /* MTTKRP space */
  ws->mttkrp_buf = NULL;

  /* Setup structures needed for constraints. */
  ws->unconstrained = true;
  for(idx_t m=0; m < nmodes; ++m) {
    ws->duals[m] = NULL;

    if(strcmp(cpd_opts->constraints[m]->description, "UNCONSTRAINED") != 0) {
      ws->unconstrained = false;
    }
  }

  ws->auxil    = NULL;
  ws->mat_init = NULL;

  return ws;
}



void cpd_free_ws(
    cpd_ws * const ws)
{
  mat_free(ws->mttkrp_buf);
  mat_free(ws->aTa_buf);
  mat_free(ws->gram);

  mat_free(ws->auxil);
  mat_free(ws->mat_init);

  for(idx_t m=0; m < ws->nmodes; ++m) {
    mat_free(ws->aTa[m]);
    mat_free(ws->duals[m]);
  }

  thd_free(ws->thds, ws->nthreads);
  splatt_free(ws);
}


val_t cpd_norm(
    cpd_ws const * const ws,
    val_t const * const restrict column_weights)
{
  idx_t const rank = ws->aTa[0]->J;
  val_t * const restrict scratch = ws->aTa_buf->vals;

  /* initialize scratch space */
  for(idx_t i=0; i < rank; ++i) {
    for(idx_t j=i; j < rank; ++j) {
      scratch[j + (i*rank)] = 1.;
    }
  }

  /* scratch = hada(aTa) */
  for(idx_t m=0; m < ws->nmodes; ++m) {
    val_t const * const restrict atavals = ws->aTa[m]->vals;
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=i; j < rank; ++j) {
        scratch[j + (i*rank)] *= atavals[j + (i*rank)];
      }
    }
  }

  /* now compute weights^T * aTa[MAX_NMODES] * weights */
  val_t norm = 0;
  for(idx_t i=0; i < rank; ++i) {
    norm += scratch[i+(i*rank)] * column_weights[i] * column_weights[i];
    for(idx_t j=i+1; j < rank; ++j) {
      norm += scratch[j+(i*rank)] * column_weights[i] * column_weights[j] * 2;
    }
  }

  return fabs(norm);
}


val_t cpd_innerprod(
    idx_t lastmode,
    cpd_ws const * const ws,
    matrix_t * * mats,
    val_t const * const restrict column_weights)
{
  idx_t const nrows = mats[lastmode]->I;
  idx_t const rank = mats[0]->J;

  val_t const * const newmat = mats[lastmode]->vals;
  val_t const * const mttkrp = ws->mttkrp_buf->vals;

  val_t myinner = 0;
  #pragma omp parallel reduction(+:myinner)
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = ws->thds[tid].scratch[0];

    for(idx_t r=0; r < rank; ++r) {
      accumF[r] = 0.;
    }

    /* Hadamard product with newest factor and previous MTTKRP */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < nrows; ++i) {
      val_t const * const restrict newmat_row = newmat + (i*rank);
      val_t const * const restrict mttkrp_row = mttkrp + (i*rank);
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] += newmat_row[r] * mttkrp_row[r];
      }
    }

    /* accumulate everything into 'myinner' */
    for(idx_t r=0; r < rank; ++r) {
      myinner += accumF[r] * column_weights[r];
    }
  } /* end omp parallel -- reduce myinner */

  /* TODO AllReduce for MPI support */

  return myinner;
}

val_t kruskal_norm(
    splatt_kruskal const * const kruskal)
{
  idx_t const rank = kruskal->rank;
  val_t * const scratch = (val_t *) splatt_malloc(rank * rank * sizeof(*scratch));

  matrix_t * ata = mat_zero(rank, rank);

  /* initialize scratch space */
  for(idx_t i=0; i < rank; ++i) {
    for(idx_t j=i; j < rank; ++j) {
      scratch[j + (i*rank)] = 1.;
    }
  }

  /* scratch = hada(aTa) */
  for(idx_t m=0; m < kruskal->nmodes; ++m) {
    matrix_t matptr;
    mat_fillptr(&matptr, kruskal->factors[m], kruskal->dims[m], rank, 1);

    mat_aTa(&matptr, ata);

    val_t const * const restrict atavals = ata->vals;
    for(idx_t i=0; i < rank; ++i) {
      for(idx_t j=i; j < rank; ++j) {
        scratch[j + (i*rank)] *= atavals[j + (i*rank)];
      }
    }
  }

  /* now compute weights^T * aTa[MAX_NMODES] * weights */
  val_t norm = 0;
  val_t const * const column_weights = kruskal->lambda;
  for(idx_t i=0; i < rank; ++i) {
    norm += scratch[i+(i*rank)] * column_weights[i] * column_weights[i];
    for(idx_t j=i+1; j < rank; ++j) {
      norm += scratch[j+(i*rank)] * column_weights[i] * column_weights[j] * 2;
    }
  }

  splatt_free(scratch);
  mat_free(ata);

  return fabs(norm);
}


double cpd_error(
    sptensor_t const * const tensor,
    splatt_kruskal const * const factored)
{
  timer_start(&timers[TIMER_FIT]);

  /* find the smallest mode for MTTKRP */
  idx_t const smallest_mode = argmin_elem(tensor->dims, tensor->nmodes);
  idx_t const nrows = tensor->dims[smallest_mode];
  idx_t const rank = factored->rank;

  /*
   * MTTKRP
   */
  matrix_t * mat_ptrs[MAX_NMODES+1];
  for(idx_t m=0; m < factored->nmodes; ++m) {
    mat_ptrs[m] = mat_mkptr(factored->factors[m], factored->dims[m], rank, 1);
  }
  mat_ptrs[MAX_NMODES] = mat_alloc(nrows, rank);
  mttkrp_stream(tensor, mat_ptrs, smallest_mode);
  
  val_t const * const smallmat = factored->factors[smallest_mode];
  val_t const * const mttkrp  = mat_ptrs[MAX_NMODES]->vals;


  /*
   * inner product between tensor and factored
   */
  double inner = 0;
  #pragma omp parallel reduction(+:inner)
  {
    int const tid = splatt_omp_get_thread_num();
    val_t * const restrict accumF = splatt_malloc(rank * sizeof(*accumF));

    for(idx_t r=0; r < rank; ++r) {
      accumF[r] = 0.;
    }

    /* Hadamard product with newest factor and previous MTTKRP */
    #pragma omp for schedule(static)
    for(idx_t i=0; i < nrows; ++i) {
      val_t const * const restrict smallmat_row = smallmat + (i*rank);
      val_t const * const restrict mttkrp_row = mttkrp + (i*rank);
      for(idx_t r=0; r < rank; ++r) {
        accumF[r] += smallmat_row[r] * mttkrp_row[r];
      }
    }

    /* accumulate everything into 'inner' */
    for(idx_t r=0; r < rank; ++r) {
      inner += accumF[r] * factored->lambda[r];
    }

    splatt_free(accumF);
  } /* end omp parallel -- reduce myinner */


  double const Xnormsq = tt_normsq(tensor);
  double const Znormsq = kruskal_norm(factored);
  double const residual = sqrt(Xnormsq + Znormsq - (2 * inner));
  double const err = residual / sqrt(Xnormsq);

#if 0
  printf("\n");
  printf("Xnormsq: %e Znormsq: %e inner: %e\n", Xnormsq, Znormsq, inner);
#endif

  /* cleanup */
  mat_free(mat_ptrs[MAX_NMODES]);
  for(idx_t m=0; m < factored->nmodes; ++m) {
    /* just the ptr */
    splatt_free(mat_ptrs[m]);
  }

  timer_stop(&timers[TIMER_FIT]);
  return err;
}




