
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "csf.h"
#include "sort.h"
#include "tile.h"

#include "io.h"

/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void __order_dims_small(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  quicksort(sorted, nmodes);

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }
}


static void __order_dims_minusone(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t const custom_mode,
  idx_t * const perm_dims)
{
  __order_dims_small(dims, nmodes, perm_dims);

  /* find where custom_mode was placed and adjust from there */
  for(idx_t m=0; m < nmodes; ++m) {
    if(perm_dims[m] == custom_mode) {
      memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
      perm_dims[0] = custom_mode;
      break;
    }
  }
}


static void __order_dims_large(
  idx_t const * const dims,
  idx_t const nmodes,
  idx_t * const perm_dims)
{
  idx_t sorted[MAX_NMODES];
  idx_t matched[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    sorted[m] = dims[m];
    matched[m] = 0;
  }
  /* sort small -> large */
  quicksort(sorted, nmodes);

  /* reverse list */
  for(idx_t m=0; m < nmodes/2; ++m) {
    idx_t tmp = sorted[nmodes-m-1];
    sorted[nmodes-m-1] = sorted[m];
    sorted[m] = tmp;
  }

  /* silly n^2 comparison to grab modes from sorted dimensions.
   * TODO: make a key/val sort...*/
  for(idx_t mfind=0; mfind < nmodes; ++mfind) {
    for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
      if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
        perm_dims[mfind] = mcheck;
        matched[mcheck] = 1;
        break;
      }
    }
  }

}


static void __print_csf(
  splatt_csf const * const ct)
{
  printf("-----------\n");
  printf("nmodes: %"SPLATT_PF_IDX" nnz: %"SPLATT_PF_IDX" ntiles: "
         "%"SPLATT_PF_IDX"\n", ct->nmodes, ct->nnz, ct->ntiles);
  printf("dims: %"SPLATT_PF_IDX"", ct->dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->dims[m]);
  }
  printf(" (%"SPLATT_PF_IDX"", ct->dim_perm[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("->%"SPLATT_PF_IDX"", ct->dim_perm[m]);
  }
  printf(") ");
  printf("tile dims: %"SPLATT_PF_IDX"", ct->tile_dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->tile_dims[m]);
  }
  printf("\n");

  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity const * const ft = ct->pt + t;
    /* skip empty tiles */
    if(ft->vals == NULL) {
      continue;
    }

    /* write slices */
    printf("tile: %"SPLATT_PF_IDX" fptr:\n", t);
    printf("[%"SPLATT_PF_IDX"] ", ft->nfibs[0]);
    for(idx_t f=0; f < ft->nfibs[0]; ++f) {
      if(ft->fids[0] == NULL) {
        printf(" %"SPLATT_PF_IDX"", ft->fptr[0][f]);
      } else {
        printf(" (%"SPLATT_PF_IDX", %"SPLATT_PF_IDX")", ft->fptr[0][f],
            ft->fids[0][f]);
      }
    }
    printf(" %"SPLATT_PF_IDX"\n", ft->fptr[0][ft->nfibs[0]]);

    /* inner nodes */
    for(idx_t m=1; m < ct->nmodes-1; ++m) {
      printf("[%"SPLATT_PF_IDX"] ", ft->nfibs[m]);
      for(idx_t f=0; f < ft->nfibs[m]; ++f) {
        printf(" (%"SPLATT_PF_IDX", %"SPLATT_PF_IDX")", ft->fptr[m][f],
            ft->fids[m][f]);
      }
      printf(" %"SPLATT_PF_IDX"\n", ft->fptr[m][ft->nfibs[m]]);
    }

    /* vals/inds */
    printf("[%"SPLATT_PF_IDX"] ", ft->nfibs[ct->nmodes-1]);
    for(idx_t f=0; f < ft->nfibs[ct->nmodes-1]; ++f) {
      printf(" %3"SPLATT_PF_IDX"", ft->fids[ct->nmodes-1][f]);
    }
    printf("\n");
    for(idx_t n=0; n < ft->nfibs[ct->nmodes-1]; ++n) {
      printf(" %0.1f", ft->vals[n]);
    }
    printf("\n");
  }

  printf("-----------\n\n");
}


static void __mk_outerptr(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr)
{
  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[0]] + nnzstart;

  /* count fibers */
  idx_t nfibs = 1;
  for(idx_t x=1; x < nnz; ++x) {
    assert(ttind[x-1] <= ttind[x]);
    if(ttind[x] != ttind[x-1]) {
      ++nfibs;
    }
  }
  ct->pt[tile_id].nfibs[0] = nfibs;
  assert(nfibs <= ct->dims[ct->dim_perm[0]]);

  /* grab sparsity pattern */
  csf_sparsity * const pt = ct->pt + tile_id;

  pt->fptr[0] = (idx_t *) malloc((nfibs+1) * sizeof(idx_t));
  if(ct->ntiles > 1) {
    pt->fids[0] = (idx_t *) malloc(nfibs * sizeof(idx_t));
  } else {
    pt->fids[0] = NULL;
  }

  idx_t  * const restrict fp = pt->fptr[0];
  idx_t  * const restrict fi = pt->fids[0];
  fp[0] = 0;
  if(fi != NULL) {
    fi[0] = ttind[0];
  }

  idx_t nfound = 1;
  for(idx_t n=1; n < nnz; ++n) {
    /* check for end of outer index */
    if(ttind[n] != ttind[n-1]) {
      if(fi != NULL) {
        fi[nfound] = ttind[n];
      }
      fp[nfound++] = n;
    }
  }

  fp[nfibs] = nnz;
}


static void __mk_fptr(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr,
  idx_t const mode)
{
  assert(mode < ct->nmodes);

  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    __mk_outerptr(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[mode]] + nnzstart;

  csf_sparsity * const pt = ct->pt + tile_id;

  /* we will edit this to point to the new fiber idxs instead of nnz */
  idx_t * const restrict fprev = pt->fptr[mode-1];

  /* first count nfibers */
  idx_t nfibs = 0;
  /* foreach 'slice' in the previous dimension */
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
    ++nfibs; /* one by default per 'slice' */
    /* count fibers in current hyperplane*/
    for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ++nfibs;
      }
    }
  }
  pt->nfibs[mode] = nfibs;


  pt->fptr[mode] = (idx_t *) malloc((nfibs+1) * sizeof(idx_t));
  pt->fids[mode] = (idx_t *) malloc(nfibs * sizeof(idx_t));
  idx_t * const restrict fp = pt->fptr[mode];
  idx_t * const restrict fi = pt->fids[mode];
  fp[0] = 0;

  /* now fill in fiber info */
  idx_t nfound = 0;
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
    idx_t const start = fprev[s]+1;
    idx_t const end = fprev[s+1];

    /* mark start of subtree */
    fprev[s] = nfound;
    fi[nfound] = ttind[start-1];
    fp[nfound++] = start-1;

    /* mark fibers in current hyperplane */
    for(idx_t f=start; f < end; ++f) {
      if(ttind[f] != ttind[f-1]) {
        fi[nfound] = ttind[f];
        fp[nfound++] = f;
      }
    }
  }

  /* mark end of last hyperplane */
  fprev[pt->nfibs[mode-1]] = nfibs;
  fp[nfibs] = nnz;
}


/**
* @brief Allocate and fill a CSF tensor from a coordinate tensor without
*        tiling.
*
* @param ft The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void __csf_alloc_untiled(
  splatt_csf * const ct,
  sptensor_t * const tt)
{
  idx_t const nmodes = tt->nmodes;
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);

  ct->ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ct->tile_dims[m] = 1;
  }
  ct->pt = (csf_sparsity *) malloc(sizeof(csf_sparsity));

  csf_sparsity * const pt = ct->pt;

  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
  pt->fids[nmodes-1] = (idx_t *) malloc(ct->nnz * sizeof(idx_t));
  pt->vals           = (val_t *) malloc(ct->nnz * sizeof(val_t));
  memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]],
      ct->nnz * sizeof(idx_t));
  memcpy(pt->vals, tt->vals, ct->nnz * sizeof(val_t));

  /* setup a basic tile ptr for one tile */
  idx_t nnz_ptr[2];
  nnz_ptr[0] = 0;
  nnz_ptr[1] = tt->nnz;

  /* create fptr entries for the rest of the modes, working down from roots.
   * Skip the bottom level (nnz) */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    __mk_fptr(ct, tt, 0, nnz_ptr, m);
  }
}


/**
* @brief Reorder the nonzeros in a sparse tensor using dense tiling and fill
*        a CSF tensor with the data.
*
* @param ft The CSF tensor to fill.
* @param tt The sparse tensor to start from.
* @param splatt_opts Options array for SPLATT - used for tile dimensions.
*/
static void __csf_alloc_densetile(
  splatt_csf * const ct,
  sptensor_t * const tt,
  double const * const splatt_opts)
{
  idx_t const nmodes = tt->nmodes;

  idx_t ntiles = 1;
  for(idx_t m=0; m < ct->nmodes; ++m) {
    idx_t const depth = csf_mode_depth(m, ct->dim_perm, ct->nmodes);
    if(depth >= MIN_TILE_DEPTH) {
      ct->tile_dims[m] = (idx_t) splatt_opts[SPLATT_OPTION_NTHREADS];
    } else {
      ct->tile_dims[m] = 1;
    }
    ntiles *= ct->tile_dims[m];
  }

  /* perform tensor tiling */
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);
  idx_t * nnz_ptr = tt_densetile(tt, ct->tile_dims);

  ct->ntiles = ntiles;
  ct->pt = (csf_sparsity *) malloc(ntiles * sizeof(csf_sparsity));

  for(idx_t t=0; t < ntiles; ++t) {
    idx_t const startnnz = nnz_ptr[t];
    idx_t const endnnz   = nnz_ptr[t+1];
    idx_t const ptnnz = endnnz - startnnz;

    csf_sparsity * const pt = ct->pt + t;

    /* empty tile */
    if(ptnnz == 0) {
      for(idx_t m=0; m < ct->nmodes; ++m) {
        pt->fptr[m] = NULL;
        pt->fids[m] = NULL;
        pt->nfibs[m] = 0;
      }
      /* first fptr may be accessed anyway */
      pt->fptr[0] = (idx_t *) malloc(2 * sizeof(idx_t));
      pt->fptr[0][0] = 0;
      pt->fptr[0][1] = 0;
      pt->vals = NULL;
      continue;
    }

    /* last row of fptr is just nonzero inds */
    pt->nfibs[nmodes-1] = ptnnz;

    pt->fids[nmodes-1] = (idx_t *) malloc(ptnnz * sizeof(idx_t));
    memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]] + startnnz,
        ptnnz * sizeof(idx_t));

    pt->vals = (val_t *) malloc(ptnnz * sizeof(val_t));
    memcpy(pt->vals, tt->vals + startnnz, ptnnz * sizeof(val_t));

    /* create fptr entries for the rest of the modes*/
    for(idx_t m=0; m < tt->nmodes-1; ++m) {
      __mk_fptr(ct, tt, t, nnz_ptr, m);
    }
  }

  free(nnz_ptr);
}


static void __mk_csf(
  splatt_csf * const ct,
  sptensor_t * const tt,
  splatt_csf_type alloc_type,
  idx_t const mode,
  double const * const splatt_opts)
{
  ct->nnz = tt->nnz;
  ct->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ct->dims[m] = tt->dims[m];
  }

  /* get the indices in order */
  csf_find_mode_order(tt->dims, tt->nmodes, alloc_type, mode, ct->dim_perm);

  ct->which_tile = splatt_opts[SPLATT_OPTION_TILE];
  switch(ct->which_tile) {
  case SPLATT_NOTILE:
    __csf_alloc_untiled(ct, tt);
    break;
  case SPLATT_DENSETILE:
    __csf_alloc_densetile(ct, tt, splatt_opts);
    break;
  default:
    fprintf(stderr, "SPLATT: tiling '%d' unsupported for CSF tensors.\n",
        ct->which_tile);
    break;
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


void csf_free(
  splatt_csf * const ct,
  double const * const opts)
{
  idx_t ntensors = 0;
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    ntensors = 1;
    break;
  case SPLATT_CSF_TWOMODE:
    ntensors = 2;
    break;
  case SPLATT_CSF_ALLMODE:
    ntensors = ct[0].nmodes;
    break;
  }

  for(idx_t i=0; i < ntensors; ++i) {
    /* free each tile of sparsity pattern */
    for(idx_t t=0; t < ct[i].ntiles; ++t) {
      free(ct[i].pt[t].vals);
      free(ct[i].pt[t].fids[ct[i].nmodes-1]);
      for(idx_t m=0; m < ct[i].nmodes-1; ++m) {
        free(ct[i].pt[t].fptr[m]);
        free(ct[i].pt[t].fids[m]);
      }
    }
    free(ct[i].pt);
  }

  free(ct);
}



void csf_find_mode_order(
  idx_t const * const dims,
  idx_t const nmodes,
  csf_mode_type which,
  idx_t const mode,
  idx_t * const perm_dims)
{
  switch(which) {
  case CSF_SORTED_SMALLFIRST:
    __order_dims_small(dims, nmodes, perm_dims);
    break;

  case CSF_SORTED_BIGFIRST:
    __order_dims_large(dims, nmodes, perm_dims);
    break;

  case CSF_SORTED_MINUSONE:
    __order_dims_minusone(dims, nmodes, mode, perm_dims);
    break;

  default:
    fprintf(stderr, "SPLATT: csf_mode_type '%d' not recognized.\n", which);
    break;
  }
}


idx_t csf_storage(
  splatt_csf const * const ct)
{
  idx_t bytes = 0;
  bytes += sizeof(splatt_csf);
  bytes += ct->nnz * sizeof(val_t); /* vals */
  bytes += ct->nnz * sizeof(idx_t); /* fids[nmodes] */
  bytes += ct->ntiles * sizeof(csf_sparsity); /* pt */

  for(idx_t t=0; t < ct->ntiles; ++t) {
    csf_sparsity const * const pt = ct->pt + t;

    for(idx_t m=0; m < ct->nmodes-1; ++m) {
      bytes += (pt->nfibs[m]+1) * sizeof(idx_t); /* fptr */
      if(pt->fids[m] != NULL) {
        bytes += pt->nfibs[m] * sizeof(idx_t); /* fids */
      }
    }
  }
  return bytes;
}


splatt_csf * splatt_csf_alloc(
  sptensor_t * const tt,
  double const * const opts)
{
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];

  splatt_csf * ret = NULL;

  printf("\n----\n");

  switch(which) {
  case SPLATT_CSF_ONEMODE:
    printf("one\n");
    ret = malloc(sizeof(*ret));
    __mk_csf(ret, tt, which, 0, opts);
    break;

  case SPLATT_CSF_TWOMODE:
    printf("two\n");
    ret = malloc(2 * sizeof(*ret));
    __mk_csf(ret + 0, tt, CSF_SORTED_SMALLFIRST, 0, opts);
    __mk_csf(ret + 1, tt, CSF_SORTED_BIGFIRST, 0, opts);
    break;

  case SPLATT_CSF_ALLMODE:
    printf("all\n");
    ret = malloc(tt->nmodes * sizeof(*ret));
    for(idx_t m=0; m < tt->nmodes; ++m) {
      __mk_csf(ret + m, tt, which, m, opts);
    }
    break;
  }

  printf("\n----\n\n");

  return ret;
}


val_t csf_frobsq(
    splatt_csf const * const tensor)
{
  idx_t const nmodes = tensor->nmodes;
  val_t norm = 0;
  #pragma omp parallel reduction(+:norm)
  {
    for(idx_t t=0; t < tensor->ntiles; ++t) {
      val_t const * const vals = tensor->pt[t].vals;
      idx_t const nnz = tensor->pt[t].nfibs[nmodes-1];

      #pragma omp for
      for(idx_t n=0; n < nnz; ++n) {
        norm += vals[n] * vals[n];
      }
    }
  }

  return norm;
}

