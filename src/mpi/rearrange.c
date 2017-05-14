
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../splatt_mpi.h"
#include "../util.h"
#include "comm_info.h"




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Fill in the best MPI dimensions we can find. The truly optimal
*        solution should involve the tensor's sparsity pattern, but in general
*        this works as good (but usually better) than the hand-tuned dimensions
*        that we tried.
*
* @param[out] mpi MPI rank information
*/
static void p_get_best_med_dim(
    splatt_comm_info * const mpi)
{
  int nprimes = 0;
  int * primes = get_primes(mpi->world_npes, &nprimes);

  idx_t total_size = 0;
  for(idx_t m=0; m < mpi->nmodes; ++m) {
    total_size += mpi->global_dims[m];

    /* reset mpi dims */
    mpi->layer_dims[m] = 1;
  }
  idx_t target = total_size / (idx_t)mpi->world_npes;

  long diffs[MAX_NMODES];

  /* start from the largest prime */
  for(int p = nprimes-1; p >= 0; --p) {
    int furthest = 0;
    /* find dim furthest from target */
    for(idx_t m=0; m < mpi->nmodes; ++m) {
      /* distance is current - target */
      idx_t const curr = mpi->global_dims[m] / mpi->layer_dims[m];
      /* avoid underflow */
      diffs[m] = (curr > target) ? (curr - target) : 0;

      if(diffs[m] > diffs[furthest]) {
        furthest = m;
      }
    }

    /* assign p processes to furthest mode */
    mpi->layer_dims[furthest] *= primes[p];
  }

  free(primes);
}





/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/



sptensor_t * mpi_rearrange_by_part(
  sptensor_t const * const ttbuf,
  int const * const parts,
  MPI_Comm comm)
{
  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  /* count how many to send to each process */
  int * nsend = calloc(npes, sizeof(*nsend));
  int * nrecv = calloc(npes, sizeof(*nrecv));
  #pragma omp parallel for schedule(static)
  for(idx_t n=0; n < ttbuf->nnz; ++n) {
    #pragma omp atomic
    ++nsend[parts[n]];
  }
  MPI_Alltoall(nsend, 1, MPI_INT, nrecv, 1, MPI_INT, comm);

  idx_t send_total = 0;
  idx_t recv_total = 0;
  for(int p=0; p < npes; ++p) {
    send_total += nsend[p];
    recv_total += nrecv[p];
  }
  assert(send_total == ttbuf->nnz);

  /* how many nonzeros I'll own */
  idx_t const nowned = recv_total;

  int * send_disp = splatt_malloc((npes+1) * sizeof(*send_disp));
  int * recv_disp = splatt_malloc((npes+1) * sizeof(*recv_disp));

  /* recv_disp is const so we'll just fill it out once */
  recv_disp[0] = 0;
  for(int p=1; p <= npes; ++p) {
    recv_disp[p] = recv_disp[p-1] + nrecv[p-1];
  }

  /* allocate my tensor and send buffer */
  sptensor_t * tt = tt_alloc(nowned, ttbuf->nmodes);
  idx_t * isend_buf = splatt_malloc(ttbuf->nnz * sizeof(*isend_buf));

  /* rearrange into sendbuf and send one mode at a time */
  for(idx_t m=0; m < ttbuf->nmodes; ++m) {
    /* prefix sum to make disps */
    send_disp[0] = send_disp[1] = 0;
    for(int p=2; p <= npes; ++p) {
      send_disp[p] = send_disp[p-1] + nsend[p-2];
    }

    idx_t const * const ind = ttbuf->ind[m];
    for(idx_t n=0; n < ttbuf->nnz; ++n) {
      idx_t const index = send_disp[parts[n]+1]++;
      isend_buf[index] = ind[n];
    }

    /* exchange indices */
    MPI_Alltoallv(isend_buf, nsend, send_disp, SPLATT_MPI_IDX,
                  tt->ind[m], nrecv, recv_disp, SPLATT_MPI_IDX,
                  comm);
  }
  splatt_free(isend_buf);

  /* lastly, rearrange vals */
  val_t * vsend_buf = splatt_malloc(ttbuf->nnz * sizeof(*vsend_buf));
  send_disp[0] = send_disp[1] = 0;
  for(int p=2; p <= npes; ++p) {
    send_disp[p] = send_disp[p-1] + nsend[p-2];
  }

  val_t const * const vals = ttbuf->vals;
  for(idx_t n=0; n < ttbuf->nnz; ++n) {
    idx_t const index = send_disp[parts[n]+1]++;
    vsend_buf[index] = vals[n];
  }
  /* exchange vals */
  MPI_Alltoallv(vsend_buf, nsend, send_disp, SPLATT_MPI_VAL,
                tt->vals,  nrecv, recv_disp, SPLATT_MPI_VAL,
                comm);
  splatt_free(vsend_buf);
  splatt_free(send_disp);
  splatt_free(recv_disp);

  /* allocated with calloc */
  free(nsend);
  free(nrecv);

  return tt;
}





/******************************************************************************
 * API FUNCTONS
 *****************************************************************************/


splatt_coord * splatt_mpi_distribute_cpd(
    char const * const fname,
    splatt_cpd_opts const * const cpd_opts,
    splatt_comm_info * const comm_info)
{
  /* do a simple distribution first */
  splatt_coord * coord = splatt_coord_load_mpi(fname, comm_info);

  /* rearrange for a CPD */
  splatt_coord * tt_cpd = splatt_mpi_rearrange_cpd(coord, cpd_opts, comm_info);

  /* clean up */
  assert(tt_cpd != coord);
  tt_free(coord);

  return tt_cpd;
}



splatt_coord * splatt_mpi_rearrange_cpd(
    splatt_coord * const coord,
    splatt_cpd_opts const * const cpd_opts,
    splatt_comm_info * const comm_info)
{
  /* The best we have right now is to just use medium-grained. */
  return splatt_mpi_rearrange_medium(coord, NULL, comm_info);
}



splatt_coord * splatt_mpi_rearrange_medium(
    splatt_coord * const coord,
    int const * const rank_dims,
    splatt_comm_info * const comm_info)
{
  assert(comm_info != NULL);

  comm_fill_global(coord, comm_info);
  comm_info->decomp = SPLATT_DECOMP_MEDIUM;

  /* set medium-grained dimensions */
  if(rank_dims == NULL) {
    p_get_best_med_dim(comm_info);
  } else {
    for(idx_t m=0; m < comm_info->nmodes; ++m) {
      comm_info->layer_dims[m] = rank_dims[m];
    }
  }

  /* sanity check */
  int total_p = 1;
  for(idx_t m=0; m < comm_info->nmodes; ++m) {
    total_p *= comm_info->layer_dims[m];
  }
  if(total_p != comm_info->world_npes) {
    fprintf(stderr, "SPLATT ERROR: supplied medium-grained decomposition "
                    "requires %d MPI ranks, but only have %d.\n",
            total_p, comm_info->world_npes);

    /* Empty the comm_info and return. */
    for(idx_t m=0; m < comm_info->nmodes; ++m) {
      comm_info->layer_dims[m] = 0;
    }
    comm_info->nmodes = 0;
    return NULL;
  }
  assert(total_p == comm_info->world_npes);

  return NULL;
}



