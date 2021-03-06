
/******************************************************************************
 * INCLUDES
 *****************************************************************************/

#include "../splatt_mpi.h"
#include "../util.h"
#include "../ccp/ccp.h"
#include "comm_info.h"




/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/



/**
* @brief Find the boundaries for one dimension of the medium-grained
*        decomposition.
*
* @param coord The tensor we are partitioning.
* @param mode The mode we are partitioning.
* @param[outt] mpi MPI rank information.
*/
static idx_t * p_partition_mode_by_nnz(
    splatt_coord const * const coord,
    idx_t const mode,
    idx_t const nparts,
    splatt_comm_info * const mpi)
{
  /* build histogram of nnz to determine boundaries of each mode */
  idx_t * slice_hist = tt_get_hist(coord, mode);
  MPI_Allreduce(
      MPI_IN_PLACE, slice_hist,
      mpi->global_dims[mode], SPLATT_MPI_IDX, MPI_SUM,
      mpi->grid_comm);

  /* partition mode */
  assert(nparts > 0);
  idx_t * part = partition_1d(slice_hist, mpi->global_dims[mode], nparts);

  splatt_free(slice_hist);

  return part;
}



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





static int p_determine_med_owner(
    splatt_coord const * const tt,
    idx_t const n,
    splatt_comm_info const * const mpi)
{
  int coords[MAX_NMODES];

  assert(mpi->decomp == SPLATT_DECOMP_MEDIUM);

  /* determine the coordinates of the owner rank */
  for(idx_t m=0; m < tt->nmodes; ++m) {
    idx_t const id = tt->ind[m][n];
    /* silly linear scan over each layer.
     * TODO: do a binary search */
    for(int l=0; l <= mpi->layer_dims[m]; ++l) {
      if(id < mpi->layer_ptrs[m][l]) {
        coords[m] = l-1;
        break;
      }
    }
  }

  /* translate that to an MPI rank */
  int owner;
  MPI_Cart_rank(mpi->grid_comm, coords, &owner);
  return owner;
}




/**
* @brief Fill in grid dimensions for a medium-grained decomposition. If
*        rank_dims is NULL, find a good arrangement.
*
* @param[out] mpi MPI structure to fill.
* @param rank_dims Requested dimensions. This can be NULL for no request.
*
* @return True, if the requested dimension is valid or we were able to find a
*         decomposition (always true). False if the requested decomposition is
*         invalid.
*/
static bool p_fill_dim_medium(
    splatt_comm_info * const mpi,
    int const * const rank_dims)
{
  /* set medium-grained dimensions */
  if(rank_dims == NULL) {
    p_get_best_med_dim(mpi);
  } else {
    for(idx_t m=0; m < mpi->nmodes; ++m) {
      mpi->layer_dims[m] = rank_dims[m];
    }
  }

  /* sanity check */
  int total_p = 1;
  for(idx_t m=0; m < mpi->nmodes; ++m) {
    total_p *= mpi->layer_dims[m];
  }
  if(total_p != mpi->world_npes) {
    fprintf(stderr, "SPLATT ERROR: supplied medium-grained decomposition "
                    "requires %d MPI ranks, but only have %d.\n",
            total_p, mpi->world_npes);

    /* Empty the comm_info and return. */
    for(idx_t m=0; m < mpi->nmodes; ++m) {
      mpi->layer_dims[m] = 0;
    }
    mpi->nmodes = 0;
    return false;
  }
  assert(total_p == mpi->world_npes);

  return true;
}


/**
* @brief Setup communicatory info for a MG distribution.
*
* @param mpi MPI rank information to fill in.
*/
static void p_setup_comms_medium(
    splatt_comm_info * const mpi)
{
  int periods[MAX_NMODES];

  idx_t const nmodes = mpi->nmodes;

  for(idx_t m=0; m < nmodes; ++m) {
    periods[m] = 1;
  }

  /* create new communicator and update global rank */
  MPI_Cart_create(mpi->world_comm, nmodes, mpi->layer_dims, periods, 0,
      &(mpi->grid_comm));
  MPI_Comm_rank(mpi->grid_comm, &(mpi->grid_rank));

  /* get 3d coordinates */
  MPI_Cart_coords(mpi->grid_comm, mpi->grid_rank, nmodes, mpi->grid_coords);

  /* compute ranks relative to tensor mode */
  for(idx_t m=0; m < nmodes; ++m) {
    int const layer_id = mpi->grid_coords[m];

    /* now split grid communicator into layers */
    MPI_Comm_split(mpi->grid_comm, layer_id, 0, &(mpi->layer_comms[m]));
    MPI_Comm_rank(mpi->layer_comms[m], &(mpi->layer_rank[m]));
    MPI_Comm_size(mpi->layer_comms[m], &(mpi->layer_size[m]));

    assert(mpi->layer_rank[m] < mpi->world_npes / mpi->layer_dims[m]);
  }
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
  assert(tt_cpd != coord);
  tt_free(coord);

  /* XXX: now compute distribution of matrices for CPD */

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

  /* setup grid dimensions */
  p_fill_dim_medium(comm_info, rank_dims);

  /* build cartiesan communicators */
  p_setup_comms_medium(comm_info);

  /* partition each mode separately */
  for(idx_t m=0; m < coord->nmodes; ++m) {
    comm_info->layer_ptrs[m] = p_partition_mode_by_nnz(coord, m,
        comm_info->layer_dims[m], comm_info);
  }

  /* figure out which MPI rank each nnz goes to */
  int * parts = splatt_malloc(coord->nnz * sizeof(*parts));
  #pragma omp parallel for schedule(static)
  for(idx_t n=0; n < coord->nnz; ++n) {
    parts[n] = p_determine_med_owner(coord, n, comm_info);
  }

  /* rearrange the data */
  splatt_coord * medium = mpi_rearrange_by_part(coord, parts,
      comm_info->grid_comm);
  splatt_free(parts);

  /* now map tensor indices to local (layer) coordinates and fill in dims */
  for(idx_t m=0; m < medium->nmodes; ++m) {
    idx_t const layer_start
        = comm_info->layer_ptrs[m][comm_info->grid_coords[m]];
    idx_t const layer_end
        = comm_info->layer_ptrs[m][comm_info->grid_coords[m] + 1];

    medium->dims[m] = layer_end - layer_start;

    #pragma omp parallel for schedule(static)
    for(idx_t n=0; n < medium->nnz; ++n) {
      assert(medium->ind[m][n] >= layer_start);
      assert(medium->ind[m][n] < layer_end);
      medium->ind[m][n] -= layer_start;
    }
  }

  return medium;
}



