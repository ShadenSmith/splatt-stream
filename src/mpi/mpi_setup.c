
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"


/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static void __fill_ineed_ptrs(
  sptensor_t const * const tt,
  idx_t const mode,
  rank_info * const rinfo,
  MPI_Comm const comm)
{
  idx_t const m = mode;
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  rinfo->nlocal2nbr[m] = 0;
  rinfo->local2nbr_ptr[m] = (int *) calloc((size+1),  sizeof(int));
  rinfo->nbr2globs_ptr[m] = (int *) malloc((size+1) * sizeof(int));

  int * const local2nbr_ptr = rinfo->local2nbr_ptr[m];
  int * const nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
  idx_t const * const mat_ptrs = rinfo->mat_ptrs[m];

  int pdest = 0;
  /* count recvs for each process */
  for(idx_t i=0; i < tt->dims[m]; ++i) {
    /* grab global index */
    idx_t const gi = (tt->indmap[m] == NULL) ? i : tt->indmap[m][i];
    /* move to the next processor if necessary */
    while(gi >= mat_ptrs[pdest+1]) {
      ++pdest;
    }

    assert(pdest < size);
    assert(gi >= mat_ptrs[pdest]);
    assert(gi < mat_ptrs[pdest+1]);

    /* if it is non-local */
    if(pdest != rank) {
      local2nbr_ptr[pdest] += 1;
      rinfo->nlocal2nbr[m] += 1;
    }
  }

  /* communicate local2nbr and receive nbr2globs */
  MPI_Alltoall(local2nbr_ptr, 1, MPI_INT, nbr2globs_ptr, 1, MPI_INT, comm);

  rinfo->nnbr2globs[m] = 0;
  for(int p=0; p < size; ++p) {
    rinfo->nnbr2globs[m] += nbr2globs_ptr[p];
  }
  nbr2globs_ptr[size] = rinfo->nnbr2globs[m];
}


static void __fill_ineed_inds(
  sptensor_t const * const tt,
  idx_t const mode,
  idx_t const nfactors,
  rank_info * const rinfo,
  MPI_Comm const comm)
{
  idx_t const m = mode;
  int size;
  int rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  /* allocate space for all communicated indices */
  rinfo->local2nbr_inds[m] = (idx_t *) malloc(rinfo->nlocal2nbr[m] *
      sizeof(idx_t));
  rinfo->nbr2local_inds[m] = (idx_t *) malloc(rinfo->nlocal2nbr[m] *
      sizeof(idx_t));
  rinfo->nbr2globs_inds[m] = (idx_t *) malloc(rinfo->nnbr2globs[m] *
      sizeof(idx_t));

  /* extract these pointers to save some typing */
  int * const local2nbr_ptr = rinfo->local2nbr_ptr[m];
  int * const nbr2globs_ptr = rinfo->nbr2globs_ptr[m];
  idx_t const * const mat_ptrs = rinfo->mat_ptrs[m];
  idx_t * const nbr2globs_inds = rinfo->nbr2globs_inds[m];
  idx_t * const local2nbr_inds = rinfo->local2nbr_inds[m];
  idx_t * const nbr2local_inds = rinfo->nbr2local_inds[m];

  /* fill indices for local2nbr */
  idx_t recvs = 0;
  int pdest = 0;
  for(idx_t i=0; i < tt->dims[m]; ++i) {
    /* grab global index */
    idx_t const gi = (tt->indmap[m] == NULL) ? i : tt->indmap[m][i];
    /* move to the next processor if necessary */
    while(gi >= mat_ptrs[pdest+1]) {
      ++pdest;
    }
    /* if it is non-local */
    if(pdest != rank) {
      nbr2local_inds[recvs] = gi;
      local2nbr_inds[recvs] = i;
      ++recvs;
    }
  }

  rinfo->local2nbr_disp[m] = (int *) malloc(size * sizeof(int));
  rinfo->nbr2globs_disp[m] = (int *) malloc(size * sizeof(int));
  int * const local2nbr_disp = rinfo->local2nbr_disp[m];
  int * const nbr2globs_disp = rinfo->nbr2globs_disp[m];

  local2nbr_disp[0] = 0;
  nbr2globs_disp[0] = 0;
  for(int p=1; p < size; ++p) {
    local2nbr_disp[p] = local2nbr_disp[p-1] + local2nbr_ptr[p-1];
    nbr2globs_disp[p] = nbr2globs_disp[p-1] + nbr2globs_ptr[p-1];
  }

  assert((int)rinfo->nlocal2nbr[m] == local2nbr_disp[size-1] +
      local2nbr_ptr[size-1]);

  /* send my non-local indices and get indices for each neighbors' partials
   * that I will receive */
  MPI_Alltoallv(nbr2local_inds, local2nbr_ptr, local2nbr_disp, SS_MPI_IDX,
                nbr2globs_inds, nbr2globs_ptr, nbr2globs_disp, SS_MPI_IDX,
                comm);

  /* we don't need nbr2local_inds anymore */
  free(rinfo->nbr2local_inds[m]);
  rinfo->nbr2local_inds[m] = NULL;

  /* sanity check on nbr2globs_inds */
  for(idx_t i=0; i < rinfo->nnbr2globs[m]; ++i) {
    assert(nbr2globs_inds[i] >= rinfo->mat_start[m]);
    assert(nbr2globs_inds[i] < rinfo->mat_end[m]);
  }

  /* scale ptrs and disps by nfactors now that indices are communicated */
  for(int p=0; p < size; ++p) {
    local2nbr_ptr[p] *= nfactors;
    nbr2globs_ptr[p] *= nfactors;
  }
  /* recompute disps */
  local2nbr_disp[0] = 0;
  nbr2globs_disp[0] = 0;
  for(int p=1; p < size; ++p) {
    local2nbr_disp[p] = local2nbr_disp[p-1] + local2nbr_ptr[p-1];
    nbr2globs_disp[p] = nbr2globs_disp[p-1] + nbr2globs_ptr[p-1];
  }
}


/**
* @brief Setup communicator info for a 1D distribution.
*
* @param rinfo MPI rank information to fill in.
*/
static void __setup_1d(
  rank_info * const rinfo)
{
  rinfo->comm_3d = MPI_COMM_WORLD;
  for(idx_t m=0; m < rinfo->nmodes; ++m) {
    rinfo->layer_comm[m] = MPI_COMM_WORLD;
    rinfo->layer_size[m] = 1;
    rinfo->dims_3d[m] = rinfo->npes;
  }
}


/**
* @brief Setup communicatory info for a 3D distribution.
*
* @param rinfo MPI rank information to fill in.
*/
static void __setup_3d(
  rank_info * const rinfo)
{
  int * const dims_3d = rinfo->dims_3d;
  int periods[MAX_NMODES];

  if(dims_3d[0] * dims_3d[1] * dims_3d[2] != rinfo->npes) {
    if(rinfo->rank == 0) {
      fprintf(stderr, "SPLATT: dimension %dx%dx%d does not match np=%d.\n",
          dims_3d[0], dims_3d[1], dims_3d[2], rinfo->npes);
    }
    MPI_Finalize();
    abort();
  }

  periods[0] = periods[1] = periods[2] = 1;

  /* create new communicator and update global rank */
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims_3d, periods, 1, &(rinfo->comm_3d));
  MPI_Comm_rank(MPI_COMM_WORLD, &(rinfo->rank));
  MPI_Comm_rank(rinfo->comm_3d, &(rinfo->rank_3d));

  /* get 3d coordinates */
  MPI_Cart_coords(rinfo->comm_3d, rinfo->rank_3d, 3, rinfo->coords_3d);

  /* compute ranks relative to tensor mode */
  for(idx_t m=0; m < 3; ++m) {
    /* map coord within layer to 1D */
    int const coord1d = rinfo->coords_3d[(m+1)%3] * dims_3d[(m+2)%3] +
                        rinfo->coords_3d[(m+2)%3];
    int const layer_id = rinfo->coords_3d[m];
    /* relative rank in this mode */
    rinfo->mode_rank[m] = (dims_3d[(m+1)%3] * dims_3d[(m+2)%3] * layer_id) + coord1d;

    /* now split 3D communicator into layers */
    MPI_Comm_split(rinfo->comm_3d, layer_id, 0, &(rinfo->layer_comm[m]));
    MPI_Comm_rank(rinfo->layer_comm[m], &(rinfo->layer_rank[m]));
    MPI_Comm_size(rinfo->layer_comm[m], &(rinfo->layer_size[m]));

    assert(rinfo->layer_rank[m] < rinfo->npes / dims_3d[m]);
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mpi_compute_ineed(
  rank_info * const rinfo,
  sptensor_t const * const tt,
  idx_t const mode,
  idx_t const nfactors,
  idx_t const distribution)
{
  /* fill local2nbr and nbr2globs ptrs */
  __fill_ineed_ptrs(tt, mode, rinfo, rinfo->layer_comm[mode]);

  /* fill indices */
  __fill_ineed_inds(tt, mode, nfactors, rinfo, rinfo->layer_comm[mode]);
}


void mpi_setup_comms(
  rank_info * const rinfo)
{
  switch(rinfo->distribution) {
  case 1:
    __setup_1d(rinfo);
    break;
  case 3:
    __setup_3d(rinfo);
    break;
  default:
    fprintf(stderr, "SPLATT: distribution %"SS_IDX" not supported. "
                    "Choose from {1,3}.\n", rinfo->distribution);
    abort();
  }
}


void rank_free(
  rank_info rinfo,
  idx_t const nmodes)
{
  switch(rinfo.distribution) {
  case 1:
    break;
  case 3:
    MPI_Comm_free(&rinfo.comm_3d);
    for(idx_t m=0; m < nmodes; ++m) {
      MPI_Comm_free(&rinfo.layer_comm[m]);
      free(rinfo.mat_ptrs[m]);

      /* send/recv structures */
      free(rinfo.nbr2globs_inds[m]);
      free(rinfo.local2nbr_inds[m]);
      free(rinfo.nbr2local_inds[m]);
      free(rinfo.local2nbr_ptr[m]);
      free(rinfo.nbr2globs_ptr[m]);
      free(rinfo.local2nbr_disp[m]);
      free(rinfo.nbr2globs_disp[m]);
    }
    break;
  }
}

