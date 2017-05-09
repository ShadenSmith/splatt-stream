/**
* @file comm_info.c
* @brief Functions for handling splatt_comm_info structures.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-07-12
*/

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "comm_info.h"



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

/**
* @brief Free an MPI communicator after checking for MPI_COMM_NULL.
*
* @param comm A pointer to the communicator which we will free.
*/
static inline void p_free_mpi_comm(
    MPI_Comm * comm)
{
  if(*comm != MPI_COMM_NULL) {
    MPI_Comm_free(comm);
  }
}



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/



/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/

splatt_comm_info * splatt_alloc_comm_info(
    MPI_Comm comm)
{
  splatt_comm_info * mpi = splatt_malloc(sizeof(*mpi));

  /* setup communicators */
  MPI_Comm_dup(comm, &(mpi->world_comm));

  mpi->grid_comm = MPI_COMM_NULL;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    mpi->layer_comms[m] = MPI_COMM_NULL;
  }

  /* get communicator statistics */
  MPI_Comm_size(mpi->world_comm, &(mpi->world_npes));
  MPI_Comm_rank(mpi->world_comm, &(mpi->world_rank));

  mpi->decomp = SPLATT_DECOMP_MEDIUM;
  mpi->comm_type = SPLATT_COMM_ALL2ALL;

  mpi->nmodes = 0;
  mpi->global_nnz = 0;
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    mpi->global_dims[m] = 0;
    mpi->layer_dims[m]  = 0;
    mpi->mat_start[m]   = 0;
    mpi->mat_end[m]     = 0;

    /* ptrs */
    mpi->layer_ptrs[m] = NULL;
    mpi->mat_ptrs[m] = NULL;

    mpi->compress[m] = true;
    mpi->indmap[m] = NULL;

    mpi->statuses = splatt_malloc(mpi->world_npes * sizeof(*mpi->statuses));
    mpi->send_reqs = splatt_malloc(mpi->world_npes * sizeof(*mpi->send_reqs));
    mpi->recv_reqs = splatt_malloc(mpi->world_npes * sizeof(*mpi->recv_reqs));
  }

  return mpi;
}


void splatt_free_comm_info(
    splatt_comm_info * comm_info)
{
  if(comm_info == NULL) {
    return;
  }

  /* free ptr arrays */
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    splatt_free(comm_info->layer_ptrs[m]);
    splatt_free(comm_info->mat_ptrs[m]);
  }

  /* free compression */
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    splatt_free(comm_info->indmap[m]);
  }

  /* free communicators */
  p_free_mpi_comm(&(comm_info->world_comm));
  p_free_mpi_comm(&(comm_info->grid_comm));
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    p_free_mpi_comm(&(comm_info->layer_comms[m]));
  }

  splatt_free(comm_info->statuses);
  splatt_free(comm_info->send_reqs);
  splatt_free(comm_info->recv_reqs);

  splatt_free(comm_info);
}


void comm_fill_global(
    sptensor_t const * const tt,
    splatt_comm_info * mpi)
{
  for(idx_t m = 0; m < MAX_NMODES; ++m) {
    mpi->global_dims[m] = 0;
  }

  /* TODO: fill dim info locally in case we haven't already */

  mpi->nmodes = tt->nmodes;
  MPI_Allreduce(&(tt->nnz), &(mpi->global_nnz), 1, SPLATT_MPI_IDX,
      MPI_SUM, mpi->world_comm);
  MPI_Allreduce(tt->dims, mpi->global_dims, tt->nmodes, SPLATT_MPI_IDX,
      MPI_MAX, mpi->world_comm);
}

