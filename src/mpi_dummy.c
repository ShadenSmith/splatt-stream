

#include "splatt_mpi.h"

/*
 * Simple non-MPI versions of some communication routines -- these just make
 * the source code easy in some places (such as not changing function
 * parameters).
 *
 * The actually meaningful implementations are found in the `mpi/` directory,
 * which is otherwise not included unless configured with MPI.
 */
#ifndef SPLATT_USE_MPI

splatt_comm_info * splatt_alloc_comm_info()
{
  splatt_comm_info * mpi = splatt_malloc(sizeof(*mpi));
  mpi->world_rank = 0;
  mpi->world_npes = 1;
  return mpi;
}



void splatt_free_comm_info(
    splatt_comm_info * comm_info)
{
  if(comm_info == NULL) {
    return;
  }

  splatt_free(comm_info);
}
#endif



