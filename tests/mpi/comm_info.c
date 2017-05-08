
#include "../ctest/ctest.h"
#include "../splatt_test.h"
#include "../../include/splatt.h"
#include "../../src/base.h"
#include "../../src/sptensor.h"
#include "../../src/splatt_mpi.h"
#include "../../src/mpi/comm_info.h"



CTEST(mpi_comm_info, alloc)
{
  int npes;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  splatt_comm_info * mpi = splatt_alloc_comm_info(MPI_COMM_WORLD);
  ASSERT_NOT_NULL(mpi);

  if(mpi->world_comm == MPI_COMM_NULL) {
    ASSERT_FAIL();
  }
  ASSERT_EQUAL(npes, mpi->world_npes);
  ASSERT_EQUAL(rank, mpi->world_rank);

  if(mpi->grid_comm != MPI_COMM_NULL) {
    ASSERT_FAIL();
  }
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    if(mpi->layer_comms[m] != MPI_COMM_NULL) {
      ASSERT_FAIL();
    }
  }

  ASSERT_EQUAL(0, mpi->nmodes);
  ASSERT_EQUAL(0, mpi->global_nnz);
  for(idx_t m=0; m < MAX_NMODES; ++m) {
    ASSERT_NULL(mpi->layer_ptrs[m]);
    ASSERT_NULL(mpi->mat_ptrs[m]);

    ASSERT_EQUAL(0, mpi->global_dims[m]);
    ASSERT_EQUAL(0, mpi->layer_dims[m]);
    ASSERT_EQUAL(0, mpi->mat_start[m]);
    ASSERT_EQUAL(0, mpi->mat_end[m]);

    ASSERT_EQUAL(true, mpi->compress[m]);
    ASSERT_NULL(mpi->indmap[m]);
  }

  ASSERT_NOT_NULL(mpi->statuses);
  ASSERT_NOT_NULL(mpi->send_reqs);
  ASSERT_NOT_NULL(mpi->recv_reqs);

  splatt_free_comm_info(mpi);
}

CTEST(mpi_comm_info, free)
{
  /* just don't crash */
  splatt_free_comm_info(NULL);
}

CTEST(mpi_comm_info, fill_global)
{
  /* load full tensor and distributed tensor */
  sptensor_t * gold_tt = tt_read(datasets[0]);
  sptensor_t * mpi_tt = mpi_simple_distribute(datasets[0], MPI_COMM_WORLD);

  /* grab stats */
  splatt_comm_info * mpi = splatt_alloc_comm_info(MPI_COMM_WORLD);
  comm_fill_global(mpi_tt, mpi);

  ASSERT_EQUAL(gold_tt->nnz, mpi->global_nnz);
  ASSERT_EQUAL(gold_tt->nmodes, mpi->nmodes);
  for(idx_t m=0; m < mpi->nmodes; ++m) {
    ASSERT_EQUAL(gold_tt->dims[m], mpi->global_dims[m]);
  }
  for(idx_t m= mpi->nmodes; m < SPLATT_MAX_NMODES; ++m) {
    ASSERT_EQUAL(0, mpi->global_dims[m]);
  }

  splatt_free_comm_info(mpi);
  tt_free(gold_tt);
  tt_free(mpi_tt);
}

