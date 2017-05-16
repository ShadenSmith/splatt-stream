
#include "../ctest/ctest.h"
#include "../splatt_test.h"

#include "../../src/io.h"
#include "../../src/util.h"
#include "../../src/sptensor.h"
#include "../../src/splatt_mpi.h"



CTEST_DATA(mpi_rearrange)
{
  idx_t ntensors;
};

CTEST_SETUP(mpi_rearrange)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
}

CTEST_TEARDOWN(mpi_rearrange)
{
}



CTEST2(mpi_rearrange, mpi_rearrange_by_part_cyclic)
{
  int npes;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  for(idx_t i=0; i < data->ntensors; ++i) {
    splatt_coord * mpi_tt = mpi_simple_distribute(datasets[i], MPI_COMM_WORLD);

    /* do a cyclic distribution based on % ind[0][x] */
    int * parts = splatt_malloc(mpi_tt->nnz * sizeof(*parts));
    for(idx_t x=0; x < mpi_tt->nnz; ++x) {
      parts[x] = mpi_tt->ind[0][x] % npes;
    }

    splatt_coord * dist = mpi_rearrange_by_part(mpi_tt, parts, MPI_COMM_WORLD);

    /* ensure all the nnz are present */
    idx_t total_nnz = dist->nnz;
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1, SPLATT_MPI_IDX, MPI_SUM,
        MPI_COMM_WORLD);
    if(rank == 0) {
      splatt_coord * gold = tt_read(datasets[i]);
      ASSERT_EQUAL(gold->nnz, total_nnz);
      tt_free(gold);
    }

    /* test that non-zeros were distributed correctly. */
    for(idx_t x=0; x < dist->nnz; ++x) {
      ASSERT_EQUAL(rank, dist->ind[0][x] % npes);
    }

    tt_free(mpi_tt);
    tt_free(dist);
    splatt_free(parts);
  }
}


/* all nnz on one rank */
CTEST2(mpi_rearrange, mpi_rearrange_by_part_all1)
{
  int npes;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  for(idx_t i=0; i < data->ntensors; ++i) {
    splatt_coord * mpi_tt = mpi_simple_distribute(datasets[i], MPI_COMM_WORLD);

    int * parts = splatt_malloc(mpi_tt->nnz * sizeof(*parts));
    splatt_coord * dist = NULL;

    /* put all nnz on first rank */
    for(idx_t x=0; x < mpi_tt->nnz; ++x) {
      parts[x] = 0;
    }
    dist = mpi_rearrange_by_part(mpi_tt, parts, MPI_COMM_WORLD);
    if(rank == 0) {
      splatt_coord * gold = tt_read(datasets[i]);
      ASSERT_EQUAL(gold->nnz, dist->nnz);
      tt_free(gold);
    } else {
      ASSERT_EQUAL(0, dist->nnz);
    }
    tt_free(dist);

    /* put all nnz on last rank */
    for(idx_t x=0; x < mpi_tt->nnz; ++x) {
      parts[x] = npes-1;
    }
    dist = mpi_rearrange_by_part(mpi_tt, parts, MPI_COMM_WORLD);
    if(rank == npes-1) {
      splatt_coord * gold = tt_read(datasets[i]);
      ASSERT_EQUAL(gold->nnz, dist->nnz);
      tt_free(gold);
    } else {
      ASSERT_EQUAL(0, dist->nnz);
    }
    tt_free(dist);


    /* put all nnz on middle rank */
    int const mid = npes / 2;
    for(idx_t x=0; x < mpi_tt->nnz; ++x) {
      parts[x] = mid;
    }
    dist = mpi_rearrange_by_part(mpi_tt, parts, MPI_COMM_WORLD);
    if(rank == mid) {
      splatt_coord * gold = tt_read(datasets[i]);
      ASSERT_EQUAL(gold->nnz, dist->nnz);
      tt_free(gold);
    } else {
      ASSERT_EQUAL(0, dist->nnz);
    }
    tt_free(dist);

    tt_free(mpi_tt);
    splatt_free(parts);
  }
}


CTEST2(mpi_rearrange, mpi_rearrange_medium_best)
{
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();
  for(idx_t i=0; i < data->ntensors; ++i) {
    splatt_comm_info * mpi = splatt_alloc_comm_info(MPI_COMM_WORLD);
    splatt_coord * mpi_tt = splatt_coord_load_mpi(datasets[i], mpi);

    /* rearrange */
    splatt_coord * med = splatt_mpi_rearrange_medium(mpi_tt, NULL, mpi);

    /* inspect decomposition */
    ASSERT_EQUAL(SPLATT_DECOMP_MEDIUM, mpi->decomp);
    for(idx_t m=0; m < mpi_tt->nmodes; ++m) {
      ASSERT_NOT_NULL(mpi->layer_ptrs[m]);
      ASSERT_EQUAL(0, mpi->layer_ptrs[m][0]);
      ASSERT_EQUAL(mpi->global_dims[m],
                   mpi->layer_ptrs[m][mpi->layer_dims[m]]);
    }

    /* inspect new tensor */
    ASSERT_NOT_NULL(med);
    ASSERT_EQUAL(mpi_tt->nmodes, med->nmodes);
    idx_t total_nnz = med->nnz;
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1, SPLATT_MPI_IDX, MPI_SUM,
        MPI_COMM_WORLD);
    ASSERT_EQUAL(mpi->global_nnz, total_nnz);

    /* now go over all nnz and ensure indices are in proper range */
    for(idx_t m=0; m < med->nmodes; ++m) {
      idx_t const layer_size =
        mpi->layer_ptrs[m][mpi->grid_coords[m]+1]
        - mpi->layer_ptrs[m][mpi->grid_coords[m]];
      for(idx_t x=0; x < med->nnz; ++x) {
        ASSERT_TRUE(med->ind[m][x] < layer_size);
      }
    }

    tt_free(med);
    tt_free(mpi_tt);
    splatt_free_comm_info(mpi);
  }
}


CTEST2(mpi_rearrange, mpi_rearrange_medium_fixeddim)
{
  int npes;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int nprimes = 0;
  int * primes = get_primes(npes, &nprimes);

  /* each rank needs to create the same decomposition */
  srand(5);

  splatt_comm_info * mpi = splatt_alloc_comm_info(MPI_COMM_WORLD);
  splatt_cpd_opts * opts = splatt_alloc_cpd_opts();
  for(idx_t i=0; i < data->ntensors; ++i) {
    splatt_coord * mpi_tt = splatt_coord_load_mpi(datasets[i], mpi);

    /* create dimensions */
    int dims[SPLATT_MAX_NMODES];
    for(idx_t m=0; m < mpi_tt->nmodes; ++m) {
      dims[m] = 1;
    }
    for(int p=0; p < nprimes; ++p) {
      idx_t const m = rand_idx() % mpi_tt->nmodes;
      dims[m] *= primes[p];
    }

    /* rearrange */
    splatt_coord * med = splatt_mpi_rearrange_medium(mpi_tt, dims, mpi);

    /* comm_info basics */
    ASSERT_EQUAL(mpi_tt->nmodes, mpi->nmodes);
    ASSERT_EQUAL(SPLATT_DECOMP_MEDIUM, mpi->decomp);
    for(idx_t m=0; m < mpi_tt->nmodes; ++m) {
      ASSERT_EQUAL(dims[m], mpi->layer_dims[m]);
      if(mpi->layer_comms[m] == MPI_COMM_NULL) {
        ASSERT_FAIL();
      }
    }
    if(mpi->grid_comm == MPI_COMM_NULL) {
      ASSERT_FAIL();
    }
    if(mpi->grid_rank >= mpi->world_npes) {
      ASSERT_FAIL();
    }
    for(idx_t m=0; m < mpi->nmodes; ++m) {
      if(mpi->grid_coords[m] >= mpi->layer_dims[m]) {
        ASSERT_FAIL();
      }
    }

    /* now newly distributed tensor basics */
    //ASSERT_NOT_NULL(med);

    tt_free(med);
    tt_free(mpi_tt);
  }
  splatt_free_comm_info(mpi);
  free(primes);
}


