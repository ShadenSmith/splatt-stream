
#include "../src/sptensor.h"

#include "ctest/ctest.h"

#include "splatt_test.h"


CTEST_DATA(sptensor)
{
  idx_t ntensors;
  sptensor_t * tensors[MAX_DSETS];
};


CTEST_SETUP(sptensor)
{
  data->ntensors = sizeof(datasets) / sizeof(datasets[0]);
  for(idx_t i=0; i < data->ntensors; ++i) {
    data->tensors[i] = tt_read(datasets[i]);
  }
}

CTEST_TEARDOWN(sptensor)
{
  for(idx_t i=0; i < data->ntensors; ++i) {
    tt_free(data->tensors[i]);
  }
}


CTEST2(sptensor, splatt_alloc_coord)
{
  splatt_coord * coord = splatt_alloc_coord();

  ASSERT_NOT_NULL(coord);

  ASSERT_EQUAL(0, coord->nmodes);
  ASSERT_EQUAL(0, coord->nnz);

  ASSERT_NULL(coord->vals);
  for(idx_t m=0; m < SPLATT_MAX_NMODES; ++m) {
    ASSERT_NULL(coord->ind[m]);
  }

  splatt_free_coord(coord);
}


