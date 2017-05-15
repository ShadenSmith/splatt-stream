
#include "../src/ccp/ccp.h"
#include "../src/util.h"
#include "../src/sort.h"

#include "ctest/ctest.h"

#include "splatt_test.h"

#define NUM_CCP_TESTS 7


static idx_t p_get_bneck(
    idx_t const * const restrict weights,
    idx_t const nitems,
    idx_t const * const restrict parts,
    idx_t const nparts)
{
  idx_t bneck = 0;
  idx_t before = 0;
  for(idx_t p=0; p < nparts; ++p) {
    idx_t const right = SS_MIN(parts[p+1]-1, nitems-1);
    idx_t const size  = weights[right] - before;

    bneck = SS_MAX(bneck, size);
    before += size;
  }

  return bneck;
}

    


CTEST_DATA(ccp)
{
  idx_t P;
  idx_t N;
  idx_t * unit_data;
  idx_t * rand_data;
  idx_t * sorted_data;
  idx_t * fororder_data;
  idx_t * revorder_data;
  idx_t * bigend_data;
  idx_t * fibonacci_data;

  idx_t * ptrs[NUM_CCP_TESTS];
};


CTEST_SETUP(ccp)
{
  data->P = 31;

  data->N = 500;
  data->rand_data = malloc(data->N * sizeof(*(data->rand_data)));
  data->sorted_data = malloc(data->N * sizeof(*(data->sorted_data)));
  data->fororder_data = malloc(data->N * sizeof(*(data->fororder_data)));
  data->revorder_data = malloc(data->N * sizeof(*(data->revorder_data)));
  data->bigend_data = malloc(data->N * sizeof(*(data->bigend_data)));
  data->unit_data = malloc(data->N * sizeof(*(data->unit_data)));
  data->fibonacci_data = malloc(data->N * sizeof(*(data->fibonacci_data)));

  for(idx_t x=0; x < data->N; ++x) {
    data->unit_data[x] = 1;
    data->rand_data[x] = rand_idx() % 131;
    data->sorted_data[x] = rand_idx() % 131;
    data->bigend_data[x] = rand_idx() % 131;

    data->fororder_data[x] = x;
    data->revorder_data[x] = data->N - x;

    if(x < 2) {
      data->fibonacci_data[x] = 1;
    } else {
      if(x < 10) {
        data->fibonacci_data[x] = data->fibonacci_data[x-1] +
            data->fibonacci_data[x-2];
      } else {
        data->fibonacci_data[x] = data->fibonacci_data[x-1] + 1000;
      }
    }
  }


  splatt_quicksort(data->sorted_data, data->N);
  data->bigend_data[data->N - 1] = 999;

  data->ptrs[0] = data->rand_data;
  data->ptrs[1] = data->sorted_data;
  data->ptrs[2] = data->fororder_data;
  data->ptrs[3] = data->revorder_data;
  data->ptrs[4] = data->bigend_data;
  data->ptrs[5] = data->unit_data;
  data->ptrs[6] = data->fibonacci_data;
}

CTEST_TEARDOWN(ccp)
{
  for(idx_t t=0; t < NUM_CCP_TESTS; ++t) {
    free(data->ptrs[t]);
  }
}


CTEST2(ccp, prefix_sum_inc)
{
  idx_t * pref = malloc(data->N * sizeof(*pref));

  for(idx_t t=0; t < NUM_CCP_TESTS; ++t) {
    idx_t * const restrict weights = data->ptrs[t];

    idx_t total = 0;
    for(idx_t x=0; x < data->N; ++x) {
      total += weights[x];
    }

    /* make a copy */
    memcpy(pref, weights, data->N * sizeof(*pref));

    prefix_sum_inc(pref, data->N);

    ASSERT_EQUAL(total, pref[data->N - 1]);

    idx_t running = 0;
    for(idx_t x=0; x < data->N; ++x) {
      running += weights[x];
      ASSERT_EQUAL(running, pref[x]);
    }
  }
  free(pref);
}


CTEST2(ccp, prefix_sum_exc)
{
  /* make a copy */
  idx_t * pref = malloc(data->N * sizeof(*pref));

  /* foreach test */
  for(idx_t t=0; t < NUM_CCP_TESTS; ++t) {
    idx_t * const restrict weights = data->ptrs[t];
    memcpy(pref, weights,  data->N * sizeof(*pref));

    prefix_sum_exc(pref, data->N);

    idx_t running = 0;
    for(idx_t x=0; x < data->N; ++x) {
      ASSERT_EQUAL(running, pref[x]);
      running += weights[x];
    }
  }

  free(pref);
}


CTEST2(ccp, partition_1d)
{
  /* foreach test */
  for(idx_t t=0; t < NUM_CCP_TESTS; ++t) {
    idx_t * const restrict weights = data->ptrs[t];

    idx_t * parts = partition_1d(weights, data->N, data->P);
    idx_t bneck = p_get_bneck(weights, data->N, parts, data->P);

    /* check bounds */
    ASSERT_EQUAL(0, parts[0]);
    ASSERT_EQUAL(data->N, parts[data->P]);

    /* check non-overlapping partitions */
    for(idx_t p=0; p < data->P; ++p) {
      ASSERT_TRUE(parts[p] <= parts[p+1]);
    }

    /* check actual optimality */
    ASSERT_FALSE(lprobe(weights, data->N, parts, data->P, bneck-1));

    splatt_free(parts);
  } /* end foreach test */
}


CTEST2(ccp, probe)
{
  idx_t total = 0;
  for(idx_t x=0; x < data->N; ++x) {
    total += data->rand_data[x];
  }

  idx_t * parts = splatt_malloc((1 + data->P) * sizeof(*parts));

  prefix_sum_exc(data->rand_data, data->N);
  bool result = lprobe(data->rand_data, data->N, parts, data->P,
      (total / data->P) - 1);
  ASSERT_EQUAL(false, result);

  idx_t bottleneck = total / data->P;
  while(!result) {
    result = lprobe(data->rand_data, data->N, parts, data->P, bottleneck);
    ++bottleneck;
  }
  --bottleneck;

  /* check bounds */
  ASSERT_EQUAL(0, parts[0]);
  ASSERT_EQUAL(data->N, parts[data->P]);

  /* check non-overlapping partitions */
  for(idx_t p=1; p < data->P; ++p) {
    ASSERT_TRUE(parts[p] >= parts[p-1]);
  }

  /* check actual bneck */
  for(idx_t p=1; p < data->P; ++p) {
    ASSERT_TRUE(parts[p] - parts[p-1] <= bottleneck);
  }

  splatt_free(parts);
}


CTEST2_SKIP(ccp, bigpart)
{
  idx_t const N = 25000000;
  idx_t const P = 24;

  idx_t * weights = splatt_malloc(N * sizeof(*weights));
  for(idx_t x=0; x < N; ++x) {
    weights[x] = rand_idx() % 100;
  }

  sp_timer_t part;
  timer_fstart(&part);
  idx_t * parts = partition_1d(weights, N, P);
  idx_t bneck = p_get_bneck(weights, N, parts, P);
  timer_stop(&part);

  /* correctness */
  bool success;
  success = lprobe(weights, N, parts, P, bneck);
  ASSERT_EQUAL(true, success);
  success = lprobe(weights, N, parts, P, bneck-1);
  ASSERT_EQUAL(false, success);

  splatt_free(weights);
  splatt_free(parts);
}


CTEST2(ccp, part_equalsize)
{
  idx_t const P = 24;
  idx_t const CHUNK = 10000;
  idx_t const N = P * CHUNK;

  idx_t * weights = splatt_malloc(N * sizeof(*weights));

  for(idx_t x=0; x < N; ++x) {
    weights[x] = 1;
  }

  idx_t * parts = partition_1d(weights, N, P);
  idx_t bneck = p_get_bneck(weights, N, parts, P);
  ASSERT_EQUAL(CHUNK, bneck);

  lprobe(weights, N, parts, P, bneck);

  for(idx_t p=0; p < P; ++p) {
    ASSERT_EQUAL(CHUNK * p, parts[p]);
  }
  ASSERT_EQUAL(N, parts[P]);

  splatt_free(weights);
  splatt_free(parts);
}


/*
 * Just a simple handmade example to ensure every output value is what it
 * should be. [2, 3, 4, 6] is chosen to ensure that the optimal partitioning
 * is not uniform in size: {2,3,4} and {6} is the optimal partitioning for two
 * parts.
 */
CTEST(ccp, easy)
{
  idx_t data[] = { 2, 3, 4, 6 };

  idx_t * part = partition_1d(data, 4, 2);

  ASSERT_EQUAL(2,  data[0]);
  ASSERT_EQUAL(5,  data[1]);
  ASSERT_EQUAL(9,  data[2]);
  ASSERT_EQUAL(15, data[3]);

  ASSERT_EQUAL(0, part[0]);
  ASSERT_EQUAL(3, part[1]);
  ASSERT_EQUAL(4, part[2]);

  ASSERT_EQUAL(9, p_get_bneck(data, 4, part, 2));

  splatt_free(part);
}
