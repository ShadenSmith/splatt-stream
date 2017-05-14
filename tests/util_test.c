
#include "ctest/ctest.h"
#include "splatt_test.h"
#include "../src/util.h"



/* gracefully handle passing 0 to get_primes() */
CTEST(util, get_primes_zero)
{
  int nprimes = 100;

  int * primes = get_primes(0, &nprimes);
  ASSERT_NULL(primes);
  ASSERT_EQUAL(0, nprimes);
}


/* handle some simple known primes */
CTEST(util, get_primes_primes)
{
  int nprimes = 0;
  int * primes;
  
  primes = get_primes(2, &nprimes);
  ASSERT_NOT_NULL(primes);
  ASSERT_EQUAL(1, nprimes);
  ASSERT_EQUAL(2, primes[0]);
  free(primes);


  primes = get_primes(31, &nprimes);
  ASSERT_NOT_NULL(primes);
  ASSERT_EQUAL(1, nprimes);
  ASSERT_EQUAL(31, primes[0]);
  free(primes);
}


/* try a lot of numbers */
CTEST(util, get_primes)
{
  for(int x=1; x < 20000; ++x) {
    int nprimes = 0;
    int * primes = get_primes(x, &nprimes);

    ASSERT_NOT_NULL(primes);

    /* first ensure the primes actually reconstruct the number */
    int reconstructed = 1;
    for(int p=0; p < nprimes; ++p) {
      reconstructed *= primes[p];
    }
    ASSERT_EQUAL(x, reconstructed);

    /* now actually make sure each 'prime' is prime */
    for(int p=0; p < nprimes; ++p) {
      int nsubprimes = 0;
      int * subprimes = get_primes(primes[p], &nsubprimes);
      ASSERT_EQUAL(1, nsubprimes);
      ASSERT_EQUAL(primes[p], subprimes[0]);
      free(subprimes);
    }

    free(primes);
  }
}



