#ifndef SPLATT_CPD_STREAM_PARSER_SIMPLE_HXX
#define SPLATT_CPD_STREAM_PARSER_SIMPLE_HXX

#include "ParserBase.hxx"

extern "C" {
#include "../../sptensor.h"
#include "../../reorder.h"
}

class ParserSimple : public ParserBase
{
public:
  ParserSimple(
      std::string filename,
      idx_t stream_mode
  );
  ~ParserSimple();

  sptensor_t * next_batch();
  idx_t num_modes();

  /* invert mapping to get original indices */
  inline idx_t lookup_ind(idx_t mode, idx_t ind) { return _perm->iperms[mode][ind]; }

  inline idx_t * perm(idx_t mode) { return _perm->perms[mode]; }
  inline idx_t * iperm(idx_t mode) { return _perm->iperms[mode]; }

  bool last_batch();

  sptensor_t * full_stream();
  sptensor_t * stream_until(idx_t time);
  sptensor_t * stream_prev(idx_t previous);

private:
  sptensor_t * _tensor;

  idx_t _prev_dim[SPLATT_MAX_NMODES];

  permutation_t * _perm;

  /* batch state */
  idx_t _batch_num;
  idx_t _nnz_ptr;
};

#endif
