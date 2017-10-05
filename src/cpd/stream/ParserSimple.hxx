#ifndef SPLATT_CPD_STREAM_PARSER_SIMPLE_HXX
#define SPLATT_CPD_STREAM_PARSER_SIMPLE_HXX

#include "ParserBase.hxx"

#include <unordered_map>


extern "C" {
#include "../../sptensor.h"
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
  idx_t mode_length(idx_t which_mode);
  idx_t num_modes();

  /* invert mapping to get original indices */
  inline idx_t lookup_ind(idx_t mode, idx_t ind) { return _ind_maps_inv[mode][ind]; }

private:
  sptensor_t * _tensor;

  /* mapping of original indices to seen ones */
  std::unordered_map<idx_t, idx_t> _ind_maps[SPLATT_MAX_NMODES];
  std::unordered_map<idx_t, idx_t> _ind_maps_inv[SPLATT_MAX_NMODES];

  /* batch state */
  idx_t _batch_num;
  idx_t _nnz_ptr;
};

#endif
