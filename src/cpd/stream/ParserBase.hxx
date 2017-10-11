#ifndef SPLATT_CPD_STREAM_PARSER_BASE_HXX
#define SPLATT_CPD_STREAM_PARSER_BASE_HXX

#include <string>

extern "C" {
#include "../../base.h"
#include "../../sptensor.h"
}

class ParserBase
{
public:
  ParserBase(
      idx_t stream_mode
  );
  ~ParserBase();

  /* return the next batch of non-zeros */
  virtual sptensor_t * next_batch() = 0;

  idx_t stream_mode();

  virtual idx_t num_modes() = 0;

  virtual idx_t * iperm(idx_t mode) { return NULL; }
  virtual idx_t lookup_ind(idx_t mode, idx_t ind) { return ind; }

  virtual sptensor_t * full_stream() { return NULL; }

  virtual sptensor_t * stream_until(idx_t time) { return NULL; }


protected:
  idx_t _stream_mode;

};

#endif
