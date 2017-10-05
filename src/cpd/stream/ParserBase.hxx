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

  /* return the length of the mode (over all time). if unknown, return 0 */
  virtual idx_t mode_length(idx_t which_mode) = 0;

  virtual idx_t num_modes() = 0;

  virtual idx_t lookup_ind(idx_t mode, idx_t ind) { return ind; }


protected:
  idx_t _stream_mode;

};

#endif
