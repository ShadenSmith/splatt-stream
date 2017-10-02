#ifndef SPLATT_CPD_STREAM_STREAMPARSER_BASE_HXX
#define SPLATT_CPD_STREAM_STREAMPARSER_BASE_HXX

#include <string>

extern "C" {
#include "../../base.h"
#include "../../sptensor.h"
}

class StreamParserBase
{
public:
  StreamParserBase(
      idx_t stream_mode
  );
  ~StreamParserBase();

  virtual sptensor_t * next_batch() = 0;

protected:
  idx_t _stream_mode;

};

#endif
