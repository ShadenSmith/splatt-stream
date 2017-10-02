#ifndef SPLATT_CPD_STREAM_STREAMPARSER_SIMPLE_HXX
#define SPLATT_CPD_STREAM_STREAMPARSER_SIMPLE_HXX

#include "StreamParserBase.hxx"


extern "C" {
#include "../../sptensor.h"
}

class StreamParserSimple : StreamParserBase
{
public:
  StreamParserSimple(
      std::string filename,
      idx_t stream_mode
  );

  ~StreamParserSimple();

  void next_batch();

private:
  sptensor_t * _tensor;

};

#endif
