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

  sptensor_t * next_batch();

private:
  sptensor_t * _tensor;

  idx_t _batch_num;
  idx_t _nnz_ptr;
};

#endif
