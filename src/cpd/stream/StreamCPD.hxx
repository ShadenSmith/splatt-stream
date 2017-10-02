#ifndef SPLATT_CPD_STREAM_STREAMCPD_HXX
#define SPLATT_CPD_STREAM_STREAMCPD_HXX

extern "C" {
#include "../../base.h"
#include "../../sptensor.h"
}

class StreamCPD
{
public:
  StreamCPD(
      StreamParserBase * source
  );

  ~StreamCPD();

private:
  StreamParserBase  * _source;
};

#endif
