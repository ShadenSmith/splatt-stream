#ifndef SPLATT_CPD_STREAM_STREAMCPD_HXX
#define SPLATT_CPD_STREAM_STREAMCPD_HXX

extern "C" {
#include "../../base.h"
#include "../../sptensor.h"
#include "../cpd.h"
}


#include "ParserBase.hxx"
#include "StreamMatrix.hxx"

class StreamCPD
{
public:
  StreamCPD(
      ParserBase * source
  );
  ~StreamCPD();


  splatt_kruskal *  compute(
      splatt_idx_t const rank,
      double const forget,
      splatt_cpd_opts const * const cpd_opts,
      splatt_global_opts const * const global_opts);

private:
  splatt_kruskal * get_kruskal(StreamMatrix * * mats);

  ParserBase  * _source;

  idx_t _nmodes;
  idx_t _rank;
  idx_t _stream_mode;
};

#endif
