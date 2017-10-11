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
      splatt_cpd_opts * const cpd_opts,
      splatt_global_opts const * const global_opts);

private:
  /*
   * methods
   */
  splatt_kruskal * get_kruskal();

  splatt_kruskal * get_prev_kruskal(idx_t previous);

  double compute_errorsq(idx_t num_previous);

  /*
   * Grow factor matrices and update Gram matrices
   */
  void grow_mats(idx_t const * const new_dims);

  /*
   * Add the historical data to the MTTKRP output before ADMM. */
  void add_historical(idx_t const mode);


  /*
   * vars
   */
  cpd_ws * _cpd_ws;

  ParserBase  * _source;

  StreamMatrix * _mttkrp_buf;
  StreamMatrix * _stream_mats_new[MAX_NMODES];
  StreamMatrix * _stream_mats_old[MAX_NMODES];
  /* ADMM */
  StreamMatrix * _stream_init;
  StreamMatrix * _stream_auxil;
  StreamMatrix * _stream_duals[MAX_NMODES];

  /* Stores complete streaming matrix -- TODO toggle */
  StreamMatrix * _global_time;

  matrix_t * _old_gram;

  matrix_t * _mat_ptrs[MAX_NMODES+1];

  idx_t _nmodes;
  idx_t _rank;
  idx_t _stream_mode;
};

#endif
