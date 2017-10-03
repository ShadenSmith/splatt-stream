
#include "StreamCPD.hxx"
#include "StreamMatrix.hxx"

StreamCPD::StreamCPD(
    ParserBase * source
) :
    _source(source)
{
}

StreamCPD::~StreamCPD()
{
}


void StreamCPD::compute(
    splatt_idx_t const rank,
    double const forget,
    splatt_cpd_opts const * const cpd_opts,
    splatt_global_opts const * const global_opts)
{
  StreamMatrix * mats[SPLATT_MAX_NMODES];
  for(idx_t m=0; m < _source->num_modes(); ++m) {
    mats[m] = new StreamMatrix(rank);
  }

  /* Stream */
  idx_t it = 0;
  sptensor_t * batch = _source->next_batch();
  while(batch != NULL) {
    printf("batch %5lu: %lu nnz\n", it+1, batch->nnz);

    /* resize factor matrices if necessary */
    for(idx_t m=0; m < _source->num_modes(); ++m) {
      if(m != _source->stream_mode()) {
        mats[m]->reserve(_source->mode_length(m));
      }
    }

    /* prepare for next batch */
    tt_free(batch);
    batch = _source->next_batch();
    ++it;
  }

  for(idx_t m=0; m < _source->num_modes(); ++m) {
    delete mats[m];
  }
}

