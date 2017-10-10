
#include "ParserSimple.hxx"


extern "C" {
#include "../../io.h"
#include "../../sort.h"
#include "../../util.h"
}


ParserSimple::ParserSimple(
    std::string filename,
    idx_t stream_mode
) : ParserBase(stream_mode),
    _batch_num(0),
    _nnz_ptr(0)
{
  _tensor = tt_read_file(filename.c_str());

  if(_stream_mode >= _tensor->nmodes) {
    fprintf(stderr, "ERROR: streaming mode %" SPLATT_PF_IDX
                    " is larger than # modes.\n",
        _stream_mode + 1);
    exit(1);
  }

  /* sort tensor by streamed mode */
  tt_sort(_tensor, _stream_mode, NULL);
}


ParserSimple::~ParserSimple()
{
  tt_free(_tensor);
}


sptensor_t * ParserSimple::next_batch()
{
  idx_t const * const restrict streaming_inds = _tensor->ind[_stream_mode];

  /* if we have already reached the end */
  if(_nnz_ptr == _tensor->nnz) {
    return NULL;
  }
  
  /* find starting nnz */
  idx_t start_nnz = _nnz_ptr;
  while((start_nnz < _tensor->nnz) && (streaming_inds[start_nnz] < _batch_num)) {
    ++start_nnz;
  }

  /* find ending nnz */
  idx_t end_nnz = start_nnz;
  while((end_nnz < _tensor->nnz) && (streaming_inds[end_nnz] < _batch_num + 1)) {
    ++end_nnz;
  }

  idx_t const nnz = end_nnz - start_nnz;

  /* end of stream */
  assert(nnz > 0);

  /* copy into new tensor */
  sptensor_t * ret = tt_alloc(nnz, _tensor->nmodes);
  par_memcpy(ret->vals, &(_tensor->vals[start_nnz]), nnz * sizeof(*(ret->vals)));

  /* streaming mode is just 0s */
  #pragma omp parallel for schedule(static)
  for(idx_t x=0; x < nnz; ++x) {
    ret->ind[_stream_mode][x] = 0;
  }

  /* non-streaming modes use _ind_maps */
  #pragma omp parallel for schedule(static, 1)
  for(idx_t m=0; m < _tensor->nmodes; ++m) {
    if(m == _stream_mode) {
      ret->dims[_stream_mode] = 1;
      continue;
    }

    for(idx_t x=0; x < nnz; ++x) {
      idx_t const orig_ind = _tensor->ind[m][start_nnz + x];

      /* insert if not found */
      if(_ind_maps[m].find(orig_ind) == _ind_maps[m].end()) {
        size_t const size = _ind_maps[m].size();
        _ind_maps[m][orig_ind] = size;
        _ind_maps_inv[m][size] = orig_ind;
      }
      /* map original index to increasing one */
      ret->ind[m][x] = _ind_maps[m][orig_ind];
    }

    ret->dims[m] = _ind_maps[m].size();
  }

  /* update state for next batch */
  _nnz_ptr = end_nnz;
  ++_batch_num;

  return ret;
}

idx_t ParserSimple::num_modes()
{
  return _tensor->nmodes;
}

idx_t ParserSimple::mode_length(
    idx_t which_mode)
{
  if(which_mode < _tensor->nmodes) {
    return _ind_maps[which_mode].size();

  } else {
    fprintf(stderr, "ERROR: requesting mode %lu of %lu\n",
        which_mode+1, _tensor->nmodes);
    return 0;
  }
}


sptensor_t * ParserSimple::full_stream()
{
  return _tensor;
}


sptensor_t * ParserSimple::stream_until(idx_t time)
{
  idx_t nnz = 0;
  while((nnz < _tensor->nnz) &&  _tensor->ind[_stream_mode][nnz] < time) {
    ++nnz;
  }

  sptensor_t * ret = tt_alloc(nnz, _tensor->nmodes);
  par_memcpy(ret->vals, _tensor->vals, nnz * sizeof(*(ret->vals)));
  for(idx_t m=0; m < _tensor->nmodes; ++m) {
    par_memcpy(ret->ind[m], _tensor->ind[m], nnz * sizeof(idx_t));

    ret->dims[m] = _tensor->dims[m];
  }

  ret->dims[_stream_mode] = time;

  return ret;
}



