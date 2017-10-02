
#include "StreamParserSimple.hxx"


extern "C" {
#include "../../io.h"
#include "../../sort.h"
#include "../../util.h"
}


StreamParserSimple::StreamParserSimple(
    std::string filename,
    idx_t stream_mode
) : StreamParserBase(stream_mode),
    _batch_num(0),
    _nnz_ptr(0)
{
  _tensor = tt_read_file(filename.c_str());

  /* sort tensor by streamed mode */
  tt_sort(_tensor, _stream_mode, NULL);
}


StreamParserSimple::~StreamParserSimple()
{
  tt_free(_tensor);
}


sptensor_t * StreamParserSimple::next_batch()
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

  sptensor_t * ret = tt_alloc(nnz, _tensor->nmodes);
  for(idx_t m=0; m < _tensor->nmodes; ++m) {
    par_memcpy(ret->ind[m], &(_tensor->ind[m][start_nnz]),
        nnz * sizeof(**(ret->ind)));
  }
  par_memcpy(ret->vals, &(_tensor->vals[start_nnz]), nnz * sizeof(*(ret->vals)));

  _nnz_ptr = end_nnz;
  ++_batch_num;

  return ret;
}


