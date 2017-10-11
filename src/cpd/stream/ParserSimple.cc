
#include "ParserSimple.hxx"


extern "C" {
#include "../../io.h"
#include "../../reorder.h"
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

  /* Construct permutation */
  _perm = perm_alloc(_tensor->dims, _tensor->nmodes);
  #pragma omp parallel for schedule(static, 1)
  for(idx_t m=0; m < _tensor->nmodes; ++m) {
    idx_t * const restrict perm  = _perm->perms[m];
    idx_t * const restrict iperm = _perm->iperms[m];

    /* streaming mode is sorted and just gets identity */
    if(m == stream_mode) {
      for(idx_t i=0; i < _tensor->dims[m]; ++i) {
        perm[i]  = i;
        iperm[i] = i;
      }
      continue;
    }

    /* initialize */
    for(idx_t i=0; i < _tensor->dims[m]; ++i) {
      perm[i]  = SPLATT_IDX_MAX;
      iperm[i] = SPLATT_IDX_MAX;
    }


    /*
     * Relabel tensor.
     */

    idx_t seen = 0; /* current index */

    idx_t * const restrict inds = _tensor->ind[m];
    for(idx_t x=0; x < _tensor->nnz; ++x) {
      idx_t const ind = inds[x];

      /* if this is the first appearance of ind */
      if(perm[ind] == SPLATT_IDX_MAX) {
        perm[ind]   = seen;
        iperm[seen] = ind;
        ++seen;
      }

      inds[x] = perm[ind];
    }
  } /* foreach mode */
}


ParserSimple::~ParserSimple()
{
  tt_free(_tensor);
  perm_free(_perm);
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

  for(idx_t m=0; m < _tensor->nmodes; ++m) {
    if(m == _stream_mode) {
      ret->dims[_stream_mode] = 1;
      continue;
    }

    par_memcpy(ret->ind[m], &(_tensor->ind[m][start_nnz]),
        nnz * sizeof(**(ret->ind)));

    idx_t dim = 0;
    #pragma omp parallel for schedule(static) reduction(max: dim)
    for(idx_t x=0; x < nnz; ++x) {
      dim = SS_MAX(dim, ret->ind[m][x]);
    }

    ret->dims[m] = dim + 1;
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



