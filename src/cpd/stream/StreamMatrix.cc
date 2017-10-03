

#include "StreamMatrix.hxx"

extern "C" {
#include "../../util.h"
}


StreamMatrix::StreamMatrix(
    idx_t rank
) :
    _nrows(0),
    _ncols(rank),
    _row_capacity(128),
    _mat(NULL)
{
  _mat = mat_alloc(_row_capacity, rank);
}

StreamMatrix::~StreamMatrix()
{
  mat_free(_mat);
}


void StreamMatrix::reserve(
    idx_t new_rows)
{
  idx_t const old_capacity = _row_capacity;

  /* double until we can fit all of the requested rows */
  while(new_rows > _row_capacity) {
    _row_capacity *= 2;
  }

  if(old_capacity < _row_capacity) {
    matrix_t * newmat = mat_alloc(new_rows, _ncols);
    par_memcpy(newmat->vals, _mat->vals, _nrows * _ncols * sizeof(*newmat->vals));
    mat_free(_mat);
    _mat = newmat;
  }
}
