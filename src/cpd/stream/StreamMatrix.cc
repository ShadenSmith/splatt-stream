

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


void StreamMatrix::grow(
    idx_t new_rows)
{
  if(new_rows < _nrows) {
    return;
  }

  /* grow allocation */
  if(new_rows >= _row_capacity) {
    /* double until we can fit all of the requested rows */
    while(new_rows > _row_capacity) {
      _row_capacity *= 2;
    }

    matrix_t * newmat = mat_alloc(_row_capacity, _ncols);
    par_memcpy(newmat->vals, _mat->vals, _nrows * _ncols * sizeof(*newmat->vals));
    mat_free(_mat);
    _mat = newmat;
  }

  /* fill in new rows with initialized values */
#if 1
  fill_rand(&(_mat->vals[_nrows * _ncols]), (new_rows - _nrows) * _ncols);
#else
  /* zero values? */
  for(idx_t x=_nrows * _ncols; x < new_rows * _ncols; ++x) {
    _mat->vals[x] = 0.;
  }
#endif

  /* store new number of rows */
  _nrows = new_rows;
  //_mat->I = _nrows;
}

