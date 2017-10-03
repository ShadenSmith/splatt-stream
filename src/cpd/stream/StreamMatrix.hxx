#ifndef SPLATT_CPD_STREAM_STREAMMATRIX_HXX
#define SPLATT_CPD_STREAM_STREAMMATRIX_HXX

extern "C" {
#include "../../matrix.h"
}


class StreamMatrix
{
public:
  StreamMatrix(idx_t rank);
  ~StreamMatrix();

  void reserve(idx_t nrows);

  inline matrix_t * mat() { return _mat; };
  inline val_t * vals() { return _mat->vals; };
  inline idx_t num_rows() { return _nrows; };

private:
  idx_t _nrows;
  idx_t _ncols;
  idx_t _row_capacity;

  matrix_t * _mat;
};

#endif
