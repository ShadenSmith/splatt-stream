
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../splatt_mpi.h"

#include "comm_info.h"



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/



sptensor_t * mpi_rearrange_by_part(
  sptensor_t const * const ttbuf,
  int const * const parts,
  MPI_Comm comm)
{
  int rank, npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  /* count how many to send to each process */
  int * nsend = calloc(npes, sizeof(*nsend));
  int * nrecv = calloc(npes, sizeof(*nrecv));
  #pragma omp parallel for schedule(static)
  for(idx_t n=0; n < ttbuf->nnz; ++n) {
    #pragma omp atomic
    ++nsend[parts[n]];
  }
  MPI_Alltoall(nsend, 1, MPI_INT, nrecv, 1, MPI_INT, comm);

  idx_t send_total = 0;
  idx_t recv_total = 0;
  for(int p=0; p < npes; ++p) {
    send_total += nsend[p];
    recv_total += nrecv[p];
  }
  assert(send_total == ttbuf->nnz);

  /* how many nonzeros I'll own */
  idx_t const nowned = recv_total;

  int * send_disp = splatt_malloc((npes+1) * sizeof(*send_disp));
  int * recv_disp = splatt_malloc((npes+1) * sizeof(*recv_disp));

  /* recv_disp is const so we'll just fill it out once */
  recv_disp[0] = 0;
  for(int p=1; p <= npes; ++p) {
    recv_disp[p] = recv_disp[p-1] + nrecv[p-1];
  }

  /* allocate my tensor and send buffer */
  sptensor_t * tt = tt_alloc(nowned, ttbuf->nmodes);
  idx_t * isend_buf = splatt_malloc(ttbuf->nnz * sizeof(*isend_buf));

  /* rearrange into sendbuf and send one mode at a time */
  for(idx_t m=0; m < ttbuf->nmodes; ++m) {
    /* prefix sum to make disps */
    send_disp[0] = send_disp[1] = 0;
    for(int p=2; p <= npes; ++p) {
      send_disp[p] = send_disp[p-1] + nsend[p-2];
    }

    idx_t const * const ind = ttbuf->ind[m];
    for(idx_t n=0; n < ttbuf->nnz; ++n) {
      idx_t const index = send_disp[parts[n]+1]++;
      isend_buf[index] = ind[n];
    }

    /* exchange indices */
    MPI_Alltoallv(isend_buf, nsend, send_disp, SPLATT_MPI_IDX,
                  tt->ind[m], nrecv, recv_disp, SPLATT_MPI_IDX,
                  comm);
  }
  splatt_free(isend_buf);

  /* lastly, rearrange vals */
  val_t * vsend_buf = splatt_malloc(ttbuf->nnz * sizeof(*vsend_buf));
  send_disp[0] = send_disp[1] = 0;
  for(int p=2; p <= npes; ++p) {
    send_disp[p] = send_disp[p-1] + nsend[p-2];
  }

  val_t const * const vals = ttbuf->vals;
  for(idx_t n=0; n < ttbuf->nnz; ++n) {
    idx_t const index = send_disp[parts[n]+1]++;
    vsend_buf[index] = vals[n];
  }
  /* exchange vals */
  MPI_Alltoallv(vsend_buf, nsend, send_disp, SPLATT_MPI_VAL,
                tt->vals,  nrecv, recv_disp, SPLATT_MPI_VAL,
                comm);
  splatt_free(vsend_buf);
  splatt_free(send_disp);
  splatt_free(recv_disp);

  /* allocated with calloc */
  free(nsend);
  free(nrecv);

  return tt;
}





/******************************************************************************
 * API FUNCTONS
 *****************************************************************************/



