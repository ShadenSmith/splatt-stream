/**
* @file include/splatt/mpi.h
* @brief Functions for distributed-memory SPLATT (MPI).
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/


#ifndef SPLATT_INCLUDE_MPI_H
#define SPLATT_INCLUDE_MPI_H



/**
* @brief Tensor decomposition schemes.
*/
typedef enum
{
  /** @brief Coarse-grained decomposition is using a separate 1D decomposition
   *         for each mode. */
  SPLATT_DECOMP_COARSE,
  /** @brief Medium-grained decomposition is an 'nmodes'-dimensional
   *         decomposition. */
  SPLATT_DECOMP_MEDIUM,
  /** @brief Fine-grained decomposition distributes work at the nonzero level.
   *         NOTE: requires a partitioning on the nonzeros. */
  SPLATT_DECOMP_FINE
} splatt_decomp_type;


/**
* @brief Communication pattern type. We support point-to-point, and all-to-all
*        (vectorized).
*/
typedef enum
{
  /**
  * @brief Use point-to-point communications (`MPI_Isend`/`MPI_Irecv`) during
  *        major exchanges.
  */
  SPLATT_COMM_POINT2POINT,

  /**
  * @brief Use personalized all-to-all communications (`MPI_Alltoallv`) during
  *        major exchanges.
  */
  SPLATT_COMM_ALL2ALL
} splatt_comm_type;




/**
* @brief Opaque type for MPI communication.
*/
typedef struct _splatt_comm_info splatt_comm_info;



#ifdef __cplusplus
extern "C" {
#endif


/**
\defgroup api_mpi_list List of functions for \splatt MPI.
@{
*/




/**
* @brief Free the memory allocated by `splatt_alloc_comm_info()`.
*
*        NOTE: this function exists whether MPI is enabled or not -- it mildly
*        simplifies SPLATT internals.
*
* @param comm_info The object to free.
*/
void splatt_free_comm_info(
    splatt_comm_info * comm_info);



/*
 * Dummy MPI implementations for when MPI is not enabled.
 */
#ifndef SPLATT_USE_MPI
/**
* @brief Allocate a `splatt_comm_info` structure. This data must be freed with
*        `splatt_free_comm_info()`.
*
*        NOTE: this overloaded API function is only exposed when MPI is *not*
*        enabled.  Otherwise this function accepts an `MPI_Comm` parameter.
*
* @return A new `splatt_comm_info` object.
*/
splatt_comm_info * splatt_alloc_comm_info();
#endif



#ifdef SPLATT_USE_MPI


/**
* @brief Allocate a `splatt_comm_info` structure. This data must be freed with
*        `splatt_free_comm_info()`.
*
* @param comm The MPI communicator to work from (the communicator will be
*             duplicated).
*
* @return A new `splatt_comm_info` object.
*/
splatt_comm_info * splatt_alloc_comm_info(
    MPI_Comm comm);


/**
* @brief Read a tensor from a file, distribute among an MPI communicator, and
*        convert to CSF format.
*
* @param fname The filename to read from.
* @param[out] nmodes SPLATT will fill in the number of modes found.
* @param[out] tensors An array of splatt_csf structure(s). Allocation scheme
*                follows opts[SPLATT_OPTION_CSF_ALLOC].
* @param options An options array allocated by splatt_default_opts(). The
*                distribution scheme follows opts[SPLATT_OPTION_DECOMP].
* @param comm The MPI communicator to distribute among.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_mpi_csf_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_csf ** tensors,
    double const * const options,
    MPI_Comm comm);



/**
* @brief Load a tensor in coordinate from from a file and distribute it among
*        an MPI communicator.
*
* @param fname The file to read from.
* @param[out] nmodes The number of modes in the tensor.
* @param[out] nnz The number of nonzeros in my portion.
* @param[out] inds An array of indices for each mode.
* @param[out] vals The tensor nonzero values.
* @param options SPLATT options array. Currently unused.
* @param comm Which communicator to distribute among.
*
* @return SPLATT error code (splatt_error_t). SPLATT_SUCCESS on success.
*/
int splatt_mpi_coord_load(
    char const * const fname,
    splatt_idx_t * nmodes,
    splatt_idx_t * nnz,
    splatt_idx_t *** inds,
    splatt_val_t ** vals,
    double const * const options,
    MPI_Comm comm);




#endif /* if mpi */

/** @} */

#ifdef __cplusplus
}
#endif



#endif
