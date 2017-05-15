/**
* @file splatt_mpi.h
* @brief Internal functions for distributed-memory SPLATT.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-05-10
*/


#ifndef SPLATT_MPI_H
#define SPLATT_MPI_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"



/******************************************************************************
 * MPI DISABLED
 *****************************************************************************/
# ifndef SPLATT_USE_MPI
/* Just a dummy for when MPI is not enabled. */
typedef struct
{
  int rank;
} rank_info;


/**
* @brief Dummy implementation of `splatt_comm_info` for when MPI is not
*        enabled.
*/
struct _splatt_comm_info
{
  /**
  * @brief My rank in comm_world.
  */
  int world_rank;

  /**
  * @brief The number of MPI ranks in comm_world.
  */
  int world_npes;
};


# else


/******************************************************************************
 * FULL MPI SUPPORT
 *****************************************************************************/











/******************************************************************************
 * PUBLIC STRUCTURES
 *****************************************************************************/


/**
* @brief Implementation of `splatt_comm_info`. This structure holds data needed
*        to maintain a distributed tensor and to also compute factorizations.
*/
struct _splatt_comm_info
{
  /**
  * @brief The MPI communicator that all process live within.
  */
  MPI_Comm world_comm;

  /**
  * @brief My rank in comm_world.
  */
  int world_rank;

  /**
  * @brief The number of MPI ranks in comm_world.
  */
  int world_npes;


  /*
   * Global tensor data.
   */

  /**
  * @brief The number of modes in the distributed tensor.
  */
  splatt_idx_t nmodes;

  /**
  * @brief The number of non-zeros in the globally-distributed tensor.
  */
  splatt_idx_t global_nnz;

  /**
  * @brief The dimensions of the globally-distributed tensor.
  */
  splatt_idx_t global_dims[SPLATT_MAX_NMODES];




  /*
   * Decomposition information.
   */

  /**
  * @brief The type of decomposition (e.g., medium-grained).
  */
  splatt_decomp_type decomp;

  /**
  * @brief The communication pattern to use (e.g., point-to-point).
  */
  splatt_comm_type comm_type;

  /**
  * @brief The communicator resulting from arranging comm_world in a Cartesian
  *        grid.
  */
  MPI_Comm grid_comm;

  /**
  * @brief My new rank in grid_comm.
  */
  int grid_rank;

  /**
  * @brief My coordinate in grid_comm (in terms of MPI ranks).
  */
  int grid_coords[SPLATT_MAX_NMODES];

  /**
  * @brief Dimensions of grid decomposition (in terms of MPI ranks). If = 0,
  *        decomposition has not yet been established. Coarse- and fine-grained
  *        decompositions have all modes=1.
  */
  int layer_dims[SPLATT_MAX_NMODES];

  /**
  * @brief MPI communicators for each layer. Will be MPI_COMM_NULL if
  *        decomposition has not yet been established.
  */
  MPI_Comm layer_comms[SPLATT_MAX_NMODES];


  /**
  * @brief The number of MPI ranks in my layer. This should be 
  *        prod(layer_dims[:]) / layer_dims[m].
  */
  int layer_size[SPLATT_MAX_NMODES];

  /**
  * @brief My MPI rank in my layer (layer_rank[m] is my rank in communicator
  *        layer_comm[m]).
  */
  int layer_rank[SPLATT_MAX_NMODES];

  /**
  * @brief Marks the start/end of layers in terms of global_dims[].
  *
  *        layer_ptrs[m] is an array of of length layer_dims[m]+1, and entry
  *        layer_ptrs[m][d] is the end of layer[m].
  */
  splatt_idx_t * layer_ptrs[SPLATT_MAX_NMODES];


  /**
  * @brief Marks the start/end of the matrix rows assigned to each rank in the
  *        layer, in terms of global_dims[].
  *
  *        mat_ptrs[m] is an array of length (layer_size[m]+1), and rank p
  *        owns [mat_ptrs[m][p], mat_ptrs[m][p+1]). Note that p is a rank
  *        within communicator layer_comm[m].
  */
  splatt_idx_t * mat_ptrs[SPLATT_MAX_NMODES];

  /**
  * @brief The first row that I own in the layer. This is simply an alias for
  *        mat_ptrs[m][layer_rank[m]].
  */
  splatt_idx_t mat_start[SPLATT_MAX_NMODES];

  /**
  * @brief The last row that I own in the layer. This is simply an alias for
  *        mat_ptrs[m][layer_rank[m] + 1].
  */
  splatt_idx_t mat_end[SPLATT_MAX_NMODES];

  /**
  * @brief Whether we compress coordinate system to local with `indmap` mapping
   *       back to global. This is useful when the tensor mode is very sparse
   *       and messages are small. When 'dense' communication is more likely,
   *       or packing/unpacking buffers is relatively expensive, this should be
   *       set to false before distributing a tensor.
  */
  bool compress[SPLATT_MAX_NMODES];

  /**
  * @brief Mapping of local-to-global indices if a tensor mode is compressed.
  *        indmap[m][i] specifies the index in mode 'm' of local index 'i',
  *        mapping back into global_dims[m].
  *
  *        If compress[m] is false, indmap[m] is NULL.
  */
  splatt_idx_t * indmap[SPLATT_MAX_NMODES];



  /*
   * Communication structures.
   */

  /**
  * @brief Requests used for various `MPI_Isend()` calls. This is an array the
  *        length of world_npes.
  */
  MPI_Request * send_reqs;

  /**
  * @brief Requests used for various `MPI_Irecv()` calls. This is an array the
  *        length of world_npes.
  */
  MPI_Request * recv_reqs;

  /**
  * @brief Message statuses. This is an array the length of world_npes.
  */
  MPI_Status * statuses;
};






/**
* @brief A structure for MPI rank structures (communicators, etc.).
*/
typedef struct
{
  /* Tensor information */
  idx_t nmodes;
  idx_t global_nnz;
  idx_t global_dims[MAX_NMODES];
  idx_t mat_start[MAX_NMODES];
  idx_t mat_end[MAX_NMODES];
  idx_t layer_starts[MAX_NMODES];
  idx_t layer_ends[MAX_NMODES];

  /* tt indices [ownstart, ownend) are mine. These operate in global indexing,
   * so indmap is used if present. */
  idx_t nowned[MAX_NMODES];      /** number of rows owned */
  idx_t ownstart[MAX_NMODES];
  idx_t ownend[MAX_NMODES];

  idx_t * indmap[MAX_NMODES]; /** Maps local to global indices */

  /* start/end idxs for each process */
  idx_t * mat_ptrs[MAX_NMODES];
  idx_t * layer_ptrs[MAX_NMODES];

  /* same as cpd_args distribution. */
  splatt_decomp_type decomp;


  /* Send/Recv Structures
   * nlocal2nbr: This is the number of rows that I have in my tensor but do not
   *             own. I must send nlocal2nbr partial products AND receive
   *             nlocal2nbr updated rows after each iteration.
   *
   * nnbr2globs: This is the number of rows that other ranks use but I own,
   *             summed across all ranks. I receive this many partial updates
   *             and send this many updated rows after each iteration.
   *
   * local2nbr: These are rows that I compute for but do not own. These partial
   *            products must be sent to neigbors.
   *
   * nbr2local: These are neigbors' rows that I need for MTTKRP. For every row
   *            in local2nbr I need their updated factor matrices.
   *
   * nbr2globs: These are rows that neigbors have but I own. These partial
   *            products are received and I update global matrices with them.
   */
  idx_t nlocal2nbr[MAX_NMODES];
  idx_t nnbr2globs[MAX_NMODES];
  idx_t * nbr2globs_inds[MAX_NMODES];
  idx_t * local2nbr_inds[MAX_NMODES];
  idx_t * nbr2local_inds[MAX_NMODES];
  int   * local2nbr_ptr[MAX_NMODES];
  int   * nbr2globs_ptr[MAX_NMODES];
  int   * local2nbr_disp[MAX_NMODES];
  int   * nbr2globs_disp[MAX_NMODES];


  /* Communicators */
  MPI_Comm comm_3d;
  MPI_Comm layer_comm[MAX_NMODES];

  /* Rank information */
  int rank;
  int npes;
  int rank_3d;
  int dims_3d[MAX_NMODES];
  int coords_3d[MAX_NMODES];
  int layer_rank[MAX_NMODES];
  int layer_size[MAX_NMODES];

  /* Miscellaneous */
  MPI_Status status;
  MPI_Request req;
  MPI_Status * stats;
  MPI_Request * send_reqs;
  MPI_Request * recv_reqs;

  idx_t worksize;
} rank_info;



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sptensor.h"
#include "reorder.h"



/******************************************************************************
 * PUBLIC FUNCTONS
 *****************************************************************************/

#define mpi_cpd_als_iterate splatt_mpi_cpd_als_iterate
double mpi_cpd_als_iterate(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  matrix_t ** globmats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts);


#define mpi_update_rows splatt_mpi_update_rows
/**
* @brief Do an all-to-all communication of exchanging updated rows with other
*        ranks. We send globmats[mode] to the needing ranks and receive other
*        ranks' globmats entries which we store in mats[mode].
*
* @param indmap The local->global mapping of the tensor. May be NULL if the
*               mapping is identity.
* @param nbr2globs_buf Buffer at least as large as as there are rows to send
*                      (for each rank).
* @param nbr2local_buf Buffer at least as large as there are rows to receive.
* @param localmat Local factor matrix which receives updated values.
* @param globalmat Global factor matrix (owned by me) which is sent to ranks.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the factor matrices.
* @param mode The mode to exchange along.
* @param which Which communication pattern to use.
*/
void mpi_update_rows(
  idx_t const * const indmap,
  val_t * const restrict nbr2globs_buf,
  val_t * const restrict nbr2local_buf,
  matrix_t * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode,
  splatt_comm_type const which);


#define mpi_reduce_rows splatt_mpi_reduce_rows
/**
* @brief Do a reduction (sum) of all neighbor partial products which I own.
*        Updates are written to globalmat.
*
* @param local2nbr_buf A buffer at least as large as nlocal2nbr.
* @param nbr2globs_buf A buffer at least as large as nnbr2globs.
* @param localmat My local matrix containing partial products for other ranks.
* @param globalmat The global factor matrix to update.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param mode The mode to operate on.
* @param which Which communication pattern to use.
*/
void mpi_reduce_rows(
  val_t * const restrict local2nbr_buf,
  val_t * const restrict nbr2globs_buf,
  matrix_t const * const localmat,
  matrix_t * const globalmat,
  rank_info * const rinfo,
  idx_t const nfactors,
  idx_t const mode,
  splatt_comm_type const which);


#define mpi_add_my_partials splatt_mpi_add_my_partials
/**
* @brief Add my own partial products to the global matrix that I own.
*
* @param indmap The local->global mapping of the tensor. May be NULL if the
*               mapping is identity.
* @param localmat The local matrix containing my partial products.
* @param globmat The global factor matrix I am writing to.
* @param rinfo MPI rank information.
* @param nfactors The number of columns in the matrices.
* @param mode The mode I am operating on.
*/
void mpi_add_my_partials(
  idx_t const * const indmap,
  matrix_t const * const localmat,
  matrix_t * const globmat,
  rank_info const * const rinfo,
  idx_t const nfactors,
  idx_t const mode);


#define mpi_write_mats splatt_mpi_write_mats
/**
* @brief Write distributed matrices to 'basename<N>.mat'.
*
* @param mats The distributed matrices to write to disk.
* @param perm Any row permutation that we must undo.
* @param rinfo MPI rank information.
* @param basename Matrices are written to file 'basename'N.mat.
* @param nmodes The number of matrices to write.
*/
void mpi_write_mats(
  matrix_t ** mats,
  permutation_t const * const perm,
  rank_info const * const rinfo,
  char const * const basename,
  idx_t const nmodes);


#define mpi_write_part splatt_mpi_write_part
/**
* @brief Write a tensor to file <rank>.part. All local indices are converted to
*        global.
*
* @param tt The tensor to write.
* @param perm Any permutations that have been done on the tensor
*             (before compression).
* @param rinfo MPI rank information.
*/
void mpi_write_part(
  sptensor_t const * const tt,
  permutation_t const * const perm,
  rank_info const * const rinfo);


#define mpi_compute_ineed splatt_mpi_compute_ineed
/**
* @brief
*
* @param rinfo
* @param tt
*/
void mpi_compute_ineed(
  rank_info * const rinfo,
  sptensor_t const * const tt,
  idx_t const mode,
  idx_t const nfactors,
  splatt_decomp_type const distribution);


#define mpi_tt_read splatt_mpi_tt_read
/**
* @brief Each rank reads their 3D partition of a tensor.
*
* @param ifname The file containing the tensor.
* @param pfname The file containing partition information, if applicable.
* @param rinfo Rank information.
*
* @return The rank's subtensor.
*/
sptensor_t * mpi_tt_read(
  char const * const ifname,
  char const * const pfname,
  rank_info * const rinfo);


#define mpi_simple_distribute splatt_mpi_simple_distribute
/**
* @brief Do a simple distribution of the tensor stored in file 'ifname'.
*        Load balance is based on nonzero count. No communication or other
*        heuristics used. Tensor nonzeros are distributed among communicator
*        'comm'.
*
* @param ifname The file to read from.
* @param comm The communicator to distribute among
*
* @return My own sub-tensor.
*/
sptensor_t * mpi_simple_distribute(
  char const * const ifname,
  MPI_Comm comm);



#define mpi_rearrange_by_part splatt_mpi_rearrange_by_part
/**
* @brief Rearrange nonzeros based on an nonzero partitioning. This allocates
*        and returns a new sptensor_t.
*
* @param ttbuf The nonzeros to rearrange.
* @param parts The partitioning of length ttbuf->nnz.
* @param rinfo The communicator to rearrange along.
*
* @return A new rearranged tensor.
*/
sptensor_t * mpi_rearrange_by_part(
  sptensor_t const * const ttbuf,
  int const * const parts,
  MPI_Comm comm);


#define mpi_determine_med_owner splatt_mpi_determine_med_owner
/**
* @brief Map a nonzero to an MPI rank based on the medium-grained layer
*        boundaries.
*
* @param ttbuf The sparse tensor.
* @param n The index of the nonzero.
* @param rinfo MPI rank information (uses dims_3d and layer_ptrs).
*
* @return The MPI rank that owns ttbuf[n].
*/
int mpi_determine_med_owner(
  sptensor_t * const ttbuf,
  idx_t const n,
  rank_info * const rinfo);


#define mpi_filter_tt_1d splatt_mpi_filter_tt_1d
/**
* @brief Run nonzeros from tt through filter to 'ftt'. This is 1D filtering,
*        so we accept any nonzeros whose ind[mode] are within [start, end).
*
* @param mode The mode to filter along.
* @param tt The original tensor.
* @param ftt The tensor to filter into (pre-allocated).
* @param start The first index to accept (inclusive).
* @param end The last index to accept (exclusive).
*/
void mpi_filter_tt_1d(
  idx_t const mode,
  sptensor_t const * const tt,
  sptensor_t * const ftt,
  idx_t start,
  idx_t end);


#define mpi_distribute_mats splatt_mpi_distribute_mats
/**
* @brief Compute a distribution of factor matrices that minimizes communication
*        volume.
*
* @param rinfo MPI structure containing rank and communicator information.
* @param tt A partition of the tensor. NOTE: indices will be reordered after
*           distribution to ensure contiguous matrix partitions.
* @param distribution The dimension of the distribution to perform.
*
* @return The permutation that was applied to tt.
*/
permutation_t *  mpi_distribute_mats(
  rank_info * const rinfo,
  sptensor_t * const tt,
  splatt_decomp_type const distribution);



#define mpi_mat_rand splatt_mpi_mat_rand
/**
* @brief Allocate, initialize, and distribute a random matrix among MPI ranks.
*        This function respects permutation info (such as from
*        `mpi_distribute_mats()`) so that the same seed will result in the same
*        problem solution, no matter the number of ranks.
*
* @param mode Which mode we are allocating for.
* @param nfactors The number of columns in the matrix.
* @param perm Permutation info. perm->iperms[mode] is used.
* @param rinfo MPI rank information, rinfo->mat_start[mode] is used.
*
* @return The portion of the random matrix that I own.
*/
matrix_t * mpi_mat_rand(
  idx_t const mode,
  idx_t const nfactors,
  permutation_t const * const perm,
  rank_info * const rinfo);



#define mpi_find_owned splatt_mpi_find_owned
/**
* @brief Setup 'owned' structures which mark the location of owned rows in
*        my local tensor.
*
* @param tt My subtensor.
* @param rinfo MPI rank information.
*/
void mpi_find_owned(
  sptensor_t const * const tt,
  idx_t const mode,
  rank_info * const rinfo);


#define mpi_cpy_indmap splatt_mpi_cpy_indmap
/**
* @brief Copy the indmap information from a sparse tensor into rank_info. If
*        tt->indmap[mode] is NULL, this sets rinfo->indmap[mode] to NULL.
*
* @param tt The sparse tensor to copy from.
* @param rinfo The rank_info structure to copy to.
* @param mode Which mode to copy.
*/
void mpi_cpy_indmap(
  sptensor_t const * const tt,
  rank_info * const rinfo,
  idx_t const mode);


#define mpi_setup_comms splatt_mpi_setup_comms
/**
* @brief Fill rinfo with process' MPI rank information. Includes rank, 3D
*        communicator, etc.
*
* @param rinfo The rank data structure.
*/
void mpi_setup_comms(
  rank_info * const rinfo);


#define rank_free splatt_rank_free
/**
* @brief Free structures allocated inside rank_info.
*
* @param rinfo The rank structure to free.
* @param nmodes The number of modes that have been allocated.
*/
void rank_free(
  rank_info rinfo,
  idx_t const nmodes);


#define mpi_time_stats splatt_mpi_time_stats
/**
* @brief Update timers[] with max values on the master rank instead of only
*        local times.
*
* @param rinfo Struct containing rank information.
*/
void mpi_time_stats(
  rank_info const * const rinfo);


#define mpi_send_recv_stats splatt_mpi_send_recv_stats
/**
* @brief Print send/recieve information to STDOUT.
*
* @param rinfo MPI rank information. Assumes mpi_distribute_mats() has already
*              been called.
* @param tt The distributed tensor.
*/
void mpi_send_recv_stats(
  rank_info const * const rinfo,
  sptensor_t const * const tt);

#endif /* SPLATT_USE_MPI */
#endif
