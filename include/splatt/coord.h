/**
* @file include/splatt/coord.h
* @brief Functions for sparse tensors stored in coordinate format.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2017-05-08
*/


#ifndef SPLATT_INCLUDE_COORD_H
#define SPLATT_INCLUDE_COORD_H


/******************************************************************************
 * TYPES
 *****************************************************************************/


/**
* @brief This is a coordinate format data structure for sparse tensors. Most
*        IO and pre-processing routines use this data structure.
*/
typedef struct _splatt_coord splatt_coord;



/*
 * COORDINATE API
 */


#ifdef __cplusplus
extern "C" {
#endif

/*
 * Memory management.
 */


/**
* @brief Allocate a structure to represent a coordinate tensor. This data
*        structure must be freed by `splatt_free_coord()`.
*
*        NOTE: this function will not initialize any of the non-zero values or
*        indices.
*
* @param num_modes The number of modes in the tensor.
* @param nnz The number of non-zeros in the tensor.
*
* @return The allocated tensor.
*/
splatt_coord * splatt_alloc_coord(
    splatt_idx_t const num_modes,
    splatt_idx_t const nnz);


/**
* @brief Free any memory allocated for a coordinate tensor.
*
* @param coord The tensor to free.
*/
void splatt_free_coord(
    splatt_coord * coord);


/**
* @brief Load a coordinate sparse tensor from a file and store in `coord`.
*
*        NOTE: `coord` must have already been allocated via
*        `splatt_alloc_coord()`.
*
* @param fname The file to load.
* @param[out] coord The tensor to populate. The structure must have already
*                   been allocated.
*
* @return SPLATT error code. SPLATT_SUCCESS on success.
*/
splatt_error_type splatt_coord_load(
    char const * const fname,
    splatt_coord * const coord);

/*
 * Accessors.
 */


/**
* @brief Get the number of non-zeros in a tensor.
*
* @param coord The sparse tensor.
*
* @return The number of non-zeros.
*/
splatt_idx_t splatt_coord_get_nnz(
    splatt_coord const * const coord);

/**
* @brief Get the number of modes in a tensor.
*
* @param coord The sparse tensor.
*
* @return The number of modes.
*/
splatt_idx_t splatt_coord_get_modes(
    splatt_coord const * const coord);

/**
* @brief Get an array of indices for one of the tensor modes.
*
* @param coord The sparse tensor to query.
* @param mode The mode of interest.
*
* @return An array of indices for mode 'mode'.
*/
splatt_idx_t * splatt_coord_get_inds(
    splatt_coord const * const coord,
    splatt_idx_t const mode);

/**
* @brief Get the array of values for the tensor non-zeros.
*
* @param coord The sparse tensor.
*
* @return The sparse tensor's non-zeros.
*/
splatt_val_t * splatt_coord_get_vals(
    splatt_coord const * const coord);


#ifdef __cplusplus
}
#endif

#endif

