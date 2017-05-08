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


/**
* @brief Allocate a structure to represent a coordinate tensor. This data
*        structure must be freed by `splatt_free_coord()`.
*
* @return The allocated memory.
*/
splatt_coord * splatt_alloc_coord();


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


#ifdef __cplusplus
}
#endif

#endif

