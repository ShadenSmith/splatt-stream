/**
* @file comm_info.h
* @brief Functions for allocating/initializing splatt_comm_info.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2016-07-12
*/

#ifndef SPLATT_MPI_COMM_INFO_H
#define SPLATT_MPI_COMM_INFO_H


#include "../splatt_mpi.h"


/******************************************************************************
 * TYPES
 *****************************************************************************/


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../base.h"
#include "../sptensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/


#define comm_fill_global splatt_comm_fill_global
/**
* @brief Complete the global fields of 'mpi' that are not specific to a
*        decomposition.
*
* @param tt The tensor we are distributed.
* @param mpi The MPI structure to fill.
*/
void comm_fill_global(
    sptensor_t const * const tt,
    splatt_comm_info * mpi);

#endif
