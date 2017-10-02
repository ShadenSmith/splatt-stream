/**
* @file stream.h
* @brief Internal API functions for streaming CPD factorization.
* @author Shaden Smith <shaden@cs.umn.edu>
* @version 2.0.0
* @date 2017-10-02
*/


#ifndef SPLATT_CPD_STREAM_H
#define SPLATT_CPD_STREAM_H



/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "../base.h"


/*
 * XXX TODO what to return etc. This is very much a prototype!
 */
splatt_error_type splatt_cpd_stream(
    char const * const filename,
    splatt_idx_t rank,
    splatt_idx_t const stream_mode,
    double const forget,
    splatt_cpd_opts const * const cpd_options,
    splatt_global_opts const * const global_options);

#endif
