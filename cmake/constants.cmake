
if (DEFINED MAX_MODES)
  add_definitions(-DSPLATT_MAX_NMODES=${MAX_MODES})
  message("Setting SPLATT_MAX_NMODES=${MAX_MODES}")
endif()

if (INTEL_OPT)
  add_definitions(-DSPLATT_USE_MKL=1)
endif()
