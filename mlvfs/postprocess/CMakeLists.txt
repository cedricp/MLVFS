FILE(GLOB SOURCES patternnoise.c stripes.c cs.c amaze_demosaic_RT.c hdr.c histogram.c)
FILE(GLOB HEADERS *.h)

ADD_LIBRARY(postprocess STATIC ${SOURCES} ${HEADERS})
SET_PROPERTY(TARGET postprocess PROPERTY C_STANDARD 99)
TARGET_INCLUDE_DIRECTORIES(postprocess PUBLIC . .. ../dng)