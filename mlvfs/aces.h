#ifndef ACES_H
#define ACES_H

#include "mlvfs.h"
#ifdef __cplusplus
extern "C" {
#endif

size_t exr_get_size(struct frame_headers * frame_headers, const char* name);
void process_aces(struct frame_headers * frame_headers, struct image_buffer* image_buffer, const char* name);

#ifdef __cplusplus
}
#endif

#endif