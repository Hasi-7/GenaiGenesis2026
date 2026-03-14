#ifndef MIRROR_CAPTURE_TYPES_H
#define MIRROR_CAPTURE_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CAPTURE_CLIENT_ID_MAX 64

typedef enum client_type_e {
    CLIENT_TYPE_DESKTOP = 0,
    CLIENT_TYPE_MIRROR = 1
} client_type_t;

typedef enum pixel_format_e {
    PIXEL_FORMAT_UNKNOWN = 0,
    PIXEL_FORMAT_YUV420 = 1,
    PIXEL_FORMAT_RGB24 = 2,
    PIXEL_FORMAT_BGR24 = 3
} pixel_format_t;

typedef struct video_frame_s {
    char client_id[CAPTURE_CLIENT_ID_MAX];
    client_type_t client_type;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    pixel_format_t pixel_format;
    size_t frame_size_bytes;
    uint8_t *frame_bytes;
} video_frame_t;

typedef struct capture_status_s {
    bool healthy;
    bool opened;
    bool running;
    uint64_t last_timestamp_ns;
    uint64_t dropped_count;
    uint64_t failure_count;
    int last_error_code;
} capture_status_t;

#endif
