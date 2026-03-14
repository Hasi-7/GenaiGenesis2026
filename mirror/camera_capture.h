#ifndef MIRROR_CAMERA_CAPTURE_H
#define MIRROR_CAMERA_CAPTURE_H

#include "capture_types.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct camera_capture_config_s {
    char client_id[CAPTURE_CLIENT_ID_MAX];
    uint32_t width;
    uint32_t height;
    pixel_format_t pixel_format;
    size_t buffer_capacity;
} camera_capture_config_t;

typedef struct camera_capture_s {
    camera_capture_config_t config;
    video_frame_t *frames;
    size_t frame_count;
    size_t write_index;
    pthread_mutex_t lock;
    pthread_t thread;
    atomic_bool stop_requested;
    atomic_bool initialized;
    atomic_bool thread_started;
    capture_status_t status;
    int backend_fd;
    uint32_t backend_fourcc;
    size_t backend_frame_size_bytes;
    uint8_t *backend_frame_buffer;
} camera_capture_t;

int camera_capture_init(camera_capture_t *capture, const camera_capture_config_t *config);
int camera_capture_start(camera_capture_t *capture);
int camera_capture_stop(camera_capture_t *capture);
int camera_capture_get_latest_frame(camera_capture_t *capture, video_frame_t *out_frame);
capture_status_t camera_capture_get_status(camera_capture_t *capture);
void camera_capture_release_frame(video_frame_t *frame);
void camera_capture_destroy(camera_capture_t *capture);

#ifdef __cplusplus
}
#endif

#endif
