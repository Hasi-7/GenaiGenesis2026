#include "camera_capture.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAMERA_CAPTURE_OK 0
#define CAMERA_CAPTURE_ERR_EMPTY 5

static void sleep_ms(long milliseconds) {
    struct timespec delay;
    delay.tv_sec = milliseconds / 1000L;
    delay.tv_nsec = (milliseconds % 1000L) * 1000000L;
    nanosleep(&delay, NULL);
}

int main(int argc, char **argv) {
    camera_capture_t capture;
    camera_capture_config_t config;
    video_frame_t frame;
    const uint32_t width = argc > 1 ? (uint32_t) strtoul(argv[1], NULL, 10) : 640U;
    const uint32_t height = argc > 2 ? (uint32_t) strtoul(argv[2], NULL, 10) : 480U;
    const int timeout_ms = argc > 3 ? atoi(argv[3]) : 5000;
    int elapsed_ms = 0;
    int result;

    memset(&capture, 0, sizeof(capture));
    memset(&config, 0, sizeof(config));
    memset(&frame, 0, sizeof(frame));

    strncpy(config.client_id, "mirror-pi", CAPTURE_CLIENT_ID_MAX - 1U);
    config.width = width;
    config.height = height;
    config.pixel_format = PIXEL_FORMAT_BGR24;
    config.buffer_capacity = 2U;

    result = camera_capture_init(&capture, &config);
    if (result != CAMERA_CAPTURE_OK) {
        fprintf(stderr, "camera_capture_init failed: %d\n", result);
        return 1;
    }

    result = camera_capture_start(&capture);
    if (result != CAMERA_CAPTURE_OK) {
        fprintf(stderr, "camera_capture_start failed: %d\n", result);
        camera_capture_destroy(&capture);
        return 1;
    }

    while (elapsed_ms < timeout_ms) {
        result = camera_capture_get_latest_frame(&capture, &frame);
        if (result == CAMERA_CAPTURE_OK) {
            capture_status_t status = camera_capture_get_status(&capture);
            printf(
                "Captured frame: %ux%u %zu bytes timestamp=%llu dropped=%llu failures=%llu\n",
                frame.width,
                frame.height,
                frame.frame_size_bytes,
                (unsigned long long) frame.timestamp_ns,
                (unsigned long long) status.dropped_count,
                (unsigned long long) status.failure_count
            );
            camera_capture_release_frame(&frame);
            camera_capture_destroy(&capture);
            return 0;
        }
        if (result != CAMERA_CAPTURE_ERR_EMPTY) {
            fprintf(stderr, "camera_capture_get_latest_frame failed: %d\n", result);
            camera_capture_destroy(&capture);
            return 1;
        }
        sleep_ms(100L);
        elapsed_ms += 100;
    }

    {
        capture_status_t status = camera_capture_get_status(&capture);
        fprintf(
            stderr,
            "Timed out waiting for frame. healthy=%d opened=%d running=%d last_error=%d failures=%llu\n",
            status.healthy,
            status.opened,
            status.running,
            status.last_error_code,
            (unsigned long long) status.failure_count
        );
    }

    camera_capture_destroy(&capture);
    return 1;
}
