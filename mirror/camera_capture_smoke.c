#define _POSIX_C_SOURCE 200809L

#include "camera_capture.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAMERA_CAPTURE_OK 0
#define CAMERA_CAPTURE_ERR_INVALID 1
#define CAMERA_CAPTURE_ERR_ALLOC 2
#define CAMERA_CAPTURE_ERR_THREAD 3
#define CAMERA_CAPTURE_ERR_UNSUPPORTED 4
#define CAMERA_CAPTURE_ERR_EMPTY 5

static void write_le16(FILE *file, uint16_t value) {
    fputc((int) (value & 0xffU), file);
    fputc((int) ((value >> 8U) & 0xffU), file);
}

static void write_le32(FILE *file, uint32_t value) {
    fputc((int) (value & 0xffU), file);
    fputc((int) ((value >> 8U) & 0xffU), file);
    fputc((int) ((value >> 16U) & 0xffU), file);
    fputc((int) ((value >> 24U) & 0xffU), file);
}

static const char *describe_error(int error_code) {
    switch (error_code) {
        case CAMERA_CAPTURE_OK:
            return "ok";
        case CAMERA_CAPTURE_ERR_INVALID:
            return "invalid arguments or uninitialized capture";
        case CAMERA_CAPTURE_ERR_ALLOC:
            return "allocation failed";
        case CAMERA_CAPTURE_ERR_THREAD:
            return "thread startup failed";
        case CAMERA_CAPTURE_ERR_UNSUPPORTED:
            return "unsupported camera format or backend";
        case CAMERA_CAPTURE_ERR_EMPTY:
            return "no frame available yet";
        default:
            return strerror(error_code);
    }
}

static void sleep_ms(long milliseconds) {
    struct timespec delay;
    delay.tv_sec = milliseconds / 1000L;
    delay.tv_nsec = (milliseconds % 1000L) * 1000000L;
    nanosleep(&delay, NULL);
}

static int save_frame_as_bmp(const video_frame_t *frame, const char *output_path) {
    const size_t row_stride = (size_t) frame->width * 3U;
    const size_t row_padding = (4U - (row_stride % 4U)) % 4U;
    const size_t bmp_row_size = row_stride + row_padding;
    const uint32_t pixel_data_size = (uint32_t) (bmp_row_size * (size_t) frame->height);
    const uint32_t file_size = 54U + pixel_data_size;
    uint8_t *row_buffer;
    FILE *file;
    uint32_t row;

    if (frame == NULL || output_path == NULL || frame->frame_bytes == NULL) {
        return EINVAL;
    }
    if (
        frame->pixel_format != PIXEL_FORMAT_BGR24
        && frame->pixel_format != PIXEL_FORMAT_RGB24
    ) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    row_buffer = malloc(bmp_row_size);
    if (row_buffer == NULL) {
        return ENOMEM;
    }

    file = fopen(output_path, "wb");
    if (file == NULL) {
        free(row_buffer);
        return errno;
    }

    fputc('B', file);
    fputc('M', file);
    write_le32(file, file_size);
    write_le16(file, 0U);
    write_le16(file, 0U);
    write_le32(file, 54U);
    write_le32(file, 40U);
    write_le32(file, frame->width);
    write_le32(file, frame->height);
    write_le16(file, 1U);
    write_le16(file, 24U);
    write_le32(file, 0U);
    write_le32(file, pixel_data_size);
    write_le32(file, 2835U);
    write_le32(file, 2835U);
    write_le32(file, 0U);
    write_le32(file, 0U);

    memset(row_buffer + row_stride, 0, row_padding);
    for (row = 0U; row < frame->height; ++row) {
        const uint32_t source_row = frame->height - 1U - row;
        const uint8_t *source = frame->frame_bytes + ((size_t) source_row * row_stride);
        size_t column;

        if (frame->pixel_format == PIXEL_FORMAT_BGR24) {
            memcpy(row_buffer, source, row_stride);
        } else {
            for (column = 0U; column < (size_t) frame->width; ++column) {
                const size_t source_offset = column * 3U;
                row_buffer[source_offset + 0U] = source[source_offset + 2U];
                row_buffer[source_offset + 1U] = source[source_offset + 1U];
                row_buffer[source_offset + 2U] = source[source_offset + 0U];
            }
        }

        if (fwrite(row_buffer, 1U, bmp_row_size, file) != bmp_row_size) {
            const int error_code = ferror(file) ? EIO : errno;
            fclose(file);
            free(row_buffer);
            return error_code == 0 ? EIO : error_code;
        }
    }

    fclose(file);
    free(row_buffer);
    return CAMERA_CAPTURE_OK;
}

int main(int argc, char **argv) {
    camera_capture_t capture;
    camera_capture_config_t config;
    video_frame_t frame;
    const uint32_t width = argc > 1 ? (uint32_t) strtoul(argv[1], NULL, 10) : 640U;
    const uint32_t height = argc > 2 ? (uint32_t) strtoul(argv[2], NULL, 10) : 480U;
    const int timeout_ms = argc > 3 ? atoi(argv[3]) : 5000;
    const char *output_path = argc > 4 ? argv[4] : "captured_frame.bmp";
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
        fprintf(
            stderr,
            "camera_capture_init failed: %d (%s)\n",
            result,
            describe_error(result)
        );
        return 1;
    }

    result = camera_capture_start(&capture);
    if (result != CAMERA_CAPTURE_OK) {
        fprintf(
            stderr,
            "camera_capture_start failed: %d (%s)\n",
            result,
            describe_error(result)
        );
        if (result == ENOENT) {
            fprintf(
                stderr,
                "Hint: /dev/video0 is not available in this environment.\n"
            );
        }
        camera_capture_destroy(&capture);
        return 1;
    }

    while (elapsed_ms < timeout_ms) {
        result = camera_capture_get_latest_frame(&capture, &frame);
        if (result == CAMERA_CAPTURE_OK) {
            capture_status_t status = camera_capture_get_status(&capture);
            const int save_result = save_frame_as_bmp(&frame, output_path);
            printf(
                "Captured frame: %ux%u %zu bytes timestamp=%llu dropped=%llu failures=%llu\n",
                frame.width,
                frame.height,
                frame.frame_size_bytes,
                (unsigned long long) frame.timestamp_ns,
                (unsigned long long) status.dropped_count,
                (unsigned long long) status.failure_count
            );
            if (save_result == CAMERA_CAPTURE_OK) {
                printf("Saved frame to %s\n", output_path);
            } else {
                fprintf(
                    stderr,
                    "Captured frame but failed to save BMP: %d (%s)\n",
                    save_result,
                    describe_error(save_result)
                );
            }
            camera_capture_release_frame(&frame);
            camera_capture_destroy(&capture);
            return 0;
        }
        if (result != CAMERA_CAPTURE_ERR_EMPTY) {
            fprintf(
                stderr,
                "camera_capture_get_latest_frame failed: %d (%s)\n",
                result,
                describe_error(result)
            );
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
            "Timed out waiting for frame. healthy=%d opened=%d running=%d last_error=%d (%s) failures=%llu\n",
            status.healthy,
            status.opened,
            status.running,
            status.last_error_code,
            describe_error(status.last_error_code),
            (unsigned long long) status.failure_count
        );
    }

    camera_capture_destroy(&capture);
    return 1;
}
