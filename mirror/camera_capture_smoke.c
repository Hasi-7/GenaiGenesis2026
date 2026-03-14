#define _POSIX_C_SOURCE 200809L

#include "camera_capture.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <unistd.h>
#endif

#define CAMERA_CAPTURE_OK 0
#define CAMERA_CAPTURE_ERR_INVALID 1
#define CAMERA_CAPTURE_ERR_ALLOC 2
#define CAMERA_CAPTURE_ERR_THREAD 3
#define CAMERA_CAPTURE_ERR_UNSUPPORTED 4
#define CAMERA_CAPTURE_ERR_EMPTY 5
#define DEFAULT_STREAM_PORT 9000U
#define DEFAULT_STREAM_WIDTH 640U
#define DEFAULT_STREAM_HEIGHT 480U
#define DEFAULT_STREAM_FPS 15U

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

static void print_usage(const char *program_name) {
    fprintf(
        stderr,
        "Usage:\n"
        "  %s [width] [height] [timeout_ms] [output_path]\n"
        "  %s stream <server_ip> [port] [width] [height] [fps]\n",
        program_name,
        program_name
    );
}

static int run_stream_mode(int argc, char **argv) {
#if defined(__linux__)
    const char *server_host;
    uint32_t port;
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    const char *slash;
    size_t dir_length;
    size_t path_length;
    char *streamer_path;
    char port_arg[16];
    char width_arg[16];
    char height_arg[16];
    char fps_arg[16];
    char *exec_args[] = {
        NULL,
        NULL,
        port_arg,
        width_arg,
        height_arg,
        fps_arg,
        NULL,
    };

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    server_host = argv[2];
    port = argc > 3 ? (uint32_t) strtoul(argv[3], NULL, 10) : DEFAULT_STREAM_PORT;
    width = argc > 4 ? (uint32_t) strtoul(argv[4], NULL, 10) : DEFAULT_STREAM_WIDTH;
    height = argc > 5 ? (uint32_t) strtoul(argv[5], NULL, 10) : DEFAULT_STREAM_HEIGHT;
    fps = argc > 6 ? (uint32_t) strtoul(argv[6], NULL, 10) : DEFAULT_STREAM_FPS;

    slash = strrchr(argv[0], '/');
    dir_length = slash == NULL ? 0U : (size_t) (slash - argv[0] + 1);
    path_length = dir_length + strlen("mirror_frame_streamer") + 1U;
    streamer_path = malloc(path_length);
    if (streamer_path == NULL) {
        fprintf(stderr, "Failed to allocate streamer path buffer\n");
        return 1;
    }

    if (dir_length > 0U) {
        memcpy(streamer_path, argv[0], dir_length);
    }
    memcpy(
        streamer_path + dir_length,
        "mirror_frame_streamer",
        strlen("mirror_frame_streamer") + 1U
    );

    snprintf(port_arg, sizeof(port_arg), "%u", port);
    snprintf(width_arg, sizeof(width_arg), "%u", width);
    snprintf(height_arg, sizeof(height_arg), "%u", height);
    snprintf(fps_arg, sizeof(fps_arg), "%u", fps);

    exec_args[0] = streamer_path;
    exec_args[1] = (char *) server_host;

    printf(
        "Launching continuous stream test to %s:%u at %ux%u %u fps\n",
        server_host,
        port,
        width,
        height,
        fps
    );
    fflush(stdout);

    execv(streamer_path, exec_args);
    fprintf(
        stderr,
        "Failed to exec %s: %s\n",
        streamer_path,
        strerror(errno)
    );
    free(streamer_path);
    return 1;
#else
    (void) argc;
    (void) argv;
    fprintf(stderr, "Streaming mode is only supported on Linux\n");
    return 1;
#endif
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

    if (argc > 1 && strcmp(argv[1], "stream") == 0) {
        return run_stream_mode(argc, argv);
    }

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
