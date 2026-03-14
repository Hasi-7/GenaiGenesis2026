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
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CAMERA_CAPTURE_OK 0
#define CAMERA_CAPTURE_ERR_INVALID 1
#define CAMERA_CAPTURE_ERR_ALLOC 2
#define CAMERA_CAPTURE_ERR_THREAD 3
#define CAMERA_CAPTURE_ERR_UNSUPPORTED 4
#define CAMERA_CAPTURE_ERR_EMPTY 5

#define CAMERA_BACKEND_NONE 0
#define CAMERA_BACKEND_PI_CLI 1

static uint32_t read_le32(const uint8_t *buffer) {
    return (uint32_t) buffer[0]
        | ((uint32_t) buffer[1] << 8U)
        | ((uint32_t) buffer[2] << 16U)
        | ((uint32_t) buffer[3] << 24U);
}

static int32_t read_le32_signed(const uint8_t *buffer) {
    return (int32_t) read_le32(buffer);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t) ts.tv_sec * 1000000000ULL) + (uint64_t) ts.tv_nsec;
}

static void sleep_us(long microseconds) {
    struct timespec delay;

    delay.tv_sec = microseconds / 1000000L;
    delay.tv_nsec = (microseconds % 1000000L) * 1000L;
    nanosleep(&delay, NULL);
}

static size_t bytes_per_pixel(pixel_format_t pixel_format) {
    switch (pixel_format) {
        case PIXEL_FORMAT_RGB24:
        case PIXEL_FORMAT_BGR24:
            return 3U;
        default:
            return 0U;
    }
}

static size_t frame_byte_capacity(const camera_capture_t *capture) {
    return (size_t) capture->config.width
        * (size_t) capture->config.height
        * bytes_per_pixel(capture->config.pixel_format);
}

static void clear_frame(video_frame_t *frame) {
    if (frame == NULL) {
        return;
    }
    frame->timestamp_ns = 0;
    frame->frame_size_bytes = 0U;
}

static int allocate_frames(camera_capture_t *capture) {
    size_t i;
    size_t bytes_per_frame;

    bytes_per_frame = frame_byte_capacity(capture);
    if (bytes_per_frame == 0U) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    capture->frames = calloc(capture->config.buffer_capacity, sizeof(video_frame_t));
    if (capture->frames == NULL) {
        return CAMERA_CAPTURE_ERR_ALLOC;
    }

    for (i = 0U; i < capture->config.buffer_capacity; ++i) {
        video_frame_t *frame = &capture->frames[i];

        strncpy(frame->client_id, capture->config.client_id, CAPTURE_CLIENT_ID_MAX - 1U);
        frame->client_type = CLIENT_TYPE_MIRROR;
        frame->width = capture->config.width;
        frame->height = capture->config.height;
        frame->pixel_format = capture->config.pixel_format;
        frame->frame_bytes = malloc(bytes_per_frame);
        if (frame->frame_bytes == NULL) {
            return CAMERA_CAPTURE_ERR_ALLOC;
        }
        clear_frame(frame);
    }

    return CAMERA_CAPTURE_OK;
}

static void free_frames(camera_capture_t *capture) {
    size_t i;

    if (capture->frames == NULL) {
        return;
    }

    for (i = 0U; i < capture->config.buffer_capacity; ++i) {
        free(capture->frames[i].frame_bytes);
        capture->frames[i].frame_bytes = NULL;
    }

    free(capture->frames);
    capture->frames = NULL;
    capture->frame_count = 0U;
    capture->write_index = 0U;
}

static void mark_failure(camera_capture_t *capture, int error_code) {
    pthread_mutex_lock(&capture->lock);
    capture->status.healthy = false;
    capture->status.failure_count += 1U;
    capture->status.last_error_code = error_code;
    pthread_mutex_unlock(&capture->lock);
}

#if defined(__linux__)
static bool find_command_on_path(
    const char *command,
    char *resolved_path,
    size_t resolved_path_size
) {
    const char *path_env;
    const char *segment_start;

    if (command == NULL || resolved_path == NULL || resolved_path_size == 0U) {
        return false;
    }

    path_env = getenv("PATH");
    if (path_env == NULL) {
        return false;
    }

    segment_start = path_env;
    while (*segment_start != '\0') {
        const char *segment_end = strchr(segment_start, ':');
        const size_t segment_length = segment_end == NULL
            ? strlen(segment_start)
            : (size_t) (segment_end - segment_start);

        if (segment_length > 0U) {
            const int written = snprintf(
                resolved_path,
                resolved_path_size,
                "%.*s/%s",
                (int) segment_length,
                segment_start,
                command
            );
            if (
                written > 0
                && (size_t) written < resolved_path_size
                && access(resolved_path, X_OK) == 0
            ) {
                return true;
            }
        }

        if (segment_end == NULL) {
            break;
        }
        segment_start = segment_end + 1;
    }

    return false;
}

static int parse_bmp_frame(
    const uint8_t *bmp_bytes,
    size_t bmp_size,
    uint32_t expected_width,
    uint32_t expected_height,
    pixel_format_t pixel_format,
    uint8_t *destination,
    size_t destination_capacity
) {
    uint32_t pixel_offset;
    uint32_t dib_header_size;
    int32_t bmp_width;
    int32_t bmp_height;
    uint16_t bits_per_pixel;
    uint32_t compression;
    uint32_t absolute_height;
    size_t bytes_per_pixel_value;
    size_t row_size;
    uint32_t row_index;
    bool top_down;

    if (bmp_bytes == NULL || destination == NULL || bmp_size < 54U) {
        return EINVAL;
    }
    if (bmp_bytes[0] != 'B' || bmp_bytes[1] != 'M') {
        return EINVAL;
    }

    pixel_offset = read_le32(bmp_bytes + 10U);
    dib_header_size = read_le32(bmp_bytes + 14U);
    bmp_width = read_le32_signed(bmp_bytes + 18U);
    bmp_height = read_le32_signed(bmp_bytes + 22U);
    bits_per_pixel = (uint16_t) (bmp_bytes[28] | ((uint16_t) bmp_bytes[29] << 8U));
    compression = read_le32(bmp_bytes + 30U);

    if (dib_header_size < 40U) {
        return EINVAL;
    }
    if (bits_per_pixel != 24U && bits_per_pixel != 32U) {
        return EINVAL;
    }
    if (compression != 0U && compression != 3U) {
        return EINVAL;
    }
    if (bmp_width <= 0 || bmp_height == 0) {
        return EINVAL;
    }

    absolute_height = bmp_height < 0 ? (uint32_t) (-bmp_height) : (uint32_t) bmp_height;
    if ((uint32_t) bmp_width != expected_width || absolute_height != expected_height) {
        return EINVAL;
    }

    bytes_per_pixel_value = (size_t) bits_per_pixel / 8U;
    row_size = ((((size_t) expected_width * (size_t) bits_per_pixel) + 31U) / 32U) * 4U;
    if ((size_t) pixel_offset + row_size * (size_t) expected_height > bmp_size) {
        return EINVAL;
    }
    if (destination_capacity < (size_t) expected_width * (size_t) expected_height * 3U) {
        return EINVAL;
    }

    top_down = bmp_height < 0;
    for (row_index = 0U; row_index < expected_height; ++row_index) {
        const uint32_t source_row = top_down ? row_index : (expected_height - 1U - row_index);
        const uint8_t *source = bmp_bytes + pixel_offset + row_size * (size_t) source_row;
        uint8_t *target = destination + ((size_t) row_index * (size_t) expected_width * 3U);
        size_t pixel_index;

        for (pixel_index = 0U; pixel_index < (size_t) expected_width; ++pixel_index) {
            const size_t source_offset = pixel_index * bytes_per_pixel_value;
            const size_t target_offset = pixel_index * 3U;

            if (pixel_format == PIXEL_FORMAT_BGR24) {
                target[target_offset + 0U] = source[source_offset + 0U];
                target[target_offset + 1U] = source[source_offset + 1U];
                target[target_offset + 2U] = source[source_offset + 2U];
            } else {
                target[target_offset + 0U] = source[source_offset + 2U];
                target[target_offset + 1U] = source[source_offset + 1U];
                target[target_offset + 2U] = source[source_offset + 0U];
            }
        }
    }

    return CAMERA_CAPTURE_OK;
}

static int capture_with_pi_command(
    camera_capture_t *capture,
    uint8_t *destination,
    size_t capacity
) {
    char temp_template[] = "/tmp/mirror-frame-XXXXXX";
    char temp_path[sizeof(temp_template) + 4U];
    int temp_fd;
    pid_t child_pid;
    int wait_status;
    FILE *bmp_file;
    uint8_t *bmp_bytes;
    long bmp_size_long;
    size_t bmp_size;
    int parse_result;

    temp_fd = mkstemp(temp_template);
    if (temp_fd < 0) {
        return errno;
    }
    close(temp_fd);

    if (snprintf(temp_path, sizeof(temp_path), "%s.bmp", temp_template) >= (int) sizeof(temp_path)) {
        unlink(temp_template);
        return ENAMETOOLONG;
    }
    if (rename(temp_template, temp_path) != 0) {
        const int error_code = errno;
        unlink(temp_template);
        return error_code;
    }

    child_pid = fork();
    if (child_pid < 0) {
        unlink(temp_path);
        return errno;
    }
    if (child_pid == 0) {
        char width_arg[16];
        char height_arg[16];
        char timeout_arg[16];
        char *const argv[] = {
            capture->backend_command,
            "--nopreview",
            "--immediate",
            "--timeout",
            timeout_arg,
            "--width",
            width_arg,
            "--height",
            height_arg,
            "--encoding",
            "bmp",
            "--output",
            temp_path,
            NULL,
        };

        snprintf(width_arg, sizeof(width_arg), "%u", capture->config.width);
        snprintf(height_arg, sizeof(height_arg), "%u", capture->config.height);
        snprintf(timeout_arg, sizeof(timeout_arg), "%d", 500);
        execv(capture->backend_command, argv);
        _exit(127);
    }

    if (waitpid(child_pid, &wait_status, 0) < 0) {
        unlink(temp_path);
        return errno;
    }
    if (!WIFEXITED(wait_status) || WEXITSTATUS(wait_status) != 0) {
        unlink(temp_path);
        return EIO;
    }

    bmp_file = fopen(temp_path, "rb");
    if (bmp_file == NULL) {
        unlink(temp_path);
        return errno;
    }
    if (fseek(bmp_file, 0L, SEEK_END) != 0) {
        fclose(bmp_file);
        unlink(temp_path);
        return errno;
    }
    bmp_size_long = ftell(bmp_file);
    if (bmp_size_long < 0L) {
        fclose(bmp_file);
        unlink(temp_path);
        return errno;
    }
    if (fseek(bmp_file, 0L, SEEK_SET) != 0) {
        fclose(bmp_file);
        unlink(temp_path);
        return errno;
    }

    bmp_size = (size_t) bmp_size_long;
    bmp_bytes = malloc(bmp_size);
    if (bmp_bytes == NULL) {
        fclose(bmp_file);
        unlink(temp_path);
        return ENOMEM;
    }
    if (fread(bmp_bytes, 1U, bmp_size, bmp_file) != bmp_size) {
        free(bmp_bytes);
        fclose(bmp_file);
        unlink(temp_path);
        return EIO;
    }
    fclose(bmp_file);
    unlink(temp_path);

    parse_result = parse_bmp_frame(
        bmp_bytes,
        bmp_size,
        capture->config.width,
        capture->config.height,
        capture->config.pixel_format,
        destination,
        capacity
    );
    free(bmp_bytes);
    return parse_result;
}
#endif

static int open_backend(camera_capture_t *capture) {
#if defined(__linux__)
    char resolved_command[sizeof(capture->backend_command)];

    if (capture->backend_mode != CAMERA_BACKEND_NONE) {
        return CAMERA_CAPTURE_OK;
    }
    if (
        capture->config.pixel_format != PIXEL_FORMAT_BGR24
        && capture->config.pixel_format != PIXEL_FORMAT_RGB24
    ) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    if (
        !find_command_on_path("rpicam-still", resolved_command, sizeof(resolved_command))
        && !find_command_on_path("libcamera-still", resolved_command, sizeof(resolved_command))
    ) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    strncpy(capture->backend_command, resolved_command, sizeof(capture->backend_command) - 1U);
    capture->backend_command[sizeof(capture->backend_command) - 1U] = '\0';
    capture->backend_mode = CAMERA_BACKEND_PI_CLI;
    capture->backend_frame_size_bytes = frame_byte_capacity(capture);
    capture->backend_fourcc = 0U;
    return CAMERA_CAPTURE_OK;
#else
    (void) capture;
    return CAMERA_CAPTURE_ERR_UNSUPPORTED;
#endif
}

static void close_backend(camera_capture_t *capture) {
    capture->backend_mode = CAMERA_BACKEND_NONE;
    capture->backend_command[0] = '\0';
    capture->backend_fd = -1;
    capture->backend_buffer_type = 0U;
    capture->backend_fourcc = 0U;
    capture->backend_frame_size_bytes = 0U;
    free(capture->backend_frame_buffer);
    capture->backend_frame_buffer = NULL;
    capture->backend_streaming = false;
    capture->backend_stream_buffer_count = 0U;
}

static int read_backend_frame(
    camera_capture_t *capture,
    uint8_t *destination,
    size_t capacity
) {
#if defined(__linux__)
    if (capture->backend_mode != CAMERA_BACKEND_PI_CLI) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }
    return capture_with_pi_command(capture, destination, capacity);
#else
    (void) capture;
    (void) destination;
    (void) capacity;
    return CAMERA_CAPTURE_ERR_UNSUPPORTED;
#endif
}

static void push_frame(
    camera_capture_t *capture,
    const uint8_t *data,
    size_t data_size,
    uint64_t timestamp_ns
) {
    video_frame_t *slot;

    pthread_mutex_lock(&capture->lock);
    slot = &capture->frames[capture->write_index];
    memcpy(slot->frame_bytes, data, data_size);
    slot->frame_size_bytes = data_size;
    slot->timestamp_ns = timestamp_ns;

    capture->write_index = (capture->write_index + 1U) % capture->config.buffer_capacity;
    if (capture->frame_count == capture->config.buffer_capacity) {
        capture->status.dropped_count += 1U;
    } else {
        capture->frame_count += 1U;
    }

    capture->status.healthy = true;
    capture->status.opened = true;
    capture->status.running = true;
    capture->status.last_timestamp_ns = timestamp_ns;
    capture->status.last_error_code = 0;
    pthread_mutex_unlock(&capture->lock);
}

static void *capture_thread_main(void *arg) {
    camera_capture_t *capture = arg;
    uint8_t *scratch;
    const size_t capacity = frame_byte_capacity(capture);

    scratch = malloc(capacity);
    if (scratch == NULL) {
        mark_failure(capture, ENOMEM);
        pthread_mutex_lock(&capture->lock);
        capture->status.running = false;
        pthread_mutex_unlock(&capture->lock);
        return NULL;
    }

    while (!atomic_load(&capture->stop_requested)) {
        const int result = read_backend_frame(capture, scratch, capacity);

        if (result != CAMERA_CAPTURE_OK) {
            if (result != CAMERA_CAPTURE_ERR_EMPTY) {
                mark_failure(capture, result);
            }
            sleep_us(100000L);
            continue;
        }

        push_frame(capture, scratch, capacity, now_ns());
    }

    free(scratch);
    pthread_mutex_lock(&capture->lock);
    capture->status.running = false;
    pthread_mutex_unlock(&capture->lock);
    return NULL;
}

int camera_capture_init(camera_capture_t *capture, const camera_capture_config_t *config) {
    int result;

    if (capture == NULL || config == NULL || config->buffer_capacity == 0U) {
        return CAMERA_CAPTURE_ERR_INVALID;
    }

    memset(capture, 0, sizeof(*capture));
    capture->config = *config;
    capture->backend_mode = CAMERA_BACKEND_NONE;
    capture->backend_fd = -1;
    capture->status.healthy = false;
    capture->status.opened = false;
    capture->status.running = false;
    capture->status.last_error_code = 0;
    pthread_mutex_init(&capture->lock, NULL);
    atomic_store(&capture->stop_requested, false);
    atomic_store(&capture->initialized, true);
    atomic_store(&capture->thread_started, false);

    result = allocate_frames(capture);
    if (result != CAMERA_CAPTURE_OK) {
        camera_capture_destroy(capture);
        return result;
    }

    return CAMERA_CAPTURE_OK;
}

int camera_capture_start(camera_capture_t *capture) {
    int result;

    if (capture == NULL || !atomic_load(&capture->initialized)) {
        return CAMERA_CAPTURE_ERR_INVALID;
    }

    result = open_backend(capture);
    if (result != CAMERA_CAPTURE_OK) {
        pthread_mutex_lock(&capture->lock);
        capture->status.healthy = false;
        capture->status.opened = false;
        capture->status.running = false;
        capture->status.last_error_code = result;
        pthread_mutex_unlock(&capture->lock);
        return result;
    }

    atomic_store(&capture->stop_requested, false);
    result = pthread_create(&capture->thread, NULL, capture_thread_main, capture);
    if (result != 0) {
        pthread_mutex_lock(&capture->lock);
        capture->status.last_error_code = result;
        pthread_mutex_unlock(&capture->lock);
        close_backend(capture);
        return CAMERA_CAPTURE_ERR_THREAD;
    }

    atomic_store(&capture->thread_started, true);
    pthread_mutex_lock(&capture->lock);
    capture->status.opened = true;
    capture->status.running = true;
    pthread_mutex_unlock(&capture->lock);
    return CAMERA_CAPTURE_OK;
}

int camera_capture_stop(camera_capture_t *capture) {
    if (capture == NULL || !atomic_load(&capture->initialized)) {
        return CAMERA_CAPTURE_ERR_INVALID;
    }

    atomic_store(&capture->stop_requested, true);
    if (atomic_load(&capture->thread_started)) {
        pthread_join(capture->thread, NULL);
        atomic_store(&capture->thread_started, false);
    }

    close_backend(capture);

    pthread_mutex_lock(&capture->lock);
    capture->status.running = false;
    capture->status.opened = false;
    pthread_mutex_unlock(&capture->lock);
    return CAMERA_CAPTURE_OK;
}

int camera_capture_get_latest_frame(camera_capture_t *capture, video_frame_t *out_frame) {
    video_frame_t *source;
    size_t latest_index;

    if (capture == NULL || out_frame == NULL) {
        return CAMERA_CAPTURE_ERR_EMPTY;
    }

    pthread_mutex_lock(&capture->lock);
    if (capture->frame_count == 0U) {
        pthread_mutex_unlock(&capture->lock);
        return CAMERA_CAPTURE_ERR_EMPTY;
    }

    latest_index = (capture->write_index + capture->config.buffer_capacity - 1U)
        % capture->config.buffer_capacity;
    source = &capture->frames[latest_index];

    memset(out_frame, 0, sizeof(*out_frame));
    strncpy(out_frame->client_id, source->client_id, CAPTURE_CLIENT_ID_MAX - 1U);
    out_frame->client_type = source->client_type;
    out_frame->timestamp_ns = source->timestamp_ns;
    out_frame->width = source->width;
    out_frame->height = source->height;
    out_frame->pixel_format = source->pixel_format;
    out_frame->frame_size_bytes = source->frame_size_bytes;
    out_frame->frame_bytes = malloc(source->frame_size_bytes);
    if (out_frame->frame_bytes == NULL) {
        pthread_mutex_unlock(&capture->lock);
        return CAMERA_CAPTURE_ERR_ALLOC;
    }
    memcpy(out_frame->frame_bytes, source->frame_bytes, source->frame_size_bytes);

    capture->frame_count = 0U;
    pthread_mutex_unlock(&capture->lock);
    return CAMERA_CAPTURE_OK;
}

capture_status_t camera_capture_get_status(camera_capture_t *capture) {
    capture_status_t status;

    memset(&status, 0, sizeof(status));
    if (capture == NULL) {
        return status;
    }

    pthread_mutex_lock(&capture->lock);
    status = capture->status;
    pthread_mutex_unlock(&capture->lock);
    return status;
}

void camera_capture_release_frame(video_frame_t *frame) {
    if (frame == NULL) {
        return;
    }
    free(frame->frame_bytes);
    frame->frame_bytes = NULL;
    frame->frame_size_bytes = 0U;
}

void camera_capture_destroy(camera_capture_t *capture) {
    if (capture == NULL) {
        return;
    }

    if (atomic_load(&capture->initialized)) {
        camera_capture_stop(capture);
        pthread_mutex_destroy(&capture->lock);
    }
    free_frames(capture);
    atomic_store(&capture->initialized, false);
}
