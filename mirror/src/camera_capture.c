#include "camera_capture.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <unistd.h>
#endif

#define CAMERA_CAPTURE_OK 0
#define CAMERA_CAPTURE_ERR_INVALID 1
#define CAMERA_CAPTURE_ERR_ALLOC 2
#define CAMERA_CAPTURE_ERR_THREAD 3
#define CAMERA_CAPTURE_ERR_UNSUPPORTED 4
#define CAMERA_CAPTURE_ERR_EMPTY 5

#define CAMERA_DEVICE_PATH "/dev/video0"
#define CAMERA_WAIT_TIMEOUT_US 100000

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
    return (size_t) capture->config.width * (size_t) capture->config.height * bytes_per_pixel(capture->config.pixel_format);
}

static void clear_frame(video_frame_t *frame) {
    if (frame == NULL) {
        return;
    }
    frame->timestamp_ns = 0;
    frame->frame_size_bytes = 0;
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

    for (i = 0; i < capture->config.buffer_capacity; ++i) {
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

    for (i = 0; i < capture->config.buffer_capacity; ++i) {
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
static uint32_t requested_fourcc(pixel_format_t pixel_format) {
    switch (pixel_format) {
        case PIXEL_FORMAT_BGR24:
            return V4L2_PIX_FMT_BGR24;
        case PIXEL_FORMAT_RGB24:
            return V4L2_PIX_FMT_RGB24;
        default:
            return 0U;
    }
}

static uint32_t alternate_fourcc(pixel_format_t pixel_format) {
    switch (pixel_format) {
        case PIXEL_FORMAT_BGR24:
            return V4L2_PIX_FMT_RGB24;
        case PIXEL_FORMAT_RGB24:
            return V4L2_PIX_FMT_BGR24;
        default:
            return 0U;
    }
}

static size_t backend_frame_size(uint32_t fourcc, uint32_t width, uint32_t height) {
    switch (fourcc) {
        case V4L2_PIX_FMT_YUYV:
            return (size_t) width * (size_t) height * 2U;
        case V4L2_PIX_FMT_RGB24:
        case V4L2_PIX_FMT_BGR24:
            return (size_t) width * (size_t) height * 3U;
        default:
            return 0U;
    }
}

static uint8_t clamp_channel(int value) {
    if (value < 0) {
        return 0U;
    }
    if (value > 255) {
        return 255U;
    }
    return (uint8_t) value;
}

static void convert_yuyv_to_rgb24(const uint8_t *source, uint8_t *destination, uint32_t width, uint32_t height, bool bgr_output) {
    size_t pixel_index;
    const size_t pixel_count = (size_t) width * (size_t) height;

    for (pixel_index = 0; pixel_index + 1U < pixel_count; pixel_index += 2U) {
        const size_t source_index = pixel_index * 2U;
        const int y0 = (int) source[source_index + 0U];
        const int u = (int) source[source_index + 1U] - 128;
        const int y1 = (int) source[source_index + 2U];
        const int v = (int) source[source_index + 3U] - 128;
        const int c0 = y0 - 16;
        const int c1 = y1 - 16;
        const int d = u;
        const int e = v;
        int red;
        int green;
        int blue;
        size_t destination_index;

        red = (298 * c0 + 409 * e + 128) >> 8;
        green = (298 * c0 - 100 * d - 208 * e + 128) >> 8;
        blue = (298 * c0 + 516 * d + 128) >> 8;
        destination_index = pixel_index * 3U;
        if (bgr_output) {
            destination[destination_index + 0U] = clamp_channel(blue);
            destination[destination_index + 1U] = clamp_channel(green);
            destination[destination_index + 2U] = clamp_channel(red);
        } else {
            destination[destination_index + 0U] = clamp_channel(red);
            destination[destination_index + 1U] = clamp_channel(green);
            destination[destination_index + 2U] = clamp_channel(blue);
        }

        red = (298 * c1 + 409 * e + 128) >> 8;
        green = (298 * c1 - 100 * d - 208 * e + 128) >> 8;
        blue = (298 * c1 + 516 * d + 128) >> 8;
        destination_index += 3U;
        if (bgr_output) {
            destination[destination_index + 0U] = clamp_channel(blue);
            destination[destination_index + 1U] = clamp_channel(green);
            destination[destination_index + 2U] = clamp_channel(red);
        } else {
            destination[destination_index + 0U] = clamp_channel(red);
            destination[destination_index + 1U] = clamp_channel(green);
            destination[destination_index + 2U] = clamp_channel(blue);
        }
    }
}

static void swap_rgb_channels(const uint8_t *source, uint8_t *destination, size_t pixel_count) {
    size_t pixel_index;

    for (pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
        const size_t offset = pixel_index * 3U;
        destination[offset + 0U] = source[offset + 2U];
        destination[offset + 1U] = source[offset + 1U];
        destination[offset + 2U] = source[offset + 0U];
    }
}

static int open_backend(camera_capture_t *capture) {
    struct v4l2_capability capability;
    const uint32_t desired_fourcc = requested_fourcc(capture->config.pixel_format);
    const uint32_t fallback_fourcc = alternate_fourcc(capture->config.pixel_format);
    const uint32_t candidates[3] = {desired_fourcc, fallback_fourcc, V4L2_PIX_FMT_YUYV};
    size_t index;
    int device_fd;

    if (capture->backend_fd >= 0) {
        return CAMERA_CAPTURE_OK;
    }
    if (desired_fourcc == 0U) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    device_fd = open(CAMERA_DEVICE_PATH, O_RDWR | O_NONBLOCK);
    if (device_fd < 0) {
        return errno;
    }

    memset(&capability, 0, sizeof(capability));
    if (ioctl(device_fd, VIDIOC_QUERYCAP, &capability) < 0) {
        const int error_code = errno;
        close(device_fd);
        return error_code;
    }
    if ((capability.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U) {
        close(device_fd);
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    for (index = 0U; index < 3U; ++index) {
        struct v4l2_format format;

        memset(&format, 0, sizeof(format));
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = capture->config.width;
        format.fmt.pix.height = capture->config.height;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        format.fmt.pix.pixelformat = candidates[index];
        if (format.fmt.pix.pixelformat == 0U) {
            continue;
        }
        if (ioctl(device_fd, VIDIOC_S_FMT, &format) < 0) {
            continue;
        }
        if (format.fmt.pix.width != capture->config.width || format.fmt.pix.height != capture->config.height) {
            continue;
        }
        if (format.fmt.pix.pixelformat != candidates[index]) {
            continue;
        }

        capture->backend_frame_size_bytes = backend_frame_size(
            format.fmt.pix.pixelformat,
            capture->config.width,
            capture->config.height
        );
        if (capture->backend_frame_size_bytes == 0U) {
            close(device_fd);
            return CAMERA_CAPTURE_ERR_UNSUPPORTED;
        }

        capture->backend_frame_buffer = malloc(capture->backend_frame_size_bytes);
        if (capture->backend_frame_buffer == NULL) {
            close(device_fd);
            return CAMERA_CAPTURE_ERR_ALLOC;
        }

        capture->backend_fd = device_fd;
        capture->backend_fourcc = format.fmt.pix.pixelformat;
        return CAMERA_CAPTURE_OK;
    }

    close(device_fd);
    return CAMERA_CAPTURE_ERR_UNSUPPORTED;
}

static void close_backend(camera_capture_t *capture) {
    if (capture->backend_fd >= 0) {
        close(capture->backend_fd);
        capture->backend_fd = -1;
    }
    free(capture->backend_frame_buffer);
    capture->backend_frame_buffer = NULL;
    capture->backend_frame_size_bytes = 0U;
    capture->backend_fourcc = 0U;
}

static int wait_for_frame(camera_capture_t *capture) {
    while (!atomic_load(&capture->stop_requested)) {
        fd_set read_fds;
        struct timeval timeout;
        int select_result;

        FD_ZERO(&read_fds);
        FD_SET(capture->backend_fd, &read_fds);
        timeout.tv_sec = 0;
        timeout.tv_usec = CAMERA_WAIT_TIMEOUT_US;

        select_result = select(capture->backend_fd + 1, &read_fds, NULL, NULL, &timeout);
        if (select_result > 0) {
            return CAMERA_CAPTURE_OK;
        }
        if (select_result == 0) {
            continue;
        }
        if (errno == EINTR) {
            continue;
        }
        return errno;
    }

    return CAMERA_CAPTURE_ERR_EMPTY;
}

static int read_device_bytes(camera_capture_t *capture, uint8_t *destination, size_t capacity) {
    size_t bytes_read = 0U;

    while (bytes_read < capacity && !atomic_load(&capture->stop_requested)) {
        const ssize_t result = read(capture->backend_fd, destination + bytes_read, capacity - bytes_read);
        if (result > 0) {
            bytes_read += (size_t) result;
            continue;
        }
        if (result < 0 && (errno == EAGAIN || errno == EINTR)) {
            const int wait_result = wait_for_frame(capture);
            if (wait_result != CAMERA_CAPTURE_OK) {
                return wait_result;
            }
            continue;
        }
        return errno == 0 ? CAMERA_CAPTURE_ERR_EMPTY : errno;
    }

    return bytes_read == capacity ? CAMERA_CAPTURE_OK : CAMERA_CAPTURE_ERR_EMPTY;
}
#endif

/*
 * The Raspberry Pi capture path now uses a small V4L2 backend on Linux.
 * This keeps the public API and bounded-buffer behavior stable while giving
 * us a direct hardware smoke-test path on the Pi.
 */
static int read_backend_frame(camera_capture_t *capture, uint8_t *destination, size_t capacity) {
#if defined(__linux__)
    const size_t pixel_count = (size_t) capture->config.width * (size_t) capture->config.height;
    int result;

    if (capture->backend_fd < 0 || capture->backend_frame_buffer == NULL) {
        return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }

    result = read_device_bytes(capture, capture->backend_frame_buffer, capture->backend_frame_size_bytes);
    if (result != CAMERA_CAPTURE_OK) {
        return result;
    }

    switch (capture->backend_fourcc) {
        case V4L2_PIX_FMT_BGR24:
            if (capture->config.pixel_format == PIXEL_FORMAT_BGR24) {
                memcpy(destination, capture->backend_frame_buffer, capacity);
            } else {
                swap_rgb_channels(capture->backend_frame_buffer, destination, pixel_count);
            }
            return CAMERA_CAPTURE_OK;
        case V4L2_PIX_FMT_RGB24:
            if (capture->config.pixel_format == PIXEL_FORMAT_RGB24) {
                memcpy(destination, capture->backend_frame_buffer, capacity);
            } else {
                swap_rgb_channels(capture->backend_frame_buffer, destination, pixel_count);
            }
            return CAMERA_CAPTURE_OK;
        case V4L2_PIX_FMT_YUYV:
            convert_yuyv_to_rgb24(
                capture->backend_frame_buffer,
                destination,
                capture->config.width,
                capture->config.height,
                capture->config.pixel_format == PIXEL_FORMAT_BGR24
            );
            return CAMERA_CAPTURE_OK;
        default:
            return CAMERA_CAPTURE_ERR_UNSUPPORTED;
    }
#else
    (void) capture;
    (void) destination;
    (void) capacity;
    return CAMERA_CAPTURE_ERR_UNSUPPORTED;
#endif
}

static void push_frame(camera_capture_t *capture, const uint8_t *data, size_t data_size, uint64_t timestamp_ns) {
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
            sleep_us(CAMERA_WAIT_TIMEOUT_US);
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

#if defined(__linux__)
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
#else
    pthread_mutex_lock(&capture->lock);
    capture->status.last_error_code = CAMERA_CAPTURE_ERR_UNSUPPORTED;
    pthread_mutex_unlock(&capture->lock);
    return CAMERA_CAPTURE_ERR_UNSUPPORTED;
#endif

    atomic_store(&capture->stop_requested, false);
    result = pthread_create(&capture->thread, NULL, capture_thread_main, capture);
    if (result != 0) {
        pthread_mutex_lock(&capture->lock);
        capture->status.last_error_code = result;
        pthread_mutex_unlock(&capture->lock);
#if defined(__linux__)
        close_backend(capture);
#endif
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

#if defined(__linux__)
    close_backend(capture);
#endif

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

    latest_index = (capture->write_index + capture->config.buffer_capacity - 1U) % capture->config.buffer_capacity;
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
