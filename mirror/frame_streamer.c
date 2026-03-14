#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define FRAME_MAGIC "CSJ1"
#define EVENT_MAGIC "CSM1"
#define FRAME_HEADER_SIZE 24U
#define DEFAULT_PORT 9000
#define DEFAULT_WIDTH 640U
#define DEFAULT_HEIGHT 480U
#define DEFAULT_FPS 15U
#define PIPE_READ_CHUNK 4096U
#define INITIAL_FRAME_CAPACITY (256U * 1024U)
#define MAX_FRAME_CAPACITY (8U * 1024U * 1024U)
#define CONTROL_BUFFER_CAPACITY 4096U

#define EVENT_HELLO 1U
#define EVENT_STATE 2U
#define EVENT_FEEDBACK 3U

#define HELLO_VERSION 1U
#define SOURCE_MIRROR 2U

#define CAPABILITY_SEND_VIDEO (1U << 0)
#define CAPABILITY_RECEIVE_STATE (1U << 2)
#define CAPABILITY_BINARY_CONTROL (1U << 4)

static uint64_t now_ns(void) {
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t) ts.tv_sec * 1000000000ULL) + (uint64_t) ts.tv_nsec;
}

static void write_be32(uint8_t *buffer, uint32_t value) {
    buffer[0] = (uint8_t) ((value >> 24U) & 0xffU);
    buffer[1] = (uint8_t) ((value >> 16U) & 0xffU);
    buffer[2] = (uint8_t) ((value >> 8U) & 0xffU);
    buffer[3] = (uint8_t) (value & 0xffU);
}

static void write_be64(uint8_t *buffer, uint64_t value) {
    buffer[0] = (uint8_t) ((value >> 56U) & 0xffU);
    buffer[1] = (uint8_t) ((value >> 48U) & 0xffU);
    buffer[2] = (uint8_t) ((value >> 40U) & 0xffU);
    buffer[3] = (uint8_t) ((value >> 32U) & 0xffU);
    buffer[4] = (uint8_t) ((value >> 24U) & 0xffU);
    buffer[5] = (uint8_t) ((value >> 16U) & 0xffU);
    buffer[6] = (uint8_t) ((value >> 8U) & 0xffU);
    buffer[7] = (uint8_t) (value & 0xffU);
}

static uint32_t read_be32(const uint8_t *buffer) {
    return ((uint32_t) buffer[0] << 24U)
        | ((uint32_t) buffer[1] << 16U)
        | ((uint32_t) buffer[2] << 8U)
        | (uint32_t) buffer[3];
}

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

static bool send_all(int socket_fd, const uint8_t *buffer, size_t length) {
    size_t total_sent = 0U;

    while (total_sent < length) {
        const ssize_t bytes_sent = send(
            socket_fd,
            buffer + total_sent,
            length - total_sent,
            0
        );
        if (bytes_sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (bytes_sent == 0) {
            return false;
        }
        total_sent += (size_t) bytes_sent;
    }

    return true;
}

static int connect_to_server(const char *server_host, uint16_t port) {
    struct sockaddr_in server_addr;
    int socket_fd;

    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        return -1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, server_host, &server_addr.sin_addr) != 1) {
        close(socket_fd);
        errno = EINVAL;
        return -1;
    }

    if (connect(socket_fd, (const struct sockaddr *) &server_addr, sizeof(server_addr)) != 0) {
        close(socket_fd);
        return -1;
    }

    return socket_fd;
}

static int start_camera_process(
    const char *command,
    uint32_t width,
    uint32_t height,
    uint32_t fps,
    pid_t *child_pid,
    int *stdout_fd
) {
    int pipe_fds[2];
    pid_t pid;

    if (pipe(pipe_fds) != 0) {
        return errno;
    }

    pid = fork();
    if (pid < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return errno;
    }

    if (pid == 0) {
        char width_arg[16];
        char height_arg[16];
        char fps_arg[16];
        char *const argv[] = {
            (char *) command,
            "--nopreview",
            "--codec",
            "mjpeg",
            "--timeout",
            "0",
            "--width",
            width_arg,
            "--height",
            height_arg,
            "--framerate",
            fps_arg,
            "--output",
            "-",
            NULL,
        };

        close(pipe_fds[0]);
        if (dup2(pipe_fds[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }
        close(pipe_fds[1]);

        snprintf(width_arg, sizeof(width_arg), "%u", width);
        snprintf(height_arg, sizeof(height_arg), "%u", height);
        snprintf(fps_arg, sizeof(fps_arg), "%u", fps);
        execv(command, argv);
        _exit(127);
    }

    close(pipe_fds[1]);
    *child_pid = pid;
    *stdout_fd = pipe_fds[0];
    return 0;
}

static void stop_camera_process(pid_t child_pid, int stdout_fd) {
    int wait_status;

    if (stdout_fd >= 0) {
        close(stdout_fd);
    }
    if (child_pid <= 0) {
        return;
    }

    kill(child_pid, SIGTERM);
    waitpid(child_pid, &wait_status, 0);
}

static bool send_frame_packet(
    int socket_fd,
    uint32_t width,
    uint32_t height,
    const uint8_t *frame_bytes,
    size_t frame_size
) {
    uint8_t header[FRAME_HEADER_SIZE];

    memcpy(header, FRAME_MAGIC, 4U);
    write_be32(header + 4U, width);
    write_be32(header + 8U, height);
    write_be32(header + 12U, (uint32_t) frame_size);
    write_be64(header + 16U, now_ns());

    return send_all(socket_fd, header, sizeof(header))
        && send_all(socket_fd, frame_bytes, frame_size);
}

static bool send_control_packet(
    int socket_fd,
    uint32_t message_type,
    uint32_t flags,
    const uint8_t *payload,
    size_t payload_size
) {
    uint8_t header[FRAME_HEADER_SIZE];

    memcpy(header, EVENT_MAGIC, 4U);
    write_be32(header + 4U, message_type);
    write_be32(header + 8U, flags);
    write_be32(header + 12U, (uint32_t) payload_size);
    write_be64(header + 16U, now_ns());

    return send_all(socket_fd, header, sizeof(header))
        && send_all(socket_fd, payload, payload_size);
}

static bool send_hello_packet(int socket_fd) {
    uint8_t payload[4] = {HELLO_VERSION, SOURCE_MIRROR, 0U, 0U};
    const uint32_t flags = CAPABILITY_SEND_VIDEO
        | CAPABILITY_RECEIVE_STATE
        | CAPABILITY_BINARY_CONTROL;

    return send_control_packet(
        socket_fd,
        EVENT_HELLO,
        flags,
        payload,
        sizeof(payload)
    );
}

static const char *state_label(uint8_t state_id) {
    switch (state_id) {
        case 1U:
            return "focused";
        case 2U:
            return "fatigued";
        case 3U:
            return "stressed";
        case 4U:
            return "distracted";
        default:
            return "unknown";
    }
}

static const char *indicator_label(uint8_t indicator_id) {
    switch (indicator_id) {
        case 1U:
            return "blink rate elevated";
        case 2U:
            return "blink rate suppressed";
        case 3U:
            return "posture slouched";
        case 4U:
            return "posture leaning";
        case 5U:
            return "eye movement distracted";
        case 6U:
            return "facial tension detected";
        case 7U:
            return "speech tone stressed";
        case 8U:
            return "speech tone monotone";
        case 9U:
            return "posture upright";
        case 10U:
            return "eye engagement focused";
        case 11U:
            return "facial expression relaxed";
        case 12U:
            return "speech tone calm";
        default:
            return NULL;
    }
}

static const char *recommendation_label(uint8_t recommendation_id) {
    switch (recommendation_id) {
        case 1U:
            return "take a 10 minute break";
        case 2U:
            return "hydrate";
        case 3U:
            return "stretch and reset posture";
        case 4U:
            return "take a breathing pause";
        case 5U:
            return "refocus on one task";
        case 6U:
            return "silence distractions";
        case 7U:
            return "keep your current pace";
        case 8U:
            return "reset your posture";
        default:
            return NULL;
    }
}

static void drain_control_packets(
    int socket_fd,
    uint8_t *control_buffer,
    size_t *control_size,
    uint8_t *last_state_id,
    uint8_t *last_confidence,
    uint8_t *last_indicator_1,
    uint8_t *last_indicator_2,
    uint8_t *last_indicator_3,
    uint8_t *last_recommendation_1,
    uint8_t *last_recommendation_2,
    uint8_t *last_recommendation_3
) {
    while (true) {
        const ssize_t bytes_read = recv(
            socket_fd,
            control_buffer + *control_size,
            CONTROL_BUFFER_CAPACITY - *control_size,
            MSG_DONTWAIT
        );

        if (bytes_read < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            fprintf(stderr, "Control receive failed: %s\n", strerror(errno));
            break;
        }

        if (bytes_read == 0) {
            break;
        }

        *control_size += (size_t) bytes_read;
        while (*control_size >= FRAME_HEADER_SIZE) {
            const uint32_t payload_size = read_be32(control_buffer + 12U);
            const size_t packet_size = FRAME_HEADER_SIZE + (size_t) payload_size;
            if (*control_size < packet_size) {
                break;
            }

            if (
                memcmp(control_buffer, EVENT_MAGIC, 4U) == 0
                && read_be32(control_buffer + 4U) == EVENT_STATE
                && payload_size >= 8U
            ) {
                const uint8_t *payload = control_buffer + FRAME_HEADER_SIZE;
                const uint8_t state_id = payload[1];
                const uint8_t confidence = payload[2];
                const uint8_t indicator_1 = payload_size >= 11U ? payload[8] : 0U;
                const uint8_t indicator_2 = payload_size >= 11U ? payload[9] : 0U;
                const uint8_t indicator_3 = payload_size >= 11U ? payload[10] : 0U;
                const uint8_t recommendation_1 = payload_size >= 14U ? payload[11] : 0U;
                const uint8_t recommendation_2 = payload_size >= 14U ? payload[12] : 0U;
                const uint8_t recommendation_3 = payload_size >= 14U ? payload[13] : 0U;

                if (
                    *last_state_id != state_id
                    || *last_confidence != confidence
                    || *last_indicator_1 != indicator_1
                    || *last_indicator_2 != indicator_2
                    || *last_indicator_3 != indicator_3
                    || *last_recommendation_1 != recommendation_1
                    || *last_recommendation_2 != recommendation_2
                    || *last_recommendation_3 != recommendation_3
                ) {
                    const char *indicator_1_text = indicator_label(indicator_1);
                    const char *indicator_2_text = indicator_label(indicator_2);
                    const char *indicator_3_text = indicator_label(indicator_3);
                    const char *recommendation_1_text = recommendation_label(recommendation_1);
                    const char *recommendation_2_text = recommendation_label(recommendation_2);
                    const char *recommendation_3_text = recommendation_label(recommendation_3);

                    printf(
                        "Server state: %s (%u%%)\n",
                        state_label(state_id),
                        (unsigned int) ((confidence * 100U + 127U) / 255U)
                    );
                    if (
                        indicator_1_text != NULL
                        || indicator_2_text != NULL
                        || indicator_3_text != NULL
                    ) {
                        printf("Indicators:\n");
                        if (indicator_1_text != NULL) {
                            printf("  - %s\n", indicator_1_text);
                        }
                        if (indicator_2_text != NULL) {
                            printf("  - %s\n", indicator_2_text);
                        }
                        if (indicator_3_text != NULL) {
                            printf("  - %s\n", indicator_3_text);
                        }
                    }
                    if (
                        recommendation_1_text != NULL
                        || recommendation_2_text != NULL
                        || recommendation_3_text != NULL
                    ) {
                        printf("Recommendations:\n");
                        if (recommendation_1_text != NULL) {
                            printf("  - %s\n", recommendation_1_text);
                        }
                        if (recommendation_2_text != NULL) {
                            printf("  - %s\n", recommendation_2_text);
                        }
                        if (recommendation_3_text != NULL) {
                            printf("  - %s\n", recommendation_3_text);
                        }
                    }
                    fflush(stdout);
                    *last_state_id = state_id;
                    *last_confidence = confidence;
                    *last_indicator_1 = indicator_1;
                    *last_indicator_2 = indicator_2;
                    *last_indicator_3 = indicator_3;
                    *last_recommendation_1 = recommendation_1;
                    *last_recommendation_2 = recommendation_2;
                    *last_recommendation_3 = recommendation_3;
                }
            } else if (
                memcmp(control_buffer, EVENT_MAGIC, 4U) == 0
                && read_be32(control_buffer + 4U) == EVENT_FEEDBACK
                && payload_size > 0U
            ) {
                const uint8_t *payload = control_buffer + FRAME_HEADER_SIZE;
                printf("LLM feedback: %.*s\n", (int) payload_size, (const char *) payload);
                fflush(stdout);
            }

            memmove(
                control_buffer,
                control_buffer + packet_size,
                *control_size - packet_size
            );
            *control_size -= packet_size;
        }

        if (*control_size == CONTROL_BUFFER_CAPACITY) {
            *control_size = 0U;
        }
    }
}

int main(int argc, char **argv) {
    const char *server_host;
    uint16_t port;
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    char camera_command[256];
    pid_t child_pid = -1;
    int camera_stdout_fd = -1;
    int socket_fd = -1;
    uint8_t read_buffer[PIPE_READ_CHUNK];
    uint8_t control_buffer[CONTROL_BUFFER_CAPACITY];
    uint8_t *frame_buffer = NULL;
    size_t frame_capacity = INITIAL_FRAME_CAPACITY;
    size_t frame_size = 0U;
    size_t control_size = 0U;
    bool collecting = false;
    bool previous_was_ff = false;
    uint64_t frame_count = 0U;
    uint8_t last_state_id = 0xffU;
    uint8_t last_confidence = 0xffU;
    uint8_t last_indicator_1 = 0xffU;
    uint8_t last_indicator_2 = 0xffU;
    uint8_t last_indicator_3 = 0xffU;
    uint8_t last_recommendation_1 = 0xffU;
    uint8_t last_recommendation_2 = 0xffU;
    uint8_t last_recommendation_3 = 0xffU;

    signal(SIGPIPE, SIG_IGN);

    if (argc < 2) {
        fprintf(
            stderr,
            "Usage: %s <server_ip> [port] [width] [height] [fps]\n",
            argv[0]
        );
        return 1;
    }

    server_host = argv[1];
    port = argc > 2 ? (uint16_t) strtoul(argv[2], NULL, 10) : DEFAULT_PORT;
    width = argc > 3 ? (uint32_t) strtoul(argv[3], NULL, 10) : DEFAULT_WIDTH;
    height = argc > 4 ? (uint32_t) strtoul(argv[4], NULL, 10) : DEFAULT_HEIGHT;
    fps = argc > 5 ? (uint32_t) strtoul(argv[5], NULL, 10) : DEFAULT_FPS;

    if (
        !find_command_on_path("rpicam-vid", camera_command, sizeof(camera_command))
        && !find_command_on_path("libcamera-vid", camera_command, sizeof(camera_command))
    ) {
        fprintf(stderr, "Could not find rpicam-vid or libcamera-vid on PATH\n");
        return 1;
    }

    frame_buffer = malloc(frame_capacity);
    if (frame_buffer == NULL) {
        fprintf(stderr, "Failed to allocate frame buffer\n");
        return 1;
    }

    socket_fd = connect_to_server(server_host, port);
    if (socket_fd < 0) {
        fprintf(stderr, "Failed to connect to %s:%u: %s\n", server_host, port, strerror(errno));
        free(frame_buffer);
        return 1;
    }

    if (!send_hello_packet(socket_fd)) {
        fprintf(stderr, "Failed to send hello packet: %s\n", strerror(errno));
        close(socket_fd);
        free(frame_buffer);
        return 1;
    }

    if (start_camera_process(camera_command, width, height, fps, &child_pid, &camera_stdout_fd) != 0) {
        fprintf(stderr, "Failed to start camera process: %s\n", strerror(errno));
        close(socket_fd);
        free(frame_buffer);
        return 1;
    }

    printf(
        "Streaming %ux%u MJPEG frames at %u fps to %s:%u using %s\n",
        width,
        height,
        fps,
        server_host,
        port,
        camera_command
    );

    while (true) {
        const ssize_t bytes_read = read(camera_stdout_fd, read_buffer, sizeof(read_buffer));
        size_t index;

        if (bytes_read < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "Camera read failed: %s\n", strerror(errno));
            break;
        }
        if (bytes_read == 0) {
            fprintf(stderr, "Camera process ended unexpectedly\n");
            break;
        }

        for (index = 0U; index < (size_t) bytes_read; ++index) {
            const uint8_t byte = read_buffer[index];

            if (!collecting) {
                if (previous_was_ff && byte == 0xd8U) {
                    collecting = true;
                    frame_size = 0U;
                    frame_buffer[frame_size++] = 0xffU;
                    frame_buffer[frame_size++] = 0xd8U;
                    previous_was_ff = false;
                    continue;
                }
                previous_was_ff = byte == 0xffU;
                continue;
            }

            if (frame_size == frame_capacity) {
                uint8_t *grown_buffer;

                if (frame_capacity >= MAX_FRAME_CAPACITY) {
                    fprintf(stderr, "Frame exceeded max buffer size\n");
                    collecting = false;
                    frame_size = 0U;
                    previous_was_ff = false;
                    continue;
                }
                frame_capacity *= 2U;
                if (frame_capacity > MAX_FRAME_CAPACITY) {
                    frame_capacity = MAX_FRAME_CAPACITY;
                }
                grown_buffer = realloc(frame_buffer, frame_capacity);
                if (grown_buffer == NULL) {
                    fprintf(stderr, "Failed to grow frame buffer\n");
                    goto cleanup;
                }
                frame_buffer = grown_buffer;
            }

            frame_buffer[frame_size++] = byte;
            if (previous_was_ff && byte == 0xd9U) {
                frame_count += 1U;
                if (!send_frame_packet(socket_fd, width, height, frame_buffer, frame_size)) {
                    fprintf(stderr, "Failed to send frame %llu: %s\n", (unsigned long long) frame_count, strerror(errno));
                    goto cleanup;
                }
                drain_control_packets(
                    socket_fd,
                    control_buffer,
                    &control_size,
                    &last_state_id,
                    &last_confidence,
                    &last_indicator_1,
                    &last_indicator_2,
                    &last_indicator_3,
                    &last_recommendation_1,
                    &last_recommendation_2,
                    &last_recommendation_3
                );
                if (frame_count % 30U == 0U) {
                    printf(
                        "Sent %llu frames\n",
                        (unsigned long long) frame_count
                    );
                }
                collecting = false;
                frame_size = 0U;
                previous_was_ff = false;
                continue;
            }

            previous_was_ff = byte == 0xffU;
        }
    }

cleanup:
    stop_camera_process(child_pid, camera_stdout_fd);
    if (socket_fd >= 0) {
        close(socket_fd);
    }
    free(frame_buffer);
    return 0;
}
