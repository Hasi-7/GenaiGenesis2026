#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "lcd_display.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef COGNITIVESENSE_HAVE_GPIOD
#include <gpiod.h>
#endif

#define LCD_PAGE_INTERVAL_NS 1500000000ULL
#define LCD_DEFAULT_RS_PIN 26U
#define LCD_DEFAULT_E_PIN 19U
#define LCD_DEFAULT_D4_PIN 13U
#define LCD_DEFAULT_D5_PIN 6U
#define LCD_DEFAULT_D6_PIN 5U
#define LCD_DEFAULT_D7_PIN 11U

static uint64_t now_ns(void) {
    struct timespec ts;

    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t) ts.tv_sec * 1000000000ULL) + (uint64_t) ts.tv_nsec;
}

static bool env_flag_enabled(const char *name) {
    const char *value = getenv(name);

    if (value == NULL) {
        return false;
    }
    return strcmp(value, "1") == 0
        || strcmp(value, "true") == 0
        || strcmp(value, "TRUE") == 0
        || strcmp(value, "yes") == 0
        || strcmp(value, "on") == 0;
}

static bool parse_env_pin(const char *name, unsigned int *value) {
    const char *raw = getenv(name);
    char *end = NULL;
    unsigned long parsed;

    if (raw == NULL || raw[0] == '\0' || value == NULL) {
        return false;
    }

    errno = 0;
    parsed = strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0') {
        return false;
    }
    *value = (unsigned int) parsed;
    return true;
}

static void load_default_pins(MirrorLcdDisplay *display) {
    if (display == NULL) {
        return;
    }

    display->rs_pin = LCD_DEFAULT_RS_PIN;
    display->e_pin = LCD_DEFAULT_E_PIN;
    display->d4_pin = LCD_DEFAULT_D4_PIN;
    display->d5_pin = LCD_DEFAULT_D5_PIN;
    display->d6_pin = LCD_DEFAULT_D6_PIN;
    display->d7_pin = LCD_DEFAULT_D7_PIN;
}

static void copy_lcd_line(char *dest, size_t dest_size, const char *text) {
    size_t index = 0U;

    if (dest == NULL || dest_size == 0U) {
        return;
    }

    while (index + 1U < dest_size && index < MIRROR_LCD_LINE_LENGTH) {
        char ch = ' ';
        if (text != NULL && text[index] != '\0') {
            ch = text[index];
        }
        dest[index] = (char) (isprint((unsigned char) ch) ? ch : ' ');
        index += 1U;
    }
    while (index < MIRROR_LCD_LINE_LENGTH && index + 1U < dest_size) {
        dest[index] = ' ';
        index += 1U;
    }
    dest[index] = '\0';
}

static void cleanup_feedback_pages(MirrorLcdDisplay *display) {
    size_t page_index;
    size_t line_index;

    if (display == NULL) {
        return;
    }

    for (page_index = 0U; page_index < MIRROR_LCD_MAX_FEEDBACK_PAGES; ++page_index) {
        for (line_index = 0U; line_index < 2U; ++line_index) {
            copy_lcd_line(
                display->feedback_pages[page_index][line_index],
                sizeof(display->feedback_pages[page_index][line_index]),
                ""
            );
        }
    }
    display->feedback_page_count = 0U;
    display->feedback_page_index = 0U;
    display->feedback_active_until_ns = 0ULL;
    display->next_feedback_page_at_ns = 0ULL;
}

#ifdef COGNITIVESENSE_HAVE_GPIOD
#ifdef COGNITIVESENSE_GPIOD_V2
typedef struct MirrorGpiodLineHandle {
    struct gpiod_line_request *request;
    unsigned int offset;
} MirrorGpiodLineHandle;

static MirrorGpiodLineHandle *as_line_handle(void *line) {
    return (MirrorGpiodLineHandle *) line;
}
#else
static struct gpiod_line *as_line(void *line) {
    return (struct gpiod_line *) line;
}
#endif

static struct gpiod_chip *as_chip(void *chip) {
    return (struct gpiod_chip *) chip;
}

static void release_lines(MirrorLcdDisplay *display) {
    void *lines[] = {
        display->rs_line,
        display->e_line,
        display->d4_line,
        display->d5_line,
        display->d6_line,
        display->d7_line,
    };
    size_t index;

    for (index = 0U; index < sizeof(lines) / sizeof(lines[0]); ++index) {
        if (lines[index] != NULL) {
#ifdef COGNITIVESENSE_GPIOD_V2
            MirrorGpiodLineHandle *handle = as_line_handle(lines[index]);
            if (handle->request != NULL) {
                gpiod_line_request_release(handle->request);
            }
            free(handle);
#else
            gpiod_line_release(lines[index]);
#endif
        }
    }

    display->rs_line = NULL;
    display->e_line = NULL;
    display->d4_line = NULL;
    display->d5_line = NULL;
    display->d6_line = NULL;
    display->d7_line = NULL;
}

static bool set_line_value(void *line, int value) {
    if (line == NULL) {
        return false;
    }
#ifdef COGNITIVESENSE_GPIOD_V2
    {
        MirrorGpiodLineHandle *handle = as_line_handle(line);
        return handle->request != NULL
            && gpiod_line_request_set_value(
                handle->request,
                handle->offset,
                value ? GPIOD_LINE_VALUE_ACTIVE : GPIOD_LINE_VALUE_INACTIVE
            ) == 0;
    }
#else
    return gpiod_line_set_value(as_line(line), value) == 0;
#endif
}

static bool pulse_enable(MirrorLcdDisplay *display) {
    return set_line_value(display->e_line, 1)
        && usleep(1U) == 0
        && set_line_value(display->e_line, 0)
        && usleep(50U) == 0;
}

static bool write_nibble(MirrorLcdDisplay *display, uint8_t nibble) {
    return set_line_value(display->d4_line, (nibble >> 0U) & 1U)
        && set_line_value(display->d5_line, (nibble >> 1U) & 1U)
        && set_line_value(display->d6_line, (nibble >> 2U) & 1U)
        && set_line_value(display->d7_line, (nibble >> 3U) & 1U)
        && pulse_enable(display);
}

static bool write_byte(MirrorLcdDisplay *display, uint8_t value, bool is_data) {
    if (!set_line_value(display->rs_line, is_data ? 1 : 0)) {
        return false;
    }
    if (!write_nibble(display, (uint8_t) (value >> 4U))) {
        return false;
    }
    if (!write_nibble(display, (uint8_t) (value & 0x0fU))) {
        return false;
    }
    usleep(50U);
    return true;
}

static bool write_command(MirrorLcdDisplay *display, uint8_t command) {
    const bool ok = write_byte(display, command, false);
    if ((command == 0x01U || command == 0x02U) && ok) {
        usleep(2000U);
    }
    return ok;
}

static bool write_text(MirrorLcdDisplay *display, const char *line_1, const char *line_2) {
    size_t index;

    if (!write_command(display, 0x80U)) {
        return false;
    }
    for (index = 0U; index < MIRROR_LCD_LINE_LENGTH; ++index) {
        if (!write_byte(display, (uint8_t) line_1[index], true)) {
            return false;
        }
    }

    if (!write_command(display, 0xC0U)) {
        return false;
    }
    for (index = 0U; index < MIRROR_LCD_LINE_LENGTH; ++index) {
        if (!write_byte(display, (uint8_t) line_2[index], true)) {
            return false;
        }
    }

    return true;
}

static bool request_output_line(
    struct gpiod_chip *chip,
    unsigned int offset,
    const char *consumer,
    void **target
) {
    if (chip == NULL || target == NULL) {
        return false;
    }

#ifdef COGNITIVESENSE_GPIOD_V2
    {
        struct gpiod_line_settings *line_settings = gpiod_line_settings_new();
        struct gpiod_line_config *line_config = gpiod_line_config_new();
        struct gpiod_request_config *request_config = gpiod_request_config_new();
        struct gpiod_line_request *request = NULL;
        MirrorGpiodLineHandle *handle = NULL;
        bool ok = false;

        if (
            line_settings == NULL
            || line_config == NULL
            || request_config == NULL
        ) {
            goto cleanup_v2_request;
        }
        if (
            gpiod_line_settings_set_direction(
                line_settings,
                GPIOD_LINE_DIRECTION_OUTPUT
            ) != 0
        ) {
            goto cleanup_v2_request;
        }
        if (
            gpiod_line_settings_set_output_value(
                line_settings,
                GPIOD_LINE_VALUE_INACTIVE
            ) != 0
        ) {
            goto cleanup_v2_request;
        }
        if (
            gpiod_line_config_add_line_settings(
                line_config,
                &offset,
                1U,
                line_settings
            ) != 0
        ) {
            goto cleanup_v2_request;
        }
        gpiod_request_config_set_consumer(request_config, consumer);
        request = gpiod_chip_request_lines(chip, request_config, line_config);
        if (request == NULL) {
            goto cleanup_v2_request;
        }

        handle = calloc(1U, sizeof(*handle));
        if (handle == NULL) {
            goto cleanup_v2_request;
        }
        handle->request = request;
        handle->offset = offset;
        *target = handle;
        ok = true;

cleanup_v2_request:
        gpiod_request_config_free(request_config);
        gpiod_line_config_free(line_config);
        gpiod_line_settings_free(line_settings);
        if (!ok) {
            if (request != NULL) {
                gpiod_line_request_release(request);
            }
            free(handle);
        }
        return ok;
    }
#else
    {
        struct gpiod_line *line = gpiod_chip_get_line(chip, offset);

        if (line == NULL) {
            return false;
        }
        if (gpiod_line_request_output(line, consumer, 0) != 0) {
            return false;
        }
        *target = line;
    }
#endif
    return true;
}

static bool open_chip(MirrorLcdDisplay *display) {
    const char *configured_path = getenv("COGNITIVESENSE_LCD_GPIO_CHIP");
    const char *chip_candidates[] = {
        configured_path,
        "/dev/gpiochip4",
        "/dev/gpiochip3",
        "/dev/gpiochip2",
        "/dev/gpiochip0",
        "/dev/gpiochip1",
    };
    size_t index;

    for (index = 0U; index < sizeof(chip_candidates) / sizeof(chip_candidates[0]); ++index) {
        const char *path = chip_candidates[index];
        struct gpiod_chip *chip;

        if (path == NULL || path[0] == '\0') {
            continue;
        }
        chip = gpiod_chip_open(path);
        if (chip == NULL) {
            continue;
        }
        display->chip = chip;
        snprintf(display->chip_path, sizeof(display->chip_path), "%s", path);
        return true;
    }

    return false;
}

static bool initialize_hardware(MirrorLcdDisplay *display) {
    struct gpiod_chip *chip;

    if (!open_chip(display)) {
        fprintf(
            stderr,
            "LCD enabled, but no gpiochip could be opened. Set COGNITIVESENSE_LCD_GPIO_CHIP if needed.\n"
        );
        return false;
    }

    chip = as_chip(display->chip);
    if (
        !request_output_line(chip, display->rs_pin, "cognitivesense-lcd", &display->rs_line)
        || !request_output_line(chip, display->e_pin, "cognitivesense-lcd", &display->e_line)
        || !request_output_line(chip, display->d4_pin, "cognitivesense-lcd", &display->d4_line)
        || !request_output_line(chip, display->d5_pin, "cognitivesense-lcd", &display->d5_line)
        || !request_output_line(chip, display->d6_pin, "cognitivesense-lcd", &display->d6_line)
        || !request_output_line(chip, display->d7_pin, "cognitivesense-lcd", &display->d7_line)
    ) {
        fprintf(stderr, "LCD line request failed. Check BCM pin assignments and gpiochip.\n");
        release_lines(display);
        return false;
    }

    usleep(100000U);
    set_line_value(display->rs_line, 0);
    set_line_value(display->e_line, 0);
    write_nibble(display, 0x03U);
    usleep(5000U);
    write_nibble(display, 0x03U);
    usleep(5000U);
    write_nibble(display, 0x03U);
    usleep(500U);
    write_nibble(display, 0x02U);
    usleep(500U);
    write_command(display, 0x28U);
    write_command(display, 0x0CU);
    write_command(display, 0x06U);
    write_command(display, 0x01U);
    return true;
}

static void write_if_changed(MirrorLcdDisplay *display, const char *line_1, const char *line_2) {
    if (
        strcmp(display->rendered_line_1, line_1) == 0
        && strcmp(display->rendered_line_2, line_2) == 0
    ) {
        return;
    }
    if (!write_text(display, line_1, line_2)) {
        fprintf(stderr, "LCD write failed.\n");
        return;
    }
    snprintf(display->rendered_line_1, sizeof(display->rendered_line_1), "%s", line_1);
    snprintf(display->rendered_line_2, sizeof(display->rendered_line_2), "%s", line_2);
}
#endif

void mirror_lcd_init(MirrorLcdDisplay *display) {
    if (display == NULL) {
        return;
    }

    memset(display, 0, sizeof(*display));
    display->requested = env_flag_enabled("COGNITIVESENSE_LCD_ENABLE");
    if (!display->requested) {
        return;
    }
    load_default_pins(display);

    if (
        (getenv("COGNITIVESENSE_LCD_RS_PIN") != NULL
            && !parse_env_pin("COGNITIVESENSE_LCD_RS_PIN", &display->rs_pin))
        || (getenv("COGNITIVESENSE_LCD_E_PIN") != NULL
            && !parse_env_pin("COGNITIVESENSE_LCD_E_PIN", &display->e_pin))
        || (getenv("COGNITIVESENSE_LCD_D4_PIN") != NULL
            && !parse_env_pin("COGNITIVESENSE_LCD_D4_PIN", &display->d4_pin))
        || (getenv("COGNITIVESENSE_LCD_D5_PIN") != NULL
            && !parse_env_pin("COGNITIVESENSE_LCD_D5_PIN", &display->d5_pin))
        || (getenv("COGNITIVESENSE_LCD_D6_PIN") != NULL
            && !parse_env_pin("COGNITIVESENSE_LCD_D6_PIN", &display->d6_pin))
        || (getenv("COGNITIVESENSE_LCD_D7_PIN") != NULL
            && !parse_env_pin("COGNITIVESENSE_LCD_D7_PIN", &display->d7_pin))
    ) {
        fprintf(
            stderr,
            "LCD enabled, but one or more LCD pin env vars are invalid. Expected BCM GPIO numbers for RS, E, D4, D5, D6, and D7.\n"
        );
        return;
    }

#ifdef COGNITIVESENSE_HAVE_GPIOD
    display->available = initialize_hardware(display);
    display->enabled = display->available;
    cleanup_feedback_pages(display);
    copy_lcd_line(display->state_line_1, sizeof(display->state_line_1), "Mirror streamer");
    copy_lcd_line(display->state_line_2, sizeof(display->state_line_2), "Waiting server");
    copy_lcd_line(display->rendered_line_1, sizeof(display->rendered_line_1), "");
    copy_lcd_line(display->rendered_line_2, sizeof(display->rendered_line_2), "");
    if (display->enabled) {
        fprintf(
            stderr,
            "LCD initialized on %s (RS=%u E=%u D4=%u D5=%u D6=%u D7=%u).\n",
            display->chip_path,
            display->rs_pin,
            display->e_pin,
            display->d4_pin,
            display->d5_pin,
            display->d6_pin,
            display->d7_pin
        );
        mirror_lcd_refresh(display);
    }
#else
    fprintf(
        stderr,
        "LCD enabled, but libgpiod support was not compiled in. Install libgpiod-dev on the Raspberry Pi and rebuild.\n"
    );
    display->warned_missing_support = true;
#endif
}

void mirror_lcd_close(MirrorLcdDisplay *display) {
    if (display == NULL) {
        return;
    }

#ifdef COGNITIVESENSE_HAVE_GPIOD
    release_lines(display);
    if (display->chip != NULL) {
        gpiod_chip_close(as_chip(display->chip));
        display->chip = NULL;
    }
#endif

    display->enabled = false;
    display->available = false;
}

void mirror_lcd_set_connecting(
    MirrorLcdDisplay *display,
    const char *server_host,
    uint16_t port
) {
    char buffer[32];

    if (display == NULL || !display->enabled) {
        return;
    }

    snprintf(buffer, sizeof(buffer), "%s:%u", server_host, (unsigned int) port);
    copy_lcd_line(display->state_line_1, sizeof(display->state_line_1), "Connecting...");
    copy_lcd_line(display->state_line_2, sizeof(display->state_line_2), buffer);
}

void mirror_lcd_set_streaming(
    MirrorLcdDisplay *display,
    uint32_t width,
    uint32_t height,
    uint32_t fps
) {
    char buffer[32];

    if (display == NULL || !display->enabled) {
        return;
    }

    snprintf(buffer, sizeof(buffer), "%lux%lu %lufps", (unsigned long) width, (unsigned long) height, (unsigned long) fps);
    copy_lcd_line(display->state_line_1, sizeof(display->state_line_1), "Streaming live");
    copy_lcd_line(display->state_line_2, sizeof(display->state_line_2), buffer);
}

void mirror_lcd_set_state(
    MirrorLcdDisplay *display,
    const char *state_text,
    const char *detail_text
) {
    if (display == NULL || !display->enabled) {
        return;
    }

    copy_lcd_line(display->state_line_1, sizeof(display->state_line_1), state_text);
    copy_lcd_line(display->state_line_2, sizeof(display->state_line_2), detail_text);
}

void mirror_lcd_set_feedback(MirrorLcdDisplay *display, const char *text) {
    char clean_text[MIRROR_LCD_MAX_FEEDBACK_PAGES * 2U * MIRROR_LCD_LINE_LENGTH + 1U];
    size_t input_index = 0U;
    size_t output_index = 0U;
    size_t page_index;
    size_t line_index;

    if (display == NULL || !display->enabled || text == NULL || text[0] == '\0') {
        return;
    }

    cleanup_feedback_pages(display);
    while (text[input_index] != '\0' && output_index + 1U < sizeof(clean_text)) {
        const unsigned char ch = (unsigned char) text[input_index];
        clean_text[output_index] = (char) (isprint(ch) ? ch : ' ');
        input_index += 1U;
        output_index += 1U;
    }
    clean_text[output_index] = '\0';

    for (page_index = 0U; page_index < MIRROR_LCD_MAX_FEEDBACK_PAGES; ++page_index) {
        for (line_index = 0U; line_index < 2U; ++line_index) {
            const size_t offset =
                ((page_index * 2U) + line_index) * MIRROR_LCD_LINE_LENGTH;
            if (offset >= output_index && page_index > 0U) {
                break;
            }
            copy_lcd_line(
                display->feedback_pages[page_index][line_index],
                sizeof(display->feedback_pages[page_index][line_index]),
                clean_text + offset
            );
        }
        if (page_index == 0U || (page_index * 2U * MIRROR_LCD_LINE_LENGTH) < output_index) {
            display->feedback_page_count += 1U;
        }
        if (((page_index + 1U) * 2U * MIRROR_LCD_LINE_LENGTH) >= output_index) {
            break;
        }
    }

    display->feedback_page_index = 0U;
    display->feedback_active_until_ns =
        now_ns() + (uint64_t) display->feedback_page_count * 2U * LCD_PAGE_INTERVAL_NS;
    display->next_feedback_page_at_ns = now_ns() + LCD_PAGE_INTERVAL_NS;
}

void mirror_lcd_set_text(
    MirrorLcdDisplay *display,
    const char *line_1,
    const char *line_2
) {
    if (display == NULL || !display->enabled) {
        return;
    }

    cleanup_feedback_pages(display);
    copy_lcd_line(display->state_line_1, sizeof(display->state_line_1), line_1);
    copy_lcd_line(display->state_line_2, sizeof(display->state_line_2), line_2);
}

void mirror_lcd_refresh(MirrorLcdDisplay *display) {
    char line_1[MIRROR_LCD_LINE_LENGTH + 1U];
    char line_2[MIRROR_LCD_LINE_LENGTH + 1U];
    const uint64_t current_time_ns = now_ns();

    if (display == NULL || !display->enabled) {
        return;
    }

    if (
        display->feedback_page_count > 1U
        && display->feedback_active_until_ns > current_time_ns
        && current_time_ns >= display->next_feedback_page_at_ns
    ) {
        display->feedback_page_index =
            (display->feedback_page_index + 1U) % display->feedback_page_count;
        display->next_feedback_page_at_ns = current_time_ns + LCD_PAGE_INTERVAL_NS;
    }

    if (
        display->feedback_page_count > 0U
        && display->feedback_active_until_ns > current_time_ns
    ) {
        snprintf(
            line_1,
            sizeof(line_1),
            "%s",
            display->feedback_pages[display->feedback_page_index][0]
        );
        snprintf(
            line_2,
            sizeof(line_2),
            "%s",
            display->feedback_pages[display->feedback_page_index][1]
        );
    } else {
        snprintf(line_1, sizeof(line_1), "%s", display->state_line_1);
        snprintf(line_2, sizeof(line_2), "%s", display->state_line_2);
    }

#ifdef COGNITIVESENSE_HAVE_GPIOD
    write_if_changed(display, line_1, line_2);
#else
    (void) line_1;
    (void) line_2;
#endif
}
