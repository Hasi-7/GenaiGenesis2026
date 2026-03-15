#ifndef COGNITIVESENSE_LCD_DISPLAY_H
#define COGNITIVESENSE_LCD_DISPLAY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MIRROR_LCD_LINE_LENGTH 16U
#define MIRROR_LCD_MAX_FEEDBACK_PAGES 8U

typedef struct MirrorLcdDisplay {
    bool requested;
    bool enabled;
    bool available;
    bool warned_missing_support;
    char chip_path[64];
    unsigned int rs_pin;
    unsigned int e_pin;
    unsigned int d4_pin;
    unsigned int d5_pin;
    unsigned int d6_pin;
    unsigned int d7_pin;
    void *chip;
    void *rs_line;
    void *e_line;
    void *d4_line;
    void *d5_line;
    void *d6_line;
    void *d7_line;
    char state_line_1[MIRROR_LCD_LINE_LENGTH + 1U];
    char state_line_2[MIRROR_LCD_LINE_LENGTH + 1U];
    char rendered_line_1[MIRROR_LCD_LINE_LENGTH + 1U];
    char rendered_line_2[MIRROR_LCD_LINE_LENGTH + 1U];
    char feedback_pages[MIRROR_LCD_MAX_FEEDBACK_PAGES][2][MIRROR_LCD_LINE_LENGTH + 1U];
    size_t feedback_page_count;
    size_t feedback_page_index;
    uint64_t feedback_active_until_ns;
    uint64_t next_feedback_page_at_ns;
} MirrorLcdDisplay;

void mirror_lcd_init(MirrorLcdDisplay *display);
void mirror_lcd_close(MirrorLcdDisplay *display);
void mirror_lcd_set_connecting(
    MirrorLcdDisplay *display,
    const char *server_host,
    uint16_t port
);
void mirror_lcd_set_streaming(
    MirrorLcdDisplay *display,
    uint32_t width,
    uint32_t height,
    uint32_t fps
);
void mirror_lcd_set_state(
    MirrorLcdDisplay *display,
    const char *state_text,
    const char *detail_text
);
void mirror_lcd_set_feedback(MirrorLcdDisplay *display, const char *text);
void mirror_lcd_set_text(
    MirrorLcdDisplay *display,
    const char *line_1,
    const char *line_2
);
void mirror_lcd_refresh(MirrorLcdDisplay *display);

#endif
