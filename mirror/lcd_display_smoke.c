#define _POSIX_C_SOURCE 200809L

#include "lcd_display.h"

#include <stdio.h>
#include <time.h>

static void sleep_ms(long milliseconds) {
    struct timespec delay;

    delay.tv_sec = milliseconds / 1000L;
    delay.tv_nsec = (milliseconds % 1000L) * 1000000L;
    nanosleep(&delay, NULL);
}

int main(void) {
    MirrorLcdDisplay display;

    mirror_lcd_init(&display);
    if (!display.enabled) {
        fprintf(
            stderr,
            "LCD smoke test failed: display was not initialized. Check the stderr messages above.\n"
        );
        return 1;
    }

    mirror_lcd_set_text(&display, "LCD init OK", "Direct test mode");
    mirror_lcd_refresh(&display);
    fprintf(stderr, "LCD smoke test wrote the direct text message.\n");
    sleep_ms(4000L);

    mirror_lcd_set_feedback(&display, "Feedback smoke test is active.");
    mirror_lcd_refresh(&display);
    fprintf(stderr, "LCD smoke test wrote the feedback message.\n");
    sleep_ms(6000L);

    mirror_lcd_close(&display);
    return 0;
}
