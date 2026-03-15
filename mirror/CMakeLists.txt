cmake_minimum_required(VERSION 3.16)
project(mirror_capture C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(
    COGNITIVESENSE_REQUIRE_LCD
    "Fail configuration when libgpiod-backed LCD support is unavailable"
    OFF
)

add_library(
    mirror_capture
    camera_capture.c
)

add_executable(
    mirror_capture_smoke
    camera_capture_smoke.c
)

add_executable(
    mirror_frame_streamer
    frame_streamer.c
    lcd_display.c
)

target_include_directories(mirror_capture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mirror_capture_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mirror_frame_streamer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mirror_capture_smoke PRIVATE mirror_capture)

find_path(GPIOD_INCLUDE_DIR gpiod.h)
find_library(GPIOD_LIBRARY gpiod)

if(GPIOD_INCLUDE_DIR AND GPIOD_LIBRARY)
    message(STATUS "mirror_frame_streamer: LCD support enabled with libgpiod")
    message(STATUS "  gpiod.h: ${GPIOD_INCLUDE_DIR}")
    message(STATUS "  libgpiod: ${GPIOD_LIBRARY}")
    target_include_directories(mirror_frame_streamer PRIVATE ${GPIOD_INCLUDE_DIR})
    target_compile_definitions(mirror_frame_streamer PRIVATE COGNITIVESENSE_HAVE_GPIOD=1)
    target_link_libraries(mirror_frame_streamer PRIVATE ${GPIOD_LIBRARY})
elseif(COGNITIVESENSE_REQUIRE_LCD)
    message(FATAL_ERROR "mirror_frame_streamer: LCD support requested, but libgpiod was not found. Install libgpiod-dev and rerun CMake.")
else()
    message(WARNING "mirror_frame_streamer: LCD support disabled because libgpiod was not found. Install libgpiod-dev and configure with -DCOGNITIVESENSE_REQUIRE_LCD=ON to enforce it.")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(mirror_capture PUBLIC pthread)
endif()
