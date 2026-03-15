cmake_minimum_required(VERSION 3.16)
project(mirror_capture C)

include(FindPkgConfig)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(
    COGNITIVESENSE_REQUIRE_LCD
    "Fail configuration when libgpiod-backed LCD support is unavailable"
    OFF
)

if(COGNITIVESENSE_REQUIRE_LCD)
    message(STATUS "mirror_frame_streamer: LCD support is required for this configure run")
else()
    message(STATUS "mirror_frame_streamer: LCD support is optional for this configure run")
endif()

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

add_executable(
    mirror_lcd_smoke
    lcd_display_smoke.c
    lcd_display.c
)

target_include_directories(mirror_capture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mirror_capture_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mirror_frame_streamer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mirror_lcd_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mirror_capture_smoke PRIVATE mirror_capture)

find_path(GPIOD_INCLUDE_DIR gpiod.h)
find_library(GPIOD_LIBRARY gpiod)

if((NOT GPIOD_INCLUDE_DIR OR NOT GPIOD_LIBRARY) AND PKG_CONFIG_FOUND)
    pkg_check_modules(GPIOD_PC QUIET libgpiod)
    if(GPIOD_PC_FOUND)
        if(NOT GPIOD_INCLUDE_DIR AND GPIOD_PC_INCLUDE_DIRS)
            list(GET GPIOD_PC_INCLUDE_DIRS 0 GPIOD_INCLUDE_DIR)
        endif()
        if(NOT GPIOD_LIBRARY AND GPIOD_PC_LINK_LIBRARIES)
            list(GET GPIOD_PC_LINK_LIBRARIES 0 GPIOD_LIBRARY)
        elseif(NOT GPIOD_LIBRARY AND GPIOD_PC_LIBRARY_DIRS)
            find_library(
                GPIOD_LIBRARY
                NAMES gpiod
                PATHS ${GPIOD_PC_LIBRARY_DIRS}
                NO_DEFAULT_PATH
            )
        endif()
    endif()
endif()

if(GPIOD_INCLUDE_DIR AND GPIOD_LIBRARY)
    file(READ "${GPIOD_INCLUDE_DIR}/gpiod.h" COGNITIVESENSE_GPIOD_HEADER)
    string(FIND "${COGNITIVESENSE_GPIOD_HEADER}" "gpiod_chip_request_lines(" COGNITIVESENSE_GPIOD_V2_POS)

    message(STATUS "mirror_frame_streamer: LCD support enabled with libgpiod")
    message(STATUS "  gpiod.h: ${GPIOD_INCLUDE_DIR}")
    message(STATUS "  libgpiod: ${GPIOD_LIBRARY}")
    target_include_directories(mirror_frame_streamer PRIVATE ${GPIOD_INCLUDE_DIR})
    target_include_directories(mirror_lcd_smoke PRIVATE ${GPIOD_INCLUDE_DIR})
    target_compile_definitions(mirror_frame_streamer PRIVATE COGNITIVESENSE_HAVE_GPIOD=1)
    target_compile_definitions(mirror_lcd_smoke PRIVATE COGNITIVESENSE_HAVE_GPIOD=1)
    if(NOT COGNITIVESENSE_GPIOD_V2_POS EQUAL -1)
        message(STATUS "  libgpiod API: v2")
        target_compile_definitions(mirror_frame_streamer PRIVATE COGNITIVESENSE_GPIOD_V2=1)
        target_compile_definitions(mirror_lcd_smoke PRIVATE COGNITIVESENSE_GPIOD_V2=1)
    else()
        message(STATUS "  libgpiod API: v1")
    endif()
    target_link_libraries(mirror_frame_streamer PRIVATE ${GPIOD_LIBRARY})
    target_link_libraries(mirror_lcd_smoke PRIVATE ${GPIOD_LIBRARY})
elseif(COGNITIVESENSE_REQUIRE_LCD)
    message(FATAL_ERROR "mirror_frame_streamer: LCD support requested, but libgpiod was not found. Install libgpiod-dev and rerun CMake.")
else()
    message(WARNING "mirror_frame_streamer: LCD support disabled because libgpiod was not found. Install libgpiod-dev and configure with -DCOGNITIVESENSE_REQUIRE_LCD=ON to enforce it.")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(mirror_capture PUBLIC pthread)
endif()
