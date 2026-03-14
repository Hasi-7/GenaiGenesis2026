cmake_minimum_required(VERSION 3.16)
project(mirror_capture C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(
    mirror_capture
    src/camera_capture.c
)

add_executable(
    mirror_capture_smoke
    src/camera_capture_smoke.c
)

target_include_directories(mirror_capture PUBLIC include)
target_include_directories(mirror_capture_smoke PRIVATE include)
target_link_libraries(mirror_capture_smoke PRIVATE mirror_capture)

if(UNIX AND NOT APPLE)
    target_link_libraries(mirror_capture PUBLIC pthread)
endif()
