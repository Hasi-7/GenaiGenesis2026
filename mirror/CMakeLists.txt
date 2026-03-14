cmake_minimum_required(VERSION 3.16)
project(mirror_capture C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(MIRROR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(MIRROR_CAMERA_CAPTURE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_capture.c)
set(MIRROR_CAMERA_CAPTURE_SMOKE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_capture_smoke.c)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/camera_capture.c)
    set(MIRROR_CAMERA_CAPTURE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/camera_capture.c)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/camera_capture_smoke.c)
    set(MIRROR_CAMERA_CAPTURE_SMOKE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/camera_capture_smoke.c)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/camera_capture.h)
    set(MIRROR_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_library(
    mirror_capture
    ${MIRROR_CAMERA_CAPTURE_SOURCE}
)

add_executable(
    mirror_capture_smoke
    ${MIRROR_CAMERA_CAPTURE_SMOKE_SOURCE}
)

target_include_directories(mirror_capture PUBLIC ${MIRROR_INCLUDE_DIR})
target_include_directories(mirror_capture_smoke PRIVATE ${MIRROR_INCLUDE_DIR})
target_link_libraries(mirror_capture_smoke PRIVATE mirror_capture)

if(UNIX AND NOT APPLE)
    target_link_libraries(mirror_capture PUBLIC pthread)
endif()
