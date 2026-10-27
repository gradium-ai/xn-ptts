cmake_minimum_required(VERSION 3.16)
project(phonon_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The QNN headers come from the QAIRT SDK; its runtime libraries are loaded at
# run time (dlopen), as is ptts's text front end, so nothing links against them.
set(QAIRT_ROOT "" CACHE PATH "QAIRT SDK root, e.g. .../qairt/2.50.0.260828")
if(NOT QAIRT_ROOT)
  message(FATAL_ERROR "set -DQAIRT_ROOT=<QAIRT SDK root>")
endif()

# The QNN backends start threads inside this process; on glibc before 2.34 that
# needs libpthread linked into the executable.
find_package(Threads REQUIRED)
include(cmake/Json.cmake)

add_library(phonon STATIC ptts_text.cpp qnn_model.cpp phonon.cpp)
target_include_directories(phonon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${QAIRT_ROOT}/include/QNN)
target_link_libraries(phonon PUBLIC ${CMAKE_DL_LIBS} Threads::Threads nlohmann_json::nlohmann_json)

add_executable(phonon_cli main.cpp)
set_target_properties(phonon_cli PROPERTIES OUTPUT_NAME phonon)
target_link_libraries(phonon_cli PRIVATE phonon)
