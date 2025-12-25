cmake_minimum_required(VERSION 3.16)

project(newton_chrono_gt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Chrono REQUIRED COMPONENTS Vehicle)

add_executable(newton_chrono_gt newton_chrono_gt.cpp)

target_link_libraries(newton_chrono_gt PRIVATE Chrono::Chrono_vehicle Chrono::Chrono_core)

# Prefer a runnable local binary without needing to set LD_LIBRARY_PATH manually.
if(DEFINED CHRONO_PACKAGE_PREFIX_DIR)
  set_target_properties(newton_chrono_gt PROPERTIES BUILD_RPATH "${CHRONO_PACKAGE_PREFIX_DIR}/lib")
endif()

