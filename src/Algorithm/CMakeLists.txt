cmake_minimum_required(VERSION 3.17)

set(BGAL_Algorithm_SRC BOC/BOC.cpp)

# Get static lib
add_library(Algorithm STATIC ${BGAL_Algorithm_SRC})

target_include_directories(Algorithm PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
