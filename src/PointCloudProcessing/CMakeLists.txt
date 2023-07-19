cmake_minimum_required(VERSION 3.17)

set(BGAL_PointCloudProcessing_SRC
        Registration/ICP/ICP.cpp
        )

# Get static lib
add_library(PointCloudProcessing STATIC ${BGAL_PointCloudProcessing_SRC})
target_link_libraries(PointCloudProcessing Algorithm BaseShape ${Boost_LIBRARIES})

target_include_directories(PointCloudProcessing PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
