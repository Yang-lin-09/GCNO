cmake_minimum_required(VERSION 3.17)

project(NormalOrientation)

# Default Build_type RELEASE
set(CMAKE_BUILD_TYPE RELEASE)

# CXX STANDARD
set(CMAKE_CXX_STANDARD 17)

# Set lib output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)

# include header from project_source_dir
include_directories(${PROJECT_SOURCE_DIR}/include)

# Get Eigen3
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "${EIGEN3_VERSION_STRING}")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif ()

# Get Boost
find_package(Boost REQUIRED)
if (Boost_FOUND)
    message(STATUS "BOOST FOUNDED")
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

# Get CGAL
find_package(CGAL REQUIRED)
if (CGAL_FOUND)
    include(${CGAL_USE_FILE})
else ()
    message("ERROR: this program requires CGAL and will not be compiled.")
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# sub_directory
add_subdirectory(include)
add_subdirectory(src)

#target_link_libraries(${PROJECT_NAME}  Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D ${Boost_LIBRARIES})
add_subdirectory(MAIN)

install(TARGETS Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike
	EXPORT ${PROJECT_NAME}Targets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY ./include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")

configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)

install(EXPORT ${PROJECT_NAME}Targets
	FILE ${PROJECT_NAME}-config.cmake
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION lib/cmake/${PROJECT_NAME}
	)
