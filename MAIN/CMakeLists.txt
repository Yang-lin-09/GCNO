cmake_minimum_required(VERSION 3.17)

add_executable(main main.cpp)
target_link_libraries(main Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike)
target_include_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/include)
