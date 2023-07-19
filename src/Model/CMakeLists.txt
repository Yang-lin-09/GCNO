cmake_minimum_required(VERSION 3.17)

set(BGAL_Model_SRC
        Model.cpp
		ManifoldModel.cpp
        Model_Iterator.cpp
        )

# Get static lib
add_library(Model STATIC ${BGAL_Model_SRC})
target_link_libraries(Model Algorithm BaseShape PQP ${Boost_LIBRARIES})

target_include_directories(Model PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
