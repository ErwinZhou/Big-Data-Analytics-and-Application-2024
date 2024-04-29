cmake_minimum_required(VERSION 3.20)
project(PageRank)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_executable(${PROJECT_NAME} main.cpp
        src/preprocess.cpp
        src/pagerank.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
