cmake_minimum_required(VERSION 3.14)
project(Wavec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(wavec src/ft_wav.cc)
target_include_directories(wavec PRIVATE src)
target_link_libraries(wavec Threads::Threads)

add_executable(sim src/sim.cc)
add_executable(kmeans src/kmeans.cc)
