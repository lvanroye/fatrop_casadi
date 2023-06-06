add_library(fatrop_casadi fatrop_casadi.cpp)
target_link_libraries(fatrop_casadi casadi fatrop)
add_executable(example example.cpp)
target_link_libraries(example fatrop_casadi)