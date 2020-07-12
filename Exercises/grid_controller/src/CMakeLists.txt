#add_definitions(-ggdb)
add_executable (grid_controller grid_controller.cpp grid_controller_main.cpp)
TARGET_LINK_LIBRARIES(grid_controller ${OpenCV_LIBS})
