cmake_minimum_required(VERSION 3.8)
project(rviz_control_panel)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(pluginlib REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)

set(CMAKE_AUTOMOC ON)

qt5_wrap_cpp(MOC_FILES
  include/rviz_control_panel/control_panel.hpp
)

add_library(control_panel SHARED
  src/control_panel.cpp
  ${MOC_FILES}
)

target_include_directories(control_panel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(control_panel
  rclcpp
  rviz_common
  pluginlib
  std_srvs
  std_msgs
)

install(
  TARGETS control_panel
  EXPORT export_rviz_control_panel
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_export_include_directories(include)
ament_export_targets(export_rviz_control_panel)


ament_package()