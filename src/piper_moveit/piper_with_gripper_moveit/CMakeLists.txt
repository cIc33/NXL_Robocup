cmake_minimum_required(VERSION 3.22)
project(piper_with_gripper_moveit)

find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/piper_named_target_executor.py
  scripts/piper_velocity_gui.py
  scripts/piper_velocity_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)
