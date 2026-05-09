#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class PressEstopNode : public rclcpp::Node
{
public:
  PressEstopNode()
  : Node("press_estop_node")
  {
    sensor_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    declare_parameter<std::string>("planning_group", "arm");
    declare_parameter<std::string>("pose_topic", "/button_pose");
    declare_parameter<std::string>("depth_topic", "/button_depth_z");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<bool>("use_depth_gate", true);
    declare_parameter<double>("approach_distance", 0.08);
    declare_parameter<double>("press_distance", 0.006);
    declare_parameter<double>("retreat_distance", 0.08);
    declare_parameter<double>("pose_timeout", 1.0);
    declare_parameter<double>("depth_timeout", 0.5);
    declare_parameter<double>("z_stop_visual", 0.10);
    declare_parameter<double>("depth_approach_step", 0.002);
    declare_parameter<int>("max_visual_approach_steps", 10);
    declare_parameter<double>("depth_settle_time", 0.2);
    declare_parameter<double>("velocity_scaling", 0.2);
    declare_parameter<double>("acceleration_scaling", 0.2);
    declare_parameter<double>("cartesian_eef_step", 0.005);
    declare_parameter<double>("jump_threshold", 0.0);
    declare_parameter<double>("cartesian_fraction_threshold", 0.95);
    declare_parameter<std::string>("approach_axis", "z");
    declare_parameter<double>("press_axis_sign", 1.0);

    planning_group_ = get_parameter("planning_group").as_string();
    pose_topic_ = get_parameter("pose_topic").as_string();
    depth_topic_ = get_parameter("depth_topic").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    use_depth_gate_ = get_parameter("use_depth_gate").as_bool();
    approach_distance_ = get_parameter("approach_distance").as_double();
    press_distance_ = get_parameter("press_distance").as_double();
    retreat_distance_ = get_parameter("retreat_distance").as_double();
    pose_timeout_ = get_parameter("pose_timeout").as_double();
    depth_timeout_ = get_parameter("depth_timeout").as_double();
    z_stop_visual_ = get_parameter("z_stop_visual").as_double();
    depth_approach_step_ = get_parameter("depth_approach_step").as_double();
    max_visual_approach_steps_ = get_parameter("max_visual_approach_steps").as_int();
    depth_settle_time_ = get_parameter("depth_settle_time").as_double();
    velocity_scaling_ = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    cartesian_eef_step_ = get_parameter("cartesian_eef_step").as_double();
    jump_threshold_ = get_parameter("jump_threshold").as_double();
    cartesian_fraction_threshold_ = get_parameter("cartesian_fraction_threshold").as_double();
    approach_axis_ = get_parameter("approach_axis").as_string();
    press_axis_sign_ = get_parameter("press_axis_sign").as_double();

    rclcpp::SubscriptionOptions pose_options;
    pose_options.callback_group = sensor_callback_group_;
    button_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      pose_topic_, 10,
      std::bind(&PressEstopNode::buttonPoseCallback, this, std::placeholders::_1),
      pose_options);

    rclcpp::SubscriptionOptions depth_options;
    depth_options.callback_group = sensor_callback_group_;
    button_depth_sub_ = create_subscription<std_msgs::msg::Float32>(
      depth_topic_, 10,
      std::bind(&PressEstopNode::buttonDepthCallback, this, std::placeholders::_1),
      depth_options);

    pre_press_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/estop_pre_press_pose", 10);
    press_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/estop_press_pose", 10);
    retreat_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/estop_retreat_pose", 10);

    trigger_srv_ = create_service<std_srvs::srv::Trigger>(
      "/start_estop_press",
      std::bind(&PressEstopNode::startPressCallback, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default,
      service_callback_group_);
  }

  void initializeMoveGroup()
  {
    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);
    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    RCLCPP_INFO(get_logger(), "Press tasking node ready for group '%s'", planning_group_.c_str());
  }

private:
  void buttonPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    latest_button_pose_ = *msg;
  }

  void buttonDepthCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    latest_button_depth_ = msg->data;
    latest_button_depth_time_ = now();
  }

  void startPressCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!move_group_) {
      response->success = false;
      response->message = "MoveGroupInterface is not initialized";
      return;
    }

    auto maybe_pose = getLatestButtonPose();
    if (!maybe_pose.has_value()) {
      response->success = false;
      response->message = "No recent button pose available";
      return;
    }

    const auto & contact_pose = maybe_pose.value();
    geometry_msgs::msg::PoseStamped pre_press_pose = offsetPose(contact_pose, -approach_distance_);
    geometry_msgs::msg::PoseStamped retreat_pose = offsetPose(contact_pose, -retreat_distance_);

    pre_press_pub_->publish(pre_press_pose);
    retreat_pub_->publish(retreat_pose);

    std::string error_message;
    if (!planAndExecuteToPose(pre_press_pose, error_message)) {
      response->success = false;
      response->message = "Failed to reach pre-press pose: " + error_message;
      return;
    }

    geometry_msgs::msg::PoseStamped gate_pose = move_group_->getCurrentPose();
    if (use_depth_gate_) {
      if (!advanceUntilDepthGate(gate_pose, error_message)) {
        response->success = false;
        response->message = "Failed during depth-gated approach: " + error_message;
        return;
      }
    }

    geometry_msgs::msg::PoseStamped press_pose = offsetPose(gate_pose, press_distance_);
    press_pub_->publish(press_pose);

    if (!executeCartesianTarget(press_pose, error_message)) {
      response->success = false;
      response->message = "Failed to execute press motion: " + error_message;
      return;
    }

    if (!executeCartesianTarget(retreat_pose, error_message)) {
      response->success = false;
      response->message = "Failed to execute retreat motion: " + error_message;
      return;
    }

    response->success = true;
    response->message = "Emergency stop press sequence completed";
  }

  std::optional<geometry_msgs::msg::PoseStamped> getLatestButtonPose()
  {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    if (!latest_button_pose_.has_value()) {
      return std::nullopt;
    }

    const auto age = now() - latest_button_pose_->header.stamp;
    if (age.seconds() > pose_timeout_) {
      return std::nullopt;
    }

    return latest_button_pose_;
  }

  std::optional<double> getLatestButtonDepth()
  {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    if (!latest_button_depth_.has_value()) {
      return std::nullopt;
    }

    const auto age = now() - latest_button_depth_time_;
    if (age.seconds() > depth_timeout_) {
      return std::nullopt;
    }

    return latest_button_depth_;
  }

  bool advanceUntilDepthGate(
    geometry_msgs::msg::PoseStamped & gate_pose,
    std::string & error_message)
  {
    for (int step = 0; step < max_visual_approach_steps_; ++step) {
      auto maybe_depth = getLatestButtonDepth();
      if (!maybe_depth.has_value()) {
        error_message = "No recent button depth available";
        return false;
      }

      if (maybe_depth.value() <= z_stop_visual_) {
        gate_pose = move_group_->getCurrentPose();
        return true;
      }

      const auto next_pose = offsetPose(move_group_->getCurrentPose(), depth_approach_step_);
      if (!executeCartesianTarget(next_pose, error_message)) {
        return false;
      }

      rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(depth_settle_time_)));
    }

    auto maybe_depth = getLatestButtonDepth();
    if (maybe_depth.has_value()) {
      error_message = "Depth gate not reached after " + std::to_string(max_visual_approach_steps_) +
        " steps, latest depth=" + std::to_string(maybe_depth.value());
    } else {
      error_message = "Depth gate not reached and latest depth is unavailable";
    }
    return false;
  }

  geometry_msgs::msg::PoseStamped offsetPose(
    const geometry_msgs::msg::PoseStamped & input_pose,
    double distance) const
  {
    geometry_msgs::msg::PoseStamped output_pose = input_pose;
    const auto axis = getApproachAxis(input_pose.pose.orientation);

    output_pose.pose.position.x += axis[0] * distance;
    output_pose.pose.position.y += axis[1] * distance;
    output_pose.pose.position.z += axis[2] * distance;

    return output_pose;
  }

  std::array<double, 3> getApproachAxis(const geometry_msgs::msg::Quaternion & orientation) const
  {
    tf2::Quaternion quat;
    tf2::fromMsg(orientation, quat);

    tf2::Matrix3x3 rotation(quat);
    tf2::Vector3 local_axis(0.0, 0.0, 1.0);
    if (approach_axis_ == "x") {
      local_axis = tf2::Vector3(1.0, 0.0, 0.0);
    } else if (approach_axis_ == "y") {
      local_axis = tf2::Vector3(0.0, 1.0, 0.0);
    }

    tf2::Vector3 world_axis = rotation * local_axis;
    world_axis *= press_axis_sign_;
    world_axis.normalize();

    return {world_axis.x(), world_axis.y(), world_axis.z()};
  }

  bool planAndExecuteToPose(
    const geometry_msgs::msg::PoseStamped & target_pose,
    std::string & error_message)
  {
    move_group_->setPoseTarget(target_pose.pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool planned = static_cast<bool>(move_group_->plan(plan));
    move_group_->clearPoseTargets();

    if (!planned) {
      error_message = "planning failed";
      return false;
    }

    const auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      error_message = "plan execution failed";
      return false;
    }

    return true;
  }

  bool executeCartesianTarget(
    const geometry_msgs::msg::PoseStamped & target_pose,
    std::string & error_message)
  {
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose.pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = move_group_->computeCartesianPath(
      waypoints, cartesian_eef_step_, jump_threshold_, trajectory, false);

    if (fraction < cartesian_fraction_threshold_) {
      error_message = "cartesian path fraction too low: " + std::to_string(fraction);
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan cartesian_plan;
    cartesian_plan.trajectory_ = trajectory;

    const auto result = move_group_->execute(cartesian_plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      error_message = "cartesian execution failed";
      return false;
    }

    return true;
  }

  std::string planning_group_;
  std::string pose_topic_;
  std::string depth_topic_;
  std::string base_frame_;
  bool use_depth_gate_;
  double approach_distance_;
  double press_distance_;
  double retreat_distance_;
  double pose_timeout_;
  double depth_timeout_;
  double z_stop_visual_;
  double depth_approach_step_;
  int max_visual_approach_steps_;
  double depth_settle_time_;
  double velocity_scaling_;
  double acceleration_scaling_;
  double cartesian_eef_step_;
  double jump_threshold_;
  double cartesian_fraction_threshold_;
  std::string approach_axis_;
  double press_axis_sign_;

  std::mutex pose_mutex_;
  std::mutex depth_mutex_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_button_pose_;
  std::optional<double> latest_button_depth_;
  rclcpp::Time latest_button_depth_time_{0, 0, RCL_ROS_TIME};
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  rclcpp::CallbackGroup::SharedPtr sensor_callback_group_;
  rclcpp::CallbackGroup::SharedPtr service_callback_group_;

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr button_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr button_depth_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_srv_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pre_press_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr press_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr retreat_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PressEstopNode>();
  node->initializeMoveGroup();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
