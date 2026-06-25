#ifndef RVIZ_CONTROL_PANEL__CONTROL_PANEL_HPP_
#define RVIZ_CONTROL_PANEL__CONTROL_PANEL_HPP_

#include <map>
#include <memory>
#include <string>

#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

namespace rviz_control_panel
{

class ControlPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit ControlPanel(QWidget * parent = nullptr);
  ~ControlPanel() override = default;

  void onInitialize() override;

private:
  void addLatchedInt32TopicButton(
    const QString & button_text,
    const std::string & topic_name,
    int checked_value,
    int unchecked_value,
    bool initial_state = false);

  void addNamedTargetButton(
    const QString & label,
    const std::string & target_name);

  void addSectionLabel(const QString & text);

  void publishInt32Topic(
    const std::string & topic_name,
    int value,
    bool show_status = true);

  void updateStatus(const QString & text);

  QVBoxLayout * main_layout_;
  QLabel * status_label_;

  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr_;
  rclcpp::Node::SharedPtr node_;

  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> int32_publishers_;
  std::map<std::string, int> int32_states_;
  std::map<std::string, rclcpp::TimerBase::SharedPtr> int32_timers_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr named_target_pub_;
  static constexpr const char * NAMED_TARGET_TOPIC = "/piper/named_target";
};

}  // namespace rviz_control_panel

#endif  // RVIZ_CONTROL_PANEL__CONTROL_PANEL_HPP_
