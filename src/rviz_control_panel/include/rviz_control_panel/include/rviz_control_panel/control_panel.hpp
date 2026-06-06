#ifndef RVIZ_CONTROL_PANEL__CONTROL_PANEL_HPP_
#define RVIZ_CONTROL_PANEL__CONTROL_PANEL_HPP_

#include <memory>

#include <QLabel>
#include <QPushButton>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace rviz_control_panel
{

class ControlPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit ControlPanel(QWidget * parent = nullptr);
  ~ControlPanel() override = default;

  void onInitialize() override;

private Q_SLOTS:
  void onEnableClicked();
  void onDisableClicked();

private:
  void callSetBoolService(bool value);

  QLabel * status_label_;
  QPushButton * enable_button_;
  QPushButton * disable_button_;

  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr> int32_publishers_;
  rclcpp::Node::SharedPtr node_;

  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr set_bool_client_;

  void addLatchedInt32TopicButton(
  const QString & button_text,
  const std::string & topic_name,
  int checked_value,
  int unchecked_value,
  bool initial_state = false);

  void publishInt32Topic(
    const std::string & topic_name,
    int value);
};

}  // namespace rviz_control_panel

#endif  // RVIZ_CONTROL_PANEL__CONTROL_PANEL_HPP_