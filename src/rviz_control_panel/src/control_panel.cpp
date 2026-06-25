#include "rviz_control_panel/control_panel.hpp"

#include <QFrame>
#include <QMetaObject>
#include <QString>
#include <chrono>
#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>

#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

namespace rviz_control_panel
{

ControlPanel::ControlPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  main_layout_ = new QVBoxLayout(this);

  status_label_ = new QLabel("Estado: esperando inicialización...");
  main_layout_->addWidget(status_label_);
}

void ControlPanel::onInitialize()
{
  node_ptr_ = getDisplayContext()->getRosNodeAbstraction().lock();

  if (!node_ptr_) {
    updateStatus("Error: no se pudo obtener el nodo de RViz");
    return;
  }

  node_ = node_ptr_->get_raw_node();

  updateStatus("Estado: panel inicializado");

  named_target_pub_ = node_->create_publisher<std_msgs::msg::String>(NAMED_TARGET_TOPIC, 10);

  // ── Switches ──────────────────────────────────────────────
  addSectionLabel("── Modos ──");

  addLatchedInt32TopicButton(
    "Object Detection",
    "/object_detection_enable",
    1, 0, false);

  //addLatchedInt32TopicButton(
    //"Modo Manual",
   //"/piper/switch_mode",
    //1, 0, false);

  addLatchedInt32TopicButton(
    "Gripper Enable",
    "/gripper_enable",
    1, 0, false);

  // ── Posiciones preestablecidas ────────────────────────────
  addSectionLabel("── Posiciones ──");

  addNamedTargetButton("Zero",          "zero");
  addNamedTargetButton("Home",          "home");
  addNamedTargetButton("Escape",        "escape");
  addNamedTargetButton("Stow",          "stow");
  addNamedTargetButton("Observe Front", "observe_front");
  addNamedTargetButton("Observe Left",  "observe_left");
  addNamedTargetButton("Observe Right", "observe_right");
  addNamedTargetButton("Pre-Grasp",     "pregrasp");

  // ── Gripper ───────────────────────────────────────────────
  addSectionLabel("── Gripper ──");

  addNamedTargetButton("Abrir",  "open");
  addNamedTargetButton("Cerrar", "close");
}

void ControlPanel::addLatchedInt32TopicButton(
  const QString & button_text,
  const std::string & topic_name,
  int checked_value,
  int unchecked_value,
  bool initial_state)
{
  auto button = new QPushButton(this);

  button->setCheckable(true);
  button->setChecked(initial_state);

  button->setText(
    initial_state ?
    button_text + " [ON]" :
    button_text + " [OFF]");

  main_layout_->addWidget(button);

  if (int32_publishers_.find(topic_name) == int32_publishers_.end()) {
    int32_publishers_[topic_name] =
      node_->create_publisher<std_msgs::msg::Int32>(topic_name, 10);
  }

  int32_states_[topic_name] =
    initial_state ? checked_value : unchecked_value;

  if (int32_timers_.find(topic_name) == int32_timers_.end()) {
    int32_timers_[topic_name] =
      node_->create_wall_timer(
        std::chrono::milliseconds(100),
        [this, topic_name]()
        {
          publishInt32Topic(
            topic_name,
            int32_states_[topic_name],
            false);
        });
  }

  publishInt32Topic(
    topic_name,
    int32_states_[topic_name],
    true);

  connect(
    button,
    &QPushButton::toggled,
    this,
    [this, button, button_text, topic_name, checked_value, unchecked_value](bool checked)
    {
      int32_states_[topic_name] =
        checked ? checked_value : unchecked_value;

      button->setText(
        checked ?
        button_text + " [ON]" :
        button_text + " [OFF]");

      publishInt32Topic(
        topic_name,
        int32_states_[topic_name],
        true);
    });
}
void ControlPanel::addSectionLabel(const QString & text)
{
  auto label = new QLabel(text, this);
  label->setAlignment(Qt::AlignCenter);
  main_layout_->addWidget(label);
}

void ControlPanel::addNamedTargetButton(const QString & label, const std::string & target_name)
{
  auto button = new QPushButton(label, this);
  main_layout_->addWidget(button);
  connect(
    button, &QPushButton::clicked, this,
    [this, label, target_name]()
    {
      std_msgs::msg::String msg;
      msg.data = target_name;
      named_target_pub_->publish(msg);
      updateStatus(QString("Enviando: %1").arg(label));
      RCLCPP_INFO(node_->get_logger(), "Named target: %s", target_name.c_str());
    });
}

void ControlPanel::publishInt32Topic(
  const std::string & topic_name,
  int value,
  bool show_status)
{
  auto publisher = int32_publishers_[topic_name];

  std_msgs::msg::Int32 msg;
  msg.data = value;

  publisher->publish(msg);

  if (show_status) {
    updateStatus(
      QString("Publicado en %1: %2")
        .arg(QString::fromStdString(topic_name))
        .arg(value));

    RCLCPP_INFO(
      node_->get_logger(),
      "Publicado en %s: %d",
      topic_name.c_str(),
      value);
  }
}


void ControlPanel::updateStatus(const QString & text)
{
  QMetaObject::invokeMethod(
    status_label_,
    [this, text]() {
      status_label_->setText(text);
    },
    Qt::QueuedConnection);
}

}  // namespace rviz_control_panel

PLUGINLIB_EXPORT_CLASS(rviz_control_panel::ControlPanel, rviz_common::Panel)
