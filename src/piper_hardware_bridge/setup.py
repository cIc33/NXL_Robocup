from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'piper_hardware_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Bridge nodes for AgileX Piper hardware control.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'piper_joint_state_normalizer = piper_hardware_bridge.joint_state_normalizer:main',
            'piper_hardware_velocity_bridge = piper_hardware_bridge.hardware_velocity_bridge:main',
            'piper_follow_joint_trajectory_bridge = piper_hardware_bridge.follow_joint_trajectory_bridge:main',
            'piper_joint_state_reader = piper_hardware_bridge.joint_state_reader:main',
        ],
    },
)
