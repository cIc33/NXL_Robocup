from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'piper_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='axelcg_7905',
    maintainer_email='axelguevara7905@gmail.com',
    description='Perception package for detecting an emergency stop button.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'button_detector_node = piper_perception.button_detector_node:main',
        ],
    },
)
