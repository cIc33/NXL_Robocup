from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nixito_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Incluir launch files
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('nixito_bringup', '*.launch.py'))),
        
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='axelcg_7905',
    maintainer_email='axelguevara7905@gmail.com',
    description='Launch files y configuración del robot Orion',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            
        ],
    },
)
