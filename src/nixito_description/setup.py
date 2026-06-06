from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'nixito_description'
mesh_files = [
    path for path in glob(os.path.join('meshes', '**', '*'), recursive=True)
    if os.path.isfile(path)
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Incluir archivos URDF
        (os.path.join('share', package_name, 'urdf'), 
         glob(os.path.join('urdf', '*'))),
        # Incluir Meshes
        (os.path.join('share', package_name, 'meshes'), 
         mesh_files),
        # Incluir configuraciones de RViz
        (os.path.join('share', package_name, 'rviz'),
         glob(os.path.join('rviz', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='axelcg_7905',
    maintainer_email='axelguevara7905@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
