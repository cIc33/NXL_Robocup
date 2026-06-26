from setuptools import find_packages, setup

package_name = 'nixito_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'arm_camera = nixito_perception.test_cam1:main',
            'vision = nixito_perception.vision:main',
            'vision_maze = nixito_perception.vision_maze:main',
            'gopro = nixito_perception.gopro:main',
            'thermal = nixito_perception.thermal_topdon:main',
            'paro = nixito_perception.centro_paro:main'
        ],
    },
)
