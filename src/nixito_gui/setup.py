from setuptools import find_packages, setup

package_name = 'nixito_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
       #('share/' + package_name + '/config' , ['config/arm_presets.yaml']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='axelcg_7905',
    maintainer_email='axelguevara7905@gmail.com',
    description='Interfaz gráfica para control del robot Orion',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gui_brazo = nixito_gui.gui_arm:main',
            'test_gui = nixito_gui.test_gui:main',
            'gui_demo = nixito_gui.gui_demo:main',
        ],
    },
)
