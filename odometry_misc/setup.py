import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'odometry_misc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + "/config",
            glob('config/*.yaml')),
        ('share/' + package_name + "/trained_models",
            glob('trained_models/*.pth')),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.launch')))
    ],
    
    scripts=[
    ],
    install_requires=['setuptools', 
                      'launch'],
    zip_safe=True,
    maintainer='valerie',
    maintainer_email='v.bartel.vb@gmail.com',
    description='This package contains scripts and nodes for generating and processing odometry data and training and testing a model.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'measure_odometry = odometry_misc.measure_odometry:main',
            'model_odometry = odometry_misc.model_odometry:main',
            'measure_odometry_real_world = odometry_misc.measure_odometry_real_world:main',
        ],
    },
)
