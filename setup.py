from setuptools import find_packages, setup
import os
from glob import glob
from setuptools import setup

package_name = 'plant_shooter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'Object_Detection_Files'), glob('Object_Detection_Files/*'),),
        (os.path.join('share', package_name, 'launch'), glob('launch/*'),)
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dan',
    maintainer_email='interwin.daniel@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Spotter = plant_shooter.Spotter:main'
        ],
    },
)
