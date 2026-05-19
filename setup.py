from setuptools import find_packages, setup

package_name = 'charuco_ros2'

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
    maintainer='zin',
    maintainer_email='maemukiriku@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detect_charuco = charuco_ros2.detect_charuco:main',
            'detect_charuco_9_7 = charuco_ros2.detect_charuco_9_7:main',
            'detect_apriltag = charuco_ros2.detect_apriltag:main',
        ],
    },
)
