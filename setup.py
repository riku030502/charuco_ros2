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
            'multi_cube_charuco_detector = charuco_ros2.multi_cube_charuco_detector:main',
            'generate_charuco_5_7 = charuco_ros2.generate_charuco_5_7:main',
            'detect_charuco = charuco_ros2.detect_charuco:main',
            'detect_charuco_9_7 = charuco_ros2.detect_charuco_9_7:main',
            'detect_apriltag = charuco_ros2.detect_apriltag:main',
            'move_to_charuco = charuco_ros2.move_to_charuco:main',
            'find_cube = charuco_ros2.find_cube:main',
        ],
    },
)
