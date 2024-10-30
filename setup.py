from setuptools import setup, find_packages

setup(
    name="normalflow",
    version="0.1.0",
    description="NormalFlow: Fast, Robust, and Accurate Contact-based Object 6DoF Pose Tracking with Vision-based Tactile Sensors",
    author="Hung-Jui Huang",
    author_email="hungjuih@andrew.cmu.edu",
    packages=find_packages(),
    install_requires=[
        "pillow==10.0.0",
        "numpy==1.26.4",
        "opencv-python>=4.9.0",
        "scipy>=1.13.1",
        "PyYaml>=6.0.1",
    ],
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'realtime_object_tracking=demos.realtime_object_tracking:realtime_object_tracking',
            'test_tracking=examples.test_tracking:test_tracking',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
