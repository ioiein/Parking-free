from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="detector",
    packages=find_packages(),
    version="0.1.0",
    description="YOLOv5 detector for detection cars on parking",
    author="ioiein",
    entry_points={
        "console_scripts": [
            "detector = src.detector:detect_command"
        ]
    },
    install_requires=required,
    license="MIT",
)
