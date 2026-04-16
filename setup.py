"""Setup script for CFSR-Delta package."""

from setuptools import setup, find_packages

setup(
    name="cfsr-delta",
    version="1.0.0",
    description="Residual Refinement for Lightweight Image Super-Resolution",
    author="Aksha",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
