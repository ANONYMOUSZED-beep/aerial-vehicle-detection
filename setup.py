"""
Setup script for Aerial Vehicle Detection System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="aerial-vehicle-detection",
    version="1.0.0",
    author="Aerial VH Detection Team", 
    author_email="contact@aerialvh.com",
    description="A comprehensive deep learning system for detecting vehicles from aerial images using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aerial-vehicle-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0", 
            "flake8>=6.0.0",
            "pytest-cov>=4.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "aerial-detect=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    keywords=[
        "computer vision",
        "object detection", 
        "aerial imagery",
        "drone detection",
        "vehicle detection",
        "yolov8",
        "pytorch",
        "deep learning",
        "machine learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/aerial-vehicle-detection/issues",
        "Source": "https://github.com/yourusername/aerial-vehicle-detection",
        "Documentation": "https://github.com/yourusername/aerial-vehicle-detection/wiki",
    },
)