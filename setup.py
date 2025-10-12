"""
Setup script for Multi-Modal Recommendation Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="multimodal-recommendation-platform",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-Modal Content Recommendation & Advertiser Analytics Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-recommendation-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
        "viz": [
            "plotly>=5.15.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rec-train=src.training.trainer:main",
            "rec-server=src.api.server:main",
            "rec-build-index=src.inference.vector_search:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
