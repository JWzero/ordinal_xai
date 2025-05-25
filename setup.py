from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ordinal_xai",
    version="0.1.0",
    author="Jakob Wankmüller",
    author_email="crjakobcr@gmail.com",
    description="A Python package for ordinal regression and model agnosticinterpretation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JWzero/ordinal_xai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "gower>=0.1.0",
        "statsmodels>=0.14.0,<0.15.0",
        "xgboost>=1.7.0,<2.0.0",
        "torch>=2.0.0,<3.0.0",
        "skorch>=0.13.0,<1.0.0",
        "dlordinal>=2.4.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    package_data={
        "ordinal_xai": ["data/*.csv"],
    },
) 