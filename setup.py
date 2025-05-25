from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ordinal_xai",
    version="0.1.11",
    author="Jakob Wankmüller",
    author_email="crjakobcr@gmail.com",
    description="A Python package for ordinal regression and model-agnostic interpretation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JWzero/ordinal_xai",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ordinal_xai=ordinal_xai.__main__:main',
        ],
    },
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
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "gower",
        "statsmodels",
        "xgboost",
        "torch",
        "skorch",
        "dlordinal",
    ],
    include_package_data=True,
    package_data={
        "ordinal_xai": ["data/*.csv"],
    },
) 