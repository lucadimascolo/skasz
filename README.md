
# SKASZ
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31015/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)

A minimal library for generating mock SKA observations of the SZ effect.

## Installation

> [!WARNING]
> This package was developed using Python 3.10, and there might be some incompatibilities with other versions. This is mostly due to some conflicts between the SKA packages, various external dependencies, and the wheels available for higher Python versions.

First of all, clone this repository on your local machine and move into the cloned directory:
```
git clone https://github.com/lucadimascolo/skasz.git
cd skasz
```

To build the simulation environment, you can use `conda` and install all the required dependencies using the embedded `environment.yml` file:
```
conda env create -f environment.yml [-n custom-env-name]
conda activate custom-env-name
```
If you don't pass any name when creating the environment, it will use the default name `skate` (SKA Testing Environment).

Once the installation is completed, you should be ready to go and fire `skasz` up.

## Example

'''
