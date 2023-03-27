# Georges' accelerator physics library - Core

[![ci](https://github.com/ULB-Metronu/georges-core/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ULB-Metronu/georges-core/actions/workflows/ci.yml)
[![documentation](https://github.com/ULB-Metronu/georges-core/actions/workflows/documentation.yml/badge.svg?branch=master)](https://github.com/ULB-Metronu/georges-core/actions/workflows/documentation.yml)
![Python](docs/_static/python_versions.svg)
![version](https://img.shields.io/badge/version-2023.2-blue)

[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=ULB-Metronu_georges-core&metric=bugs)](https://sonarcloud.io/summary/new_code?id=ULB-Metronu_georges-core)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=ULB-Metronu_georges-core&metric=coverage)](https://sonarcloud.io/summary/new_code?id=ULB-Metronu_georges-core)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=ULB-Metronu_georges-core&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=ULB-Metronu_georges-core)

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Gitter](https://badges.gitter.im/ULB-Metronu/georges-core.svg)](https://gitter.im/ULB-Metronu/georges-core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


Georges-core is a Python3 library that provides a lot of basic functionalities for other libraries such as Zgoubidoo or Georges. Among the different modules, we have :

* The Kinematics module
* The beam's distribution generator
* The Twiss computation
* The plotting module
* The sequence converter

## Design
The aim of this library is to unify the description of particle accelerator beamlines for different tools
(MAD-X, MAD-NG and BDSIM at this stage) in a unique Python library.

The library design strongly follows conventions and typical uses of the *Pandas* library:
beamlines are naturally described as *Dataframes*.
A functional approach is also one of the design goal of the library, something which fits well with
*Pandas*.

Beamlines are loaded and converted using functions split in packages (one package per beam physics code, *e.g.* MAD-X or BDSIM)
in a unique format (`georges-core.Sequence`). This object serves as an input for other tracking code such as Manzoni or Zgoubidoo.

Twiss computation is also provided especially in coupled Twiss computation.

Support tools are also provided, notably a plotting library (based on *Matplotlib* or *Plotly*)

## Installation

`georges-core` is available from PyPI with pip::

    pip install georges-core

For a custom installation, please read the installation section in the [documentation](https://ulb-metronu.github.io/georges-core/installation.html).
