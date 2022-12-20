****************************************
Welcome to Georges-core's documentation!
****************************************

|Actions Status| |Documentation Status| |Python version| |version| |Bugs| |Coverage| |Reliability|
|License| |Code Style| |Gitter|

Introduction
############
Georges-core is an open source project hosted on Github under the GNU General Public Licence (GPL) version 3. It is a Python3 library that provides a lot of basic functionalities for other libraries such as Zgoubidoo or Georges. Among the different modules, we have :

* The Kinematics module
* The beam's distribution generator
* The Twiss computation
* The plotting module
* The sequence converter

Georges-core's documentation
============================

The documentation is part of the Georges-core repository itself and is made available *via* `Github Pages <https://pages.github.com>`_ .
It is hosted at `ulb-metronu.github.io/georges-core/ <https://ulb-metronu.github.io/georges-core/>`_

You can take a look at the :doc:`Examples <examples>`.
We value your contributions and you can follow the instructions in :doc:`Contributing <contributing>`.
Finally, if you’re having problems, please do let us know at our :doc:`Support <support>` page.

..  toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: Developers
    :glob:

    authors
    support
    contributing
    changelog

..  toctree::
    :maxdepth: 3
    :titlesonly:
    :caption: User Guide
    :glob:

    installation
    modules
    examples

..  toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: API Reference
    :glob:

    api/modules



Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Actions Status| image:: https://github.com/ULB-Metronu/georges-core/actions/workflows/ci.yml/badge.svg?branch=master
   :target: https://github.com/ULB-Metronu/georges-core/actions

.. |Documentation Status| image:: https://github.com/ULB-Metronu/georges-core/actions/workflows/documentation.yml/badge.svg?branch=master
   :target: https://github.com/ULB-Metronu/georges-core/actions

.. |Python version| image:: _static/python_versions.svg

.. |version| image:: https://img.shields.io/badge/version-2022.1-blue

.. |Bugs| image:: https://sonarcloud.io/api/project_badges/measure?project=ULB-Metronu_georges-core&metric=bugs
   :target: https://sonarcloud.io/summary/new_code?id=ULB-Metronu_georges-core

.. |Coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=ULB-Metronu_georges-core&metric=coverage
   :target: https://sonarcloud.io/summary/new_code?id=ULB-Metronu_georges-core

.. |Reliability| image:: https://sonarcloud.io/api/project_badges/measure?project=ULB-Metronu_georges-core&metric=reliability_rating
   :target: https://sonarcloud.io/summary/new_code?id=ULB-Metronu_georges-core

.. |License| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

.. |Code Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

.. |Gitter| image:: https://badges.gitter.im/ULB-Metronu/georges-core.svg?
   :target: https://gitter.im/ULB-Metronu/georges-core?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
