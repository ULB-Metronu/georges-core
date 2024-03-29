************
Installation
************

You can install :code:`georges-core` from PyPI with pip::

    pip install georges-core

For development purposes you can copy the source code on your computer. Georges-core is hosted on Github and can be downloaded using the following::

    git clone https://github.com/ULB-Metronu/georges-core.git

You can  stay on the bleeding-edge master branch or you can checkout
a release tag::

    git checkout tags/2023.3

The installation process to get George’s core running is relatively simple: the whole library is ready to
be installed with `Poetry <https://python-poetry.org/>`_. However, depending your configuration,
you can use a `conda` environment based on the
`Intel distribution  <https://software.intel.com/en-us/distribution-for-python>`_.

Dependencies
############

A coherent set of dependencies is listed in the `pyproject.toml` file (section tool.poetry.dependencies)
A typical user should not worry about those dependencies: they are correctly managed either with poetry

Installation using Poetry
#########################

Assuming you have Poetry and Python installed on your system, go to the location of the library and simply use
these commands::

    cd path/to/georges-core
    poetry install --without dev,docs

.. note::

    George’s-core uses python version >=3.9 and < 3.11

Georges-core can be subsequently updated by running the following::

    git pull origin master
    poetry update

.. note::

    You can install a independent python environment with :code:`pyenv` (https://github.com/pyenv/pyenv) and
    :code:`pyenv-virtualenv` (https://github.com/pyenv/pyenv-virtualenv) ::

        pyenv install 3.9-dev
        pyenv virtualenv 3.9-dev py39

    Then, activate your Python environment and install :code:`georges-core` with Poetry ::

        pyenv local py39
        poetry install --without dev,docs

Conda environment
#################

The installation procedure which follows creates a `conda` environment
based on the Intel distribution for Python with all the necessary dependencies
included and managed via `conda` itself. Georges-core is then installed using `poetry` from that `conda` environment.

1. Install `Conda <https://conda.io/docs/>`_ or `Miniconda <https://conda.io/en/latest/miniconda.html>`_
for your operating system.

2. Obtain a copy of the git repository (see previous section)
3. Create a dedicated `conda` environment (default name is `georges-core`)::

    cd path/to/georges-core
    conda env create -f environment.yml

4. Activate the environment and update manually llvmlite::

    conda activate georges-core
    pip install -I --force-reinstall llvmlite

5. Install Georges-core using `poetry` from the `conda` environment::

    poetry install --without dev,docs

Georges-core can be subsequently updated by running::

    cd path/to/georges-core
    git pull origin master
    poetry update

To ensure the installation with the Intel distribution is correctly made,
you should run the commands without errors::

    conda activate georges-core
    python
    import numpy.random_intel

In table below, we summarize the performances between numpy.random and numpy.random_intel.
We compute the time to generate a Gaussian distribution with 1e8 particles::

    res = generator([0, 0, 0, 0, 0],
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
            ),
            int(1e8),
        )

.. list-table:: Performances comparison between numpy and numpy.intel
   :widths: 25 25
   :header-rows: 1

   * - Generator
     - Time (s)
   * - numpy.random.multivariate_normal
     - 59
   * - numpy.random_intel.multivariate_normal
     - 31


Using Georges-core with Jupyter Lab
###################################

Georges-core can be used with Jupyter lab. No special care is needed,
and you can simply run (note that it is not advised to put all your
notebook within the git structure)::

    cd somewhere/good/for/notebooks
    jupyter-lab


Georges-core distribution with Docker
#####################################

A Docker image is made available to provide an easy access to a
complete Jupyter Lab + georges-core environment.

Use the *Dockerfile* to build the image::

    docker build

or, to register the image as well::

    docker build -t georges-core -f Dockerfile .

You can run a container with::

    docker run -it --rm --name georges_core -p 8899:8899 georges-core

then connect to http://127.0.0.1:8899 to access the Jupyter Lab interface
and type::

    import georges_core

