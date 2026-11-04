.. _installation_instructions:

=========================
Installation Instructions
=========================

Requirements
------------
WMQL is packaged with a ``pyproject.toml`` file and installed with ``pip``,
which fetches the required dependencies automatically:

* `Python >= 3.9 <https://www.python.org>`_
* `NumPy >= 1.26 <https://numpy.org>`_
* `NiBabel >= 4.0.0 <https://nipy.org/nibabel/>`_
* `VTK > 8.1 <https://vtk.org>`_ (Python wheels, used to read and write VTK
  tractography files, e.g. for interacting with 3D Slicer)

We recommend installing WMQL in a dedicated virtual environment, created
either with ``venv``::

  python -m venv wmql-env
  source wmql-env/bin/activate

or with ``conda``::

  conda create -n wmql python=3.12
  conda activate wmql

Installation
------------

Install the latest version directly from GitHub::

  pip install git+https://github.com/demianw/tract_querier.git

Alternatively, clone the repository and install from the source tree::

  git clone https://github.com/demianw/tract_querier.git
  cd tract_querier
  pip install .

The package version is derived from the git tags with ``setuptools_scm``, so
install from a git clone rather than from a downloaded ZIP archive of the
source code.

Now you can check if the installation worked::

  tract_querier --help

Development
-----------

To work on WMQL, install it in editable mode together with the testing
dependencies, and run the test suite::

  pip install -e ".[test]"
  pytest tract_querier

To build this documentation locally::

  pip install ".[doc]"
  make -C doc html

The HTML pages are written to ``doc/_build/html``.
