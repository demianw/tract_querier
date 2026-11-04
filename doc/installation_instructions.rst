.. _installation_instructions:

=========================
Installation Instructions
=========================

Requirements
------------
WMQL is packaged with a ``pyproject.toml`` file. Its required dependencies are
installed automatically:

* `Python >= 3.9 <https://www.python.org>`_
* `NumPy >= 1.26 <https://numpy.org>`_
* `NiBabel >= 4.0.0 <https://nipy.org/nibabel/>`_

`VTK <https://vtk.org>`_ is optional. It is only needed to read and write
tractographies in VTK format (``.vtk`` and ``.vtp`` files, e.g. for
interacting with 3D Slicer); TrackVis (``.trk``) files work without it. To
install it along with WMQL, request the ``vtk`` extra, as in the commands
below.

The commands below install the latest version from GitHub. Pick one of the
following tools.

Installing with pip
-------------------
We recommend installing WMQL in a dedicated virtual environment, created
either with ``venv``::

  python -m venv wmql-env
  source wmql-env/bin/activate

or with ``conda``::

  conda create -n wmql python=3.12
  conda activate wmql

Then install WMQL with VTK support::

  pip install "tract_querier[vtk] @ git+https://github.com/demianw/tract_querier.git"

or without it::

  pip install git+https://github.com/demianw/tract_querier.git

Installing with uv
------------------
`uv <https://docs.astral.sh/uv/>`_ can install the ``tract_querier`` and
``tract_math`` commands in an isolated environment of their own, and add them
to your ``PATH``::

  uv tool install "tract_querier[vtk] @ git+https://github.com/demianw/tract_querier.git"

To try the tools without installing them, use ``uvx``::

  uvx --from "tract_querier[vtk] @ git+https://github.com/demianw/tract_querier.git" tract_querier --help

To install WMQL as a library in a virtual environment instead::

  uv venv
  source .venv/bin/activate
  uv pip install "tract_querier[vtk] @ git+https://github.com/demianw/tract_querier.git"

Installing with pixi
--------------------
With `pixi <https://pixi.sh>`_, create a project for your analysis, take
Python, NumPy, NiBabel and VTK from conda-forge, and WMQL from GitHub::

  pixi init wmql
  cd wmql
  pixi add python=3.12 numpy nibabel vtk
  pixi add --pypi "tract_querier @ git+https://github.com/demianw/tract_querier.git"

The commands then run inside the project's environment::

  pixi run tract_querier --help

Installing from a clone of the repository
-----------------------------------------
Clone the repository and install from the source tree::

  git clone https://github.com/demianw/tract_querier.git
  cd tract_querier
  pip install ".[vtk]"

The package version is derived from the git tags with ``setuptools_scm``, so
install from a git clone rather than from a downloaded ZIP archive of the
source code.

Checking the installation
-------------------------
Now you can check if the installation worked::

  tract_querier --help

Development
-----------

To work on WMQL, install it in editable mode together with the testing
dependencies, and run the test suite::

  pip install -e ".[test,vtk]"
  pytest tract_querier

or, with uv::

  uv venv
  source .venv/bin/activate
  uv pip install -e ".[test,vtk]"
  pytest tract_querier

To build this documentation locally::

  pip install ".[doc]"
  make -C doc html

The HTML pages are written to ``doc/_build/html``.
