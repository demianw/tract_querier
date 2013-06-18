.. _installation_instructions:

=========================
Installation Instructions
=========================

Requirements
------------
* `Python 2.x, 2.7 or superior <http://www.python.org>`_
* `NumPy 1.6 or superior <http://www.numpy.org>`_
* `nibabel 1.3.x <http://nipy.sourceforge.net/nibabel/>`_
* If you want to be able to use VTK files to represent tractographies (like for interacting with slicer): `VTK 5.x installed along with its python wrappings <http://www.vtk.org>`_

All of these can be easily obtained from pre-packaged distributions such as `Canopy <https://www.enthought.com/products/canopy>`_ or `Anaconda <http://docs.continuum.io/anaconda/index.html>`_. In these cases, the packages corresponding to *VTK* and *nibabel* will need to be added.

Installation
------------

Once these requirements are met, the installation is as follows:

Downloading the source code from git::

  git clone http://github.com/demianw/tract_querier.git

Installing::

  cd tract_querier
  python setup.py install


Now you can check if the installation worked::

  tract_querier --help

