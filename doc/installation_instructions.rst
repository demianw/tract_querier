.. _installation_instructions:

=========================
Installation Instructions
=========================

Requirements
------------
* `Python >= 3.9, <3.12 <http://www.python.org>`_
* `NumPy >= 1.23, <2.0.0 <http://www.numpy.org>`_
* `nibabel >= 3.0.0, <4.0.0 <http://nipy.sourceforge.net/nibabel/>`_
* If you want to be able to use VTK files to represent tractographies (like for interacting with slicer): `VTK 8.2 installed along with its python wrappings <http://www.vtk.org>`_

All of these can be easily obtained from pre-packaged distributions such as `Anaconda <http://docs.continuum.io/anaconda/index.html>`_. In these cases, the packages corresponding to *VTK* and *nibabel* will need to be added.

Installation
------------

Once these requirements are met, the installation is as follows:

Downloading the source code from git::

  git clone http://github.com/demianw/tract_querier.git

Installing::

  cd tract_querier
  pip install .


Now you can check if the installation worked::

  tract_querier --help

