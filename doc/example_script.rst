.. _example_script:

====================
Example WMQL usage
====================

To run a WMQL script over and automatically extract a set of white matter tract bundles you will need:

* A full-brain tractography in VTK format: It must be a vtkPolyData object where all the cells are lines.
* A brain parcellation, such as the *wmparc.mgz* file obtained from  `freesurfer <https://surfer.nmr.mgh.harvard.edu>`_ in the same space as the full-brain tractography.
* A WMQL query file

Once all of these files are setup, the tract bundles are obtained by issuing the command:

.. code-block:: bash

  tract_querier -t tractography_file.vtk -a wmparc.nii.gz -q wmql_script.qry -o tract_output

where **tractography_file.vtk** is the full-brain tractography, **wmparc.nii.gz** is the brain parcellation, **wmql_script.qry** is the WMQL script and **tract_output** is the prefix for the output bundles.

There is an `example dataset available for download <_static/wmql_example_dataset.zip>`_ which you can use after following the :ref:`installation_instructions`.


WMQL Terms
----------
WMQL queries are based on combinations of the following terms

.. image:: WMQL_terms.png
   :width: 80 %
   :alt: Terms of the WMQL language
   :align: center

First WMQL example script: Cortico-Spinal Tract
-----------------------------------------------

.. literalinclude:: examples/wmql_1_cst.qry
  :language: wmql

More complex example: The Uncinate Fasciculus
---------------------------------------------

.. literalinclude:: examples/wmql_2_uf.qry
  :language: wmql

Globbing example: Commissural Tracts
-----------------------------------------
The whole point of this example is showing the use of
`glob expressions <http://en.wikipedia.org/wiki/Glob_(programming)>`_ to
define a region such as the left hemisphere

.. literalinclude:: examples/wmql_3_commissural.qry
  :language: wmql


.. toctree::
   :maxdepth: 2
