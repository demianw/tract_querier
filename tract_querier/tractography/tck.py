from warnings import warn

import numpy

from .tractography import Tractography

import nibabel as nib


def tractography_to_tck_file(filename, tractography):
    r"""
    Save a tractography as an MRtrix ``.tck`` file.

    TCK files store the tract points in RAS+ world coordinates in mm,
    the space tract_querier uses for its tracts, so the points are saved
    as they are. TCK has no image header and no per-point data: the
    ``affine`` and ``image_dims`` of the tractography, if any, are not
    saved and the tract data are dropped with a warning.

    Parameters
    ----------
    filename : str
        Name of the ``.tck`` file.
    tractography : Tractography
        Tractography to save.
    """
    dropped = sorted(
        k for k, v in tractography.tracts_data().items()
        if not isinstance(v, str)
    )
    if dropped:
        warn(
            "Tract data %s ignored as the TCK format does not store "
            "per-point data" % ', '.join(dropped)
        )

    tractogram = nib.streamlines.Tractogram(
        streamlines=tractography.tracts(),
        affine_to_rasmm=numpy.eye(4),
    )

    nib.streamlines.save(tractogram, filename)


def tractography_from_tck_file(filename):
    r"""
    Read an MRtrix ``.tck`` file.

    The tracts are returned in RAS+ world coordinates in mm, as stored in
    the file. As TCK files have no image header, the returned
    tractography has no ``affine`` or ``image_dims``: to save it as a
    ``.trk`` file give them to ``tractography_to_file``, which otherwise
    sets the identity and 1 1 1 with a warning, e.g. use those of the
    image the tracts were computed from.

    Parameters
    ----------
    filename : str
        Name of the ``.tck`` file.

    Returns
    -------
    tractography : Tractography
    """
    tck_file = nib.streamlines.load(filename)

    return Tractography(tuple(tck_file.streamlines))
