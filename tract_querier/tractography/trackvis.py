import os
from warnings import warn

import numpy

from .tractography import Tractography

import nibabel as nib


def tractography_to_trackvis_file(filename, tractography, affine=None, image_dimensions=None):

    # The below could have used
    # https://github.com/nipy/nibabel/blob/3.0.0/nibabel/streamlines/trk.py#L226
    # trk_file = TrkFile(tractogram, header)
    # trk_file.save(filename)

    trk_header_new_nbb = nib.streamlines.TrkFile.create_empty_header()

    if affine is not None:
        pass
    elif hasattr(tractography, 'affine'):
        affine = tractography.affine
    else:
        raise ValueError("Affine transform has to be provided")

    trk_header_new_nbb["vox_to_ras"] = affine

    trk_header_new_nbb["origin"] = 0.
    if image_dimensions is not None:
        trk_header_new_nbb["dimensions"] = image_dimensions
    elif hasattr(tractography, "image_dimensions"):
        trk_header_new_nbb['dimensions'] = tractography.image_dimensions
    else:
        raise ValueError("Image dimensions needed to save a trackvis file")

    orig_data = tractography.tracts_data()
    data = {}
    for k, v in orig_data.items():
        if not isinstance(v[0], numpy.ndarray):
            continue
        if (v[0].ndim > 1 and any(d > 1 for d in v[0].shape[1:])):
            warn(
                "Scalar data %s ignored as trackvis "
                "format does not handle multivalued data" % k
            )
        else:
            data[k] = v

    trk_header_new_nbb["nb_streamlines"] = len(tractography.tracts())
    trk_header_new_nbb["nb_properties_per_streamline"] = 0
    trk_header_new_nbb["nb_scalars_per_point"] = len(data)

    if len(data) > 10:
        raise ValueError('At most 10 scalars permitted per point')

    trk_header_new_nbb["scalar_name"][:len(data)] = numpy.array(
        [n[:20] for n in data],
        dtype="|S20"
    )

    trk_header_new_nbb["image_orientation_patient"] = compute_image_orientation_patient(affine, True, True)

    affine_voxmm_to_rasmm = nib.streamlines.trk.get_affine_trackvis_to_rasmm(trk_header_new_nbb)

    streamlines = tractography.tracts()
    data_per_streamline = None
    data_per_point = data
    tractogram_nb = nib.streamlines.Tractogram(
        streamlines=streamlines,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point,
        affine_to_rasmm=affine_voxmm_to_rasmm,
    )
    nib.streamlines.save(tractogram_nb, filename, header=trk_header_new_nbb)


def tractography_from_trackvis_file(filename):

    trk_file = nib.streamlines.load(filename)

    strml_nibabel = trk_file.streamlines  # same as trk_file.tractogram.streamlines
    affine = nib.streamlines.trk.get_affine_trackvis_to_rasmm(trk_file.header)

    strml_nibabel_aff = nib.affines.apply_affine(
        numpy.linalg.inv(affine), strml_nibabel.get_data())

    affine_nb = trk_file.header["voxel_to_rasmm"]
    image_dims_nb = trk_file.header["dimensions"]

    tracts_data_store_nb = trk_file.tractogram.data_per_point.store
    dpp = {k: array_sequence_to_dpp(v) for k, v in tracts_data_store_nb.items()}

    tracts_nibabel = array_sequence_data_to_tracts(
        strml_nibabel_aff, strml_nibabel._offsets, strml_nibabel._lengths)
    tr = Tractography(
        tracts_nibabel, dpp,
        affine=affine_nb, image_dims=image_dims_nb
    )

    return tr


def compute_image_orientation_patient(affine, pos_vox, set_order):

    # Borrowed from
    # https://github.com/nipy/nibabel/blob/3.0.0/nibabel/trackvis.py#L685

    affine = numpy.dot(numpy.diag([-1, -1, 1, 1]), affine)
    # trans = affine[:3, 3]
    # Get zooms
    RZS = affine[:3, :3]
    zooms = numpy.sqrt(numpy.sum(RZS * RZS, axis=0))
    RS = RZS / zooms
    # If you said we could, adjust zooms to make RS correspond (below) to a
    # true rotation matrix.  We need to set the sign of one of the zooms to
    # deal with this.  Trackvis (the application) doesn't like negative zooms
    # at all, so you might want to disallow this with the pos_vox option.
    if not pos_vox and numpy.linalg.det(RS) < 0:
        zooms[0] *= -1
        RS[:, 0] *= -1
    # retrieve rotation matrix from RS with polar decomposition.
    # Discard shears because we cannot store them.
    P, S, Qs = numpy.linalg.svd(RS)
    R = numpy.dot(P, Qs)
    return R[:, 0:2].T.ravel()


def array_sequence_to_dpp(array_seq):

    dpp = []
    for offset, length in zip(array_seq._offsets, array_seq._lengths):
        val = array_seq._data[offset: offset + length]
        dpp.append(val)

    return dpp


def array_sequence_data_to_tracts(array_seq_data, offsets, lengths):

    tracts = []
    for offset, length in zip(offsets, lengths):
        val = array_seq_data[offset: offset + length]
        tracts.append(val)

    return tuple(tracts)
