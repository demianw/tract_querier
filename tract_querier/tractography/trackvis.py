from warnings import warn

import numpy

from tract_querier.tractography import Tractography

import nibabel as nib


def tractography_to_trackvis_file(filename, tractography, affine=None, image_dimensions=None):

    trk_header = nib.streamlines.TrkFile.create_empty_header()

    if affine is not None:
        pass
    elif hasattr(tractography, 'affine'):
        affine = tractography.affine
    else:
        raise ValueError("Affine transform has to be provided")

    trk_header['voxel_to_rasmm'] = affine
    if image_dimensions is not None:
        trk_header["dimensions"] = image_dimensions
    elif hasattr(tractography, 'image_dimensions'):
        trk_header["dimensions"] = tractography.image_dimensions
    else:
        raise ValueError("Image dimensions needed to save a trackvis file")

    orig_data = tractography.tracts_data()
    data_per_point = {}
    for k, v in orig_data.items():
        if not isinstance(v[0], numpy.ndarray):
            continue
        if (v[0].ndim > 1 and any(d > 1 for d in v[0].shape[1:])):
            warn(
                "Scalar data %s ignored as trackvis "
                "format does not handle multivalued data" % k
            )
        else:
            data_per_point[k] = v

    #data_new = {}
    # for k, v in data_per_point.iteritems():
    #    if (v[0].ndim > 1 and v[0].shape[1] > 1):
    #        for i in range(v[0].shape[1]):
    #            data_new['%s_%02d' % (k, i)] = [
    #                v_[:, i] for v_ in v
    #            ]
    #    else:
    #       data_new[k] = v
    trk_header['nb_streamlines'] = len(tractography.tracts())
    trk_header['nb_properties_per_streamline'] = 0
    trk_header['nb_scalars_per_point'] = len(data_per_point)

    if len(data_per_point) > 10:
        raise ValueError('At most 10 scalars permitted per point')

    trk_header['scalar_name'][:len(data_per_point)] = numpy.array(
        [n[:20] for n in data_per_point],
        dtype='|S20'
    )

    data_per_streamline = None
    tractogram = nib.streamlines.Tractogram(
        streamlines=tractography.tracts(),
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point,
        affine_to_rasmm=numpy.eye(4),
    )

    nib.streamlines.save(tractogram, filename, header=trk_header)


def tractography_from_trackvis_file(filename):
    trk_file = nib.streamlines.load(filename)

    tracts = array_sequence_data_to_tracts(
        trk_file.streamlines.get_data(),
        trk_file.streamlines._offsets,
        trk_file.streamlines._lengths,
    )

    tracts_dpp = trk_file.tractogram.data_per_point.store
    tracts_data = {}
    if tracts_dpp:
        tracts_data = {k: array_sequence_to_dpp(array_seq) for k, array_seq in tracts_dpp.items()}

    tracts_dps = trk_file.tractogram.data_per_streamline.store
    if tracts_dps:
        properties = dps_to_tuple(tracts_dps)

    #scalar_names_unique = []
    #scalar_names_subcomp = {}
    # for sn in scalar_names:
    #    if re.match('.*_[0-9]{2}', sn):
    #        prefix = sn[:sn.rfind('_')]
    #        if prefix not in scalar_names_unique:
    #            scalar_names_unique.append(prefix)
    #            scalar_names_subcomp[prefix] = int(sn[-2:])
    #        scalar_names_subcomp[prefix] = max(sn[-2:], scalar_names_subcomp[prefix])
    #    else:
    #        scalar_names_unique.append(sn)

    affine = trk_file.affine
    image_dims = trk_file.header['dimensions']

    tr = Tractography(
        tracts, tracts_data,
        affine=affine, image_dims=image_dims
    )

    return tr


def array_sequence_to_dpp(array_seq):

    dpp = []
    for offset, length in zip(array_seq._offsets, array_seq._lengths):
        val = array_seq._data[offset: offset + length]
        dpp.append(val)

    return dpp


def streamline_property_to_tuple(property):

    num_strml = len(property[0])
    num_dpps = len(property)
    return tuple(numpy.hstack([property[j][i] for j in range(num_dpps)]) for i in range(num_strml))


def dps_to_tuple(dps):

    dps_lists = list(dps.values())
    return streamline_property_to_tuple(dps_lists)


def array_sequence_data_to_tracts(array_seq_data, offsets, lengths):

    tracts = []
    for offset, length in zip(offsets, lengths):
        val = array_seq_data[offset: offset + length]
        tracts.append(val)

    return tuple(tracts)
