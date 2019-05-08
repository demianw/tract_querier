from warnings import warn
from six.moves import range

import numpy

from .tractography import Tractography

from nibabel import trackvis


def tractography_to_trackvis_file(filename, tractography, affine=None, image_dimensions=None):
    trk_header = trackvis.empty_header()

    if affine is not None:
        pass
    elif hasattr(tractography, 'affine'):
        affine = tractography.affine
    else:
        raise ValueError("Affine transform has to be provided")

    trackvis.aff_to_hdr(affine, trk_header, True, True)
    trk_header['origin'] = 0.
    if image_dimensions is not None:
        trk_header['dim'] = image_dimensions
    elif hasattr(tractography, 'image_dimensions'):
        trk_header['dim'] = tractography.image_dimensions
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

    #data_new = {}
    # for k, v in data.iteritems():
    #    if (v[0].ndim > 1 and v[0].shape[1] > 1):
    #        for i in xrange(v[0].shape[1]):
    #            data_new['%s_%02d' % (k, i)] = [
    #                v_[:, i] for v_ in v
    #            ]
    #    else:
    #       data_new[k] = v
    trk_header['n_count'] = len(tractography.tracts())
    trk_header['n_properties'] = 0
    trk_header['n_scalars'] = len(data)

    if len(data) > 10:
        raise ValueError('At most 10 scalars permitted per point')

    trk_header['scalar_name'][:len(data)] = numpy.array(
        [n[:20] for n in data],
        dtype='|S20'
    )
    trk_tracts = []

    for i, sl in enumerate(tractography.tracts()):
        scalars = None
        if len(data) > 0:
            scalars = numpy.vstack([
                data[k.decode('utf8')][i].squeeze()
                for k in trk_header['scalar_name'][:len(data)]
            ]).T

        trk_tracts.append((sl, scalars, None))

    trackvis.write(filename, trk_tracts, trk_header, points_space='rasmm')


def tractography_from_trackvis_file(filename):
    tracts_and_data, header = trackvis.read(filename, points_space='rasmm')

    tracts, scalars, properties = list(zip(*tracts_and_data))

    scalar_names = [n for n in header['scalar_name'] if len(n) > 0]

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

    tracts_data = {}
    for i, sn in enumerate(scalar_names):
        if hasattr(sn, 'decode'):
            sn = sn.decode()
        tracts_data[sn] = [scalar[:, i][:, None] for scalar in scalars]

    affine = header['vox_to_ras']
    image_dims = header['dim']

    tr = Tractography(
        tracts, tracts_data,
        affine=affine, image_dims=image_dims
    )

    return tr
