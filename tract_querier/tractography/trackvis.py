from itertools import izip

from tractography import Tractography

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

    if image_dimensions is not None:
        trk_header['dim'] = image_dimensions
    elif hasattr(tractography, 'image_dimensions'):
        trk_header['dim'] = image_dimensions
    else:
        raise ValueError("Image dimensions needed to save a tractvis file")

    trk_header['n_count'] = len(tractography.tracts())
    trk_tracks = [
        (
            sl, None, None
        )
        for sl in tractography.tracts()
    ]

    trackvis.write(filename, trk_tracks, trk_header, points_space='rasmm')


def tractography_from_trackvis_file(filename):
    tracts_and_data, header = trackvis.read(filename, points_space='rasmm')

    tracts, scalars, properties = izip(*tracts_and_data)

    affine = header['vox_to_ras']
    image_dims = header['dim']

    tr = Tractography(tracts)
    tr.affine = affine
    tr.image_dims = image_dims

    return tr
