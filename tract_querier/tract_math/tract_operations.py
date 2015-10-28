import numpy


def tract_in_ras(image, tract_ijk):
    ijk_points = tract_ijk
    ras_points = numpy.dot(image.get_affine(), numpy.hstack((
        ijk_points,
        numpy.ones((len(ijk_points), 1))
    )).T).T[:, :-1]
    return ras_points


def tract_in_ijk(image, tractography):
    ras_points = numpy.vstack(tractography.tracts())
    ijk_points = numpy.linalg.solve(image.get_affine(), numpy.hstack((
        ras_points,
        numpy.ones((len(ras_points), 1))
    )).T).T[:, :-1]
    return ijk_points


def each_tract_in_ijk(image, tractography):
    ijk_tracts = []
    for tract in tractography.tracts():
        ijk_tracts.append(numpy.dot(numpy.linalg.inv(image.get_affine()), numpy.hstack((
            tract,
            numpy.ones((len(tract), 1))
        )).T).T[:, :-1])
    return ijk_tracts


def tract_probability_map(image, tractography):
    ijk_tracts = each_tract_in_ijk(image, tractography)
    image_data = image.get_data()

    probability_map = numpy.zeros_like(image_data, dtype=float)
    for ijk_points in ijk_tracts:
        ijk_clipped = ijk_points.clip(
            (0, 0, 0), numpy.array(image_data.shape) - 1
        ).astype(int)

        probability_map[tuple(ijk_clipped.T)] += 1

    probability_map /= len(ijk_tracts)
    return probability_map


def tract_mask(image, tractography):
    ijk_points = tract_in_ijk(image, tractography)
    image_data = image.get_data()

    ijk_clipped = ijk_points.clip(
        (0, 0, 0), numpy.array(image_data.shape) - 1
    ).astype(int)

    mask = numpy.zeros_like(image_data, dtype=float)
    mask[tuple(ijk_clipped.T)] = 1
    return mask


def voxelized_tract(tractography, resolution):
    from itertools import izip
    all_points = numpy.vstack(tractography.tracts())
    all_points /= resolution
    all_points = all_points.round(0).astype(int)
    return set(izip(*(all_points.T)))


def tract_count(tract):
    return len(tract)


def tract_length(tract):
    d2 = numpy.sqrt((numpy.diff(tract, axis=0) ** 2).sum(1))
    return d2.sum()
