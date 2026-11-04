import warnings

import numpy as np

from .aabb import BoundingBox

__all__ = ['TractographySpatialIndexing']


class TractographySpatialIndexing:

    r"""
    This class implements a mutual spatial indexing of
    an labeled image and a tractography

    Parameters
    ----------
    tractography : :class:`~tract_querier.tractography.Tractography`
                Tractography object
    image : array_like, 3-dimensional
        a piecewise constant 3D image or image of labels
    affine_ijk_2_ras : array_like, :math:`4 \times 4`
        the affine transform of each IJK coordinate on the image to RAS space
    length_threshold : float
        minimum length in mm of a tract to be considered in the indexing
    crossing_threshold : float
        the ratio of a tract that needs to be inside a label to be considered
        that it crosses it

    Attributes
    ----------
    tractography : :class:`~tract_querier.tractography.Tractography`
                Tractography object
    image : array_like, 3-dimensional
        a piecewise constant 3D image or image of labels
    affine_ijk_2_ras : array_like, :math:`4 \times 4`
        the affine transform of each IJK coordinate on the image to RAS space
    length_threshold : float
        minimum length in mm of a tract to be considered in the indexing
    crossing_threshold : float
        the ratio of a tract that needs to be inside a label to be considered
        that it crosses it
    crossing_tracts_labels : dict of sets
        Dictionary indexed by tract number of the labels traversed by the tract
    crossing_labels_tracts : dict of sets
        Dictionary indexed by label number of the tracts traversing the labels
    ending_tracts_labels : (dict of int, dict of int)
        Dictionary of each endpoint of the tracts indexed by tract number and
        containing the label at which the endpoint is
    ending_labels_tracts : (dict of sets, dict of sets)
        Dictionary of each endpoint of the tracts indexed by label number and
        containing the tracts at which the endpoint in the label is
    tract_endpoints_pos : array_like of :math:`N\times 2 \times 3` where
        :math:`N` is the number of tracts
        Contains the position of both endpoints of each tract
    """

    def __init__(
        self, tractography, image, affine_ijk_2_ras,
        length_threshold, crossing_threshold
    ):
        self.tractography = tractography
        self.image = image
        self.affine_ijk_2_ras = affine_ijk_2_ras
        self.affine_ras_2_ijk = np.linalg.inv(affine_ijk_2_ras)
        self.length_threshold = length_threshold
        self.crossing_threshold = crossing_threshold

        (
            self.crossing_tracts_labels, self.crossing_labels_tracts,
            self.ending_tracts_labels, self.ending_labels_tracts
        ) = compute_tract_label_indices(
            self.affine_ras_2_ijk, self.image,
            self.tractography, self.length_threshold, self.crossing_threshold
        )

        self.label_bounding_boxes = compute_label_bounding_boxes(
            self.image.astype(int), self.affine_ijk_2_ras
        )
        self.tract_bounding_boxes = compute_tract_bounding_boxes(
            self.tractography
        )

        self.tract_endpoints_pos = np.empty((len(self.tractography), 2, 3))

        for i, t in enumerate(self.tractography):
            self.tract_endpoints_pos[i, 0] = t[0]
            self.tract_endpoints_pos[i, 1] = t[-1]


def compute_label_bounding_boxes(image, affine_ijk_2_ras):
    r"""
    Bounding box in RAS space of each non-zero label of the image

    The box of a label spans the RAS coordinates of the centres of the
    voxels of that label.
    """
    linear_component = affine_ijk_2_ras[:3, :3]
    translation = affine_ijk_2_ras[:-1, -1]
    image = image.astype(int)

    coords = np.argwhere(image != 0)
    if len(coords) == 0:
        return {}
    labels = image[tuple(coords.T)]
    order = np.argsort(labels)
    labels = labels[order]
    ras_coords = np.dot(coords[order], linear_component.T) + translation

    unique_labels, starts = np.unique(labels, return_index=True)
    mins = np.minimum.reduceat(ras_coords, starts)
    maxs = np.maximum.reduceat(ras_coords, starts)

    return {
        int(label): BoundingBox(np.r_[label_min, label_max])
        for label, label_min, label_max in zip(unique_labels, mins, maxs)
    }


def compute_tract_bounding_boxes(tracts, affine_transform=None):
    bounding_boxes = np.empty((len(tracts), 6), dtype=float)

    if affine_transform is not None:
        linear_component = affine_transform[:3, :3]
        translation = affine_transform[:-1, -1]

    for i, tract in enumerate(tracts):
        if affine_transform is not None:
            ras_coords = (
                np.dot(linear_component, tract.T).T +
                translation
            )
        else:
            ras_coords = tract

        if len(ras_coords) < 2:
            raise ValueError(
                'Tracts in the tractography must have at least 2 points'
                ' tract #%d has less than two points.'
                ' You can use the tract_math tool to prune short tracts'
                ' and solve this problem.' % i
            )
        bounding_boxes[i] = BoundingBox(ras_coords)

    box_array = np.empty(
        len(tracts),
        dtype=[(name, float) for name in (
            'left', 'posterior', 'inferior',
            'right', 'anterior', 'superior'
        )])
    bounding_boxes = bounding_boxes.T
    for i, name in enumerate(box_array.dtype.names):
        box_array[name] = bounding_boxes[i]

    return box_array


def compute_label_crossings(tract_cumulative_lengths, point_labels, threshold):
    tracts_labels = {}
    for i in range(len(tract_cumulative_lengths) - 1):
        start = tract_cumulative_lengths[i]
        end = tract_cumulative_lengths[i + 1]
        label_crossings = np.asanyarray(point_labels[start:end], dtype=int)
        bincount = np.bincount(label_crossings)
        percentages = bincount * 1. / bincount.sum()
        tracts_labels[i] = set(np.where(percentages >= (threshold / 100.))[0])

    labels_tracts = {}
    for i, f in tracts_labels.items():
        for l in f:
            if l in labels_tracts:
                labels_tracts[l].add(i)
            else:
                labels_tracts[l] = set((i,))
    return tracts_labels, labels_tracts


def compute_label_endings(tract_cumulative_lengths, point_labels):
    tracts_labels = {}
    for i in range(len(tract_cumulative_lengths) - 1):
        start = tract_cumulative_lengths[i]
        end = tract_cumulative_lengths[i + 1]
        tracts_labels[i] = set((int(point_labels[
                               start]), int(point_labels[end - 1])))

    labels_tracts = {}
    for i, f in tracts_labels.items():
        for l in f:
            if l in labels_tracts:
                labels_tracts[l].add(i)
            else:
                labels_tracts[l] = set((i,))
    return tracts_labels, labels_tracts


def compute_label_endings_start_end(tract_cumulative_lengths, point_labels):
    tracts_labels_start = {}
    tracts_labels_end = {}
    for i in range(len(tract_cumulative_lengths) - 1):
        start = tract_cumulative_lengths[i]
        end = tract_cumulative_lengths[i + 1]
        tracts_labels_start[i] = int(point_labels[start])
        tracts_labels_end[i] = int(point_labels[end - 1])

    labels_tracts_start = {}
    labels_tracts_end = {}

    for tracts_labels, labels_tracts in (
        (tracts_labels_start, labels_tracts_start),
        (tracts_labels_end, labels_tracts_end)
    ):
        for i, l in tracts_labels.items():
            if l in labels_tracts:
                labels_tracts[l].add(i)
            else:
                labels_tracts[l] = set((i,))
    return (
        (tracts_labels_start, tracts_labels_end),
        (labels_tracts_start, labels_tracts_end)
    )


def compute_tract_label_indices(
    affine_ras_2_ijk, img,
    tracts, length_threshold, crossing_threshold
):
    if length_threshold > 0:
        tract_length = lambda tract: ((((
            tract[1:] - tract[:-1]
        ) ** 2).sum(1)) ** .5).sum()
        tracts = [f for f in tracts if tract_length(f) >= length_threshold]

    all_points = np.vstack(tracts)
    all_points_ijk = (np.dot(affine_ras_2_ijk[:-1, :-1], all_points.T).T +
                      affine_ras_2_ijk[:-1, -1])
    all_points_ijk_rounded = np.round(all_points_ijk).astype(int)

    if (any((
        (all_points_ijk_rounded[:, i] >= img.shape[i]).any() for i in range(3))
    ) or (all_points_ijk_rounded < 0).any()):
        warnings.warn("Warning tract points fall outside the image")

    for i in range(3):
        all_points_ijk_rounded[:, i] = all_points_ijk_rounded[
            :,
            i
        ].clip(0, img.shape[i] - 1)

    point_labels = img[tuple(all_points_ijk_rounded.T)]
    tract_cumulative_lengths = np.cumsum([0] + [len(f) for f in tracts])

    crossing_tracts_labels, crossing_labels_tracts = compute_label_crossings(
        tract_cumulative_lengths, point_labels, crossing_threshold
    )

    ending_tracts_labels, ending_labels_tracts = \
        compute_label_endings_start_end(tract_cumulative_lengths, point_labels)

    return (
        crossing_tracts_labels,
        crossing_labels_tracts,
        ending_tracts_labels,
        ending_labels_tracts
    )
