import sys
import numpy as np


def compute_label_extremes(image, affine_ras_2_ijk):
    labels = np.unique(image)
    affine_ijk_2_ras = np.inv(affine_ras_2_ijk)

    label_extremes = np.empty(
        (labels > 0).sum(),
        dtype = (
        [('label_number', int)] +
        zip((
            'superior', 'inferior',
            'left', 'right' ,
            'anterior', 'posterior'
        ),(float,)*6)
        )
    )

    for i, label in enumerate(np.sort(labels[label > 0])):
        coords = np.where(image == label)
        ras_coords = (
            (affine_ijk_2_ras[:3,:3].dot(coords).T +
             affine_ijk_2_ras[:-1,-1]).min(0)
        )

        label_extremes['label_number'] = label
        left_post_inf = ras_coords.min(0)
        right_ant_sup = ras_coords.min(0)

        label_extremes['left'][i] = left_post_inf[0]
        label_extremes['posterior'][i] = left_post_inf[1]
        label_extremes['inferior'][i] = left_post_inf[1]

        label_extremes['right'][i] = right_ant_sup[0]
        label_extremes['anterior'][i] = right_ant_sup[1]
        label_extremes['superior'][i] = right_ant_sup[1]

    return label_extremes


def compute_label_crossings(i, fiber_cumulative_lengths, point_labels, threshold):
    fibers_labels = {}
    for i in xrange(len(fiber_cumulative_lengths) - 1):
        start = fiber_cumulative_lengths[i]
        end = fiber_cumulative_lengths[i + 1]
        label_crossings = np.asanyarray(point_labels[start:end], dtype=int)
        bincount = np.bincount(label_crossings)
        percentages = bincount * 1. / bincount.sum()
        fibers_labels[i] = set(np.where(percentages >= (threshold / 100.))[0])

    labels_fibers = {}
    for i, f in fibers_labels.items():
        for l in f:
            if l in labels_fibers:
                labels_fibers[l].add(i)
            else:
                labels_fibers[l] = set((i,))
    return fibers_labels, labels_fibers


def compute_fiber_label_indices(affine_ras_2_ijk, img, fibers, length_threshold, crossing_threshold):
    if length_threshold > 0:
        fiber_length = lambda fiber: ((((fiber[1:] - fiber[:-1]) ** 2).sum(1)) ** .5).sum()
        fibers = [f for f in fibers if fiber_length(f) >= length_threshold]

    all_points = np.vstack(fibers)
    all_points_ijk = (np.dot(affine_ras_2_ijk[:-1, :-1], all_points.T).T +\
                      affine_ras_2_ijk[:-1, -1])
    all_points_ijk_rounded = np.round(all_points_ijk).astype(int)

    if any( ((all_points_ijk_rounded[:, i] >= img.shape[i]).any() for i in xrange(3)))  or (all_points_ijk_rounded < 0).any():
        print >>sys.stderr, "Warning tract points fall outside the image"

    for i in xrange(3):
        all_points_ijk_rounded[:, i] = all_points_ijk_rounded[:, i].clip(0, img.shape[i] - 1)

    point_labels = img[tuple(all_points_ijk_rounded.T)]
    fiber_cumulative_lengths = np.cumsum([0] + [len(f) for f in fibers])

    fibers_labels, labels_fibers = compute_label_crossings(i, fiber_cumulative_lengths, point_labels, crossing_threshold)
    return fibers_labels, labels_fibers
