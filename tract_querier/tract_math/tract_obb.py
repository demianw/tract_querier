from numpy import (
    vstack, where, intersect1d, in1d, unique,
    cross, abs,
    dot, array, nan_to_num,
    hstack, repeat, bincount, arange
)
from numpy.linalg import norm


class Box2D:

    def __init__(self, *args, **kwargs):
        if len(args) <= 5:
            self._compute_bounding_box(*args, **kwargs)
        else:
            self._set_variables(*args)

    def _compute_bounding_box(self, points, point_ids, vectors, labels=None, level=None):
        center = points.mean(0)
        centered_points = points - center

        orientation = vectors.sum(0)
        orientation /= norm(orientation)

        orthogonal_direction = orthogonal_vector(orientation)
        orthogonal_direction /= norm(orthogonal_direction)

        points_orthogonal = dot(
            orthogonal_direction,
            centered_points.T
        )

        points_orientation = dot(orientation, centered_points.T)

        max_main = points_orientation.max()
        min_main = points_orientation.min()
        max_orthogonal = points_orthogonal.max()
        min_orthogonal = points_orthogonal.min()

        bounding_box_corners = (vstack((
            orientation * max_main + orthogonal_direction * max_orthogonal,
            orientation * max_main + orthogonal_direction * min_orthogonal,
            orientation * min_main + orthogonal_direction * min_orthogonal,
            orientation * min_main + orthogonal_direction * max_orthogonal,
        )) + center)

        center = bounding_box_corners.mean(0)

        volume = (max_main - min_main) * (max_orthogonal - min_orthogonal)

        self.orthogonal = orthogonal_direction
        self.points_orientation = points_orientation
        self.points_orthogonal = points_orthogonal

        self._set_variables(
            bounding_box_corners, center, orientation,
            labels, points, point_ids, vectors, volume,
            None, None, None, level
        )

    def _set_variables(self,
                       box,
                       center,
                       orientation,
                       labels,
                       points,
                       point_ids,
                       vectors,
                       volume,
                       parent,
                       left,
                       right,
                       level
                       ):
        self.box = box
        self.center = center
        self.orientation = orientation
        self.labels = labels
        self.points = points
        self.point_ids = point_ids
        self.vectors = vectors
        self.volume = volume
        self.parent = parent
        self.left = left
        self.right = right
        self.level = level

        self._calculate_orientation_limits()
        self._calculate_orthogonal_limits()

    def _calculate_orientation_limits(self):
        projections = [dot(self.orientation, point) for point in self.box]
        self.orientation_limits = (min(projections), max(projections))

    def _calculate_orthogonal_limits(self):
        projections = [dot(self.orthogonal, point) for point in self.box]
        self.orthogonal_limits = (min(projections), max(projections))

    def siblings(self, generations_up=0, generations_down=0):
        if generations_up == 0 and generations_down == 1:
            left = [self.left] if self.left is not None else []
            right = [self.right] if self.right is not None else []
            return left + right

        elif generations_up > 0:
            if self.parent is None:
                return []
            return self.parent.siblings(generations_up - 1, generations_down + 1)
        elif generations_down > 1:
            if self.left is not None:
                left = self.left.siblings(0, generations_down - 1)
            else:
                left = []
            if self.right is not None:
                right = self.right.siblings(0, generations_down - 1)
            else:
                right = []

            return left + right

    def swap_direction(self):
        self.orientation *= -1
        self._calculate_orientation_limits()

    def overlap_main(self, box):
        projections = [dot(self.orientation, point) for point in box.box]
        orientation_limits = (min(projections), max(projections))

        if  (
            self.orientation_limits[0] <= orientation_limits[0] <= self.orientation_limits[1] or
            self.orientation_limits[0] <= orientation_limits[1] <= self.orientation_limits[1] or
            orientation_limits[0] <= self.orientation_limits[0] <= orientation_limits[1] or
            orientation_limits[0] <= self.orientation_limits[1] <= orientation_limits[1]
        ):
                return True
        return False

    def center_signed_orientational_distance(self, box):
        return dot(self.orientation, self.center - box.center)

    def center_distance(self, box):
        return norm(box.center - self.center)

    def __repr__(self):
        return self.box.__repr__() + '\n' +\
            'level:' + repr(self.level)

    def __str__(self):
        return self.box.__str__() + '\n' +\
            'level:' + str(self.level)


class Box3D(Box2D):

    def _compute_bounding_box(self, points, point_ids, vectors, labels=None, level=None):
        original_points = points
        original_point_ids = point_ids
        original_labels = labels
        original_vectors = vectors

        orientation = vectors.mean(0)
        orientation /= norm(orientation)

        orthogonal_direction1 = orthogonal_vector(orientation)
        orthogonal_direction2 = cross(orientation, orthogonal_direction1)

        orthogonal_direction1 /= norm(orthogonal_direction1)
        orthogonal_direction2 /= norm(orthogonal_direction2)

        center = points.mean(0)
        centered_points = points - center

        points_orientation = dot(orientation, centered_points.T)
        points_orthogonal1 = dot(
            orthogonal_direction1,
            centered_points.T
        )
        points_orthogonal2 = dot(
            orthogonal_direction2,
            centered_points.T
        )

        max_main, min_main = points_orientation.max(), points_orientation.min()
        max_orthogonal1, min_orthogonal1 = (
            points_orthogonal1.max(),
            points_orthogonal1.min()
        )
        max_orthogonal2, min_orthogonal2 = (
            points_orthogonal2.max(),
            points_orthogonal2.min()
        )

        bounding_box_corners = (vstack((
            orientation * max_main + orthogonal_direction1 *
            max_orthogonal1 + orthogonal_direction2 * max_orthogonal2,
            orientation * max_main + orthogonal_direction1 *
            max_orthogonal1 + orthogonal_direction2 * min_orthogonal2,
            orientation * max_main + orthogonal_direction1 *
            min_orthogonal1 + orthogonal_direction2 * min_orthogonal2,
            orientation * max_main + orthogonal_direction1 *
            min_orthogonal1 + orthogonal_direction2 * max_orthogonal2,
            orientation * min_main + orthogonal_direction1 *
            max_orthogonal1 + orthogonal_direction2 * max_orthogonal2,
            orientation * min_main + orthogonal_direction1 *
            max_orthogonal1 + orthogonal_direction2 * min_orthogonal2,
            orientation * min_main + orthogonal_direction1 *
            min_orthogonal1 + orthogonal_direction2 * min_orthogonal2,
            orientation * min_main + orthogonal_direction1 *
            min_orthogonal1 + orthogonal_direction2 * max_orthogonal2,
        )) + center)

        center = bounding_box_corners.mean(0)

        volume = (
            (max_main - min_main) *
            (max_orthogonal1 - min_orthogonal1) *
            (max_orthogonal2 - min_orthogonal2)
        )

        self.orthogonal1 = orthogonal_direction1
        self.orthogonal2 = orthogonal_direction2
        self.points_orientation = points_orientation
        self.points_orthogonal1 = points_orthogonal1
        self.points_orthogonal2 = points_orthogonal2

        self._set_variables(
            bounding_box_corners, center, orientation,
            original_labels, original_points, original_point_ids, original_vectors, volume,
            None, None, None, level
        )

    def _calculate_orthogonal_limits(self):
        projections = dot(self.orthogonal1, self.box.T).T
        self.orthogonal1_limits = (min(projections), max(projections))
        projections = dot(self.orthogonal2, self.box.T).T
        self.orthogonal2_limits = (min(projections), max(projections))

    def overlap_main(self, box):
        projections = dot(self.orientation, box.box.T).T
        orientation_limits = (min(projections), max(projections))

        return (
            self.orientation_limits[0] <= orientation_limits[0] <= self.orientation_limits[1] or
            self.orientation_limits[0] <= orientation_limits[1] <= self.orientation_limits[1] or
            orientation_limits[0] <= self.orientation_limits[0] <= orientation_limits[1] or
            orientation_limits[0] <= self.orientation_limits[1] <= orientation_limits[1]
        )

    def overlap_orthogonal(self, box):
        projections = dot(self.orthogonal1, box.box.T).T
        orthogonal1_limits = (min(projections), max(projections))

        if (
            self.orthogonal1_limits[0] <= orthogonal1_limits[0] <= self.orthogonal1_limits[1] or
            self.orthogonal1_limits[0] <= orthogonal1_limits[1] <= self.orthogonal1_limits[1] or
            orthogonal1_limits[0] <= self.orthogonal1_limits[0] <= orthogonal1_limits[1] or
            orthogonal1_limits[0] <= self.orthogonal1_limits[1] <= orthogonal1_limits[1]
        ):
            overlap_orthogonal1 = True

        if not overlap_orthogonal1:
            return False

        projections = dot(self.orthogonal2, box.box.T).T
        orthogonal2_limits = (min(projections), max(projections))

        if (
            self.orthogonal2_limits[0] <= orthogonal2_limits[0] <= self.orthogonal2_limits[1] or
            self.orthogonal2_limits[0] <= orthogonal2_limits[1] <= self.orthogonal2_limits[1] or
            orthogonal2_limits[0] <= self.orthogonal2_limits[0] <= orthogonal2_limits[1] or
            orthogonal2_limits[0] <= self.orthogonal2_limits[1] <= orthogonal2_limits[1]
        ):
                overlap_orthogonal2 = True

        return overlap_orthogonal1 and overlap_orthogonal2

    def overlap(self, box):
        return self.overlap_main(box) and self.overlap_orthogonal(box)

    def overlap_volume(self, box):
        projections = dot(self.orientation, box.box.T).T
        orientation_limits = (min(projections), max(projections))

        if not (
            self.orientation_limits[0] <= orientation_limits[0] <= self.orientation_limits[1] or
            self.orientation_limits[0] <= orientation_limits[1] <= self.orientation_limits[1] or
            orientation_limits[0] <= self.orientation_limits[0] <= orientation_limits[1] or
            orientation_limits[0] <= self.orientation_limits[1] <= orientation_limits[1]
        ):
            return 0

        overlap_main_length =\
            min(orientation_limits[1], self.orientation_limits[1]) -\
            max(orientation_limits[0], self.orientation_limits[0])

        projections = dot(self.orthogonal1, box.box.T).T
        orthogonal1_limits = (min(projections), max(projections))

        if not\
            (self.orthogonal1_limits[0] <= orthogonal1_limits[0] <= self.orthogonal1_limits[1] or
             self.orthogonal1_limits[0] <= orthogonal1_limits[1] <= self.orthogonal1_limits[1] or
             orthogonal1_limits[0] <= self.orthogonal1_limits[0] <= orthogonal1_limits[1] or
             orthogonal1_limits[0] <= self.orthogonal1_limits[1] <= orthogonal1_limits[1]):
                return 0

        overlap_orthogonal1_length = \
            min(orthogonal1_limits[1], self.orthogonal1_limits[1]) -\
            max(orthogonal1_limits[0], self.orthogonal1_limits[0])

        projections = dot(self.orthogonal2, box.box.T)
        orthogonal2_limits = (min(projections), max(projections))

        if not\
            (self.orthogonal2_limits[0] <= orthogonal2_limits[0] <= self.orthogonal2_limits[1] or
             self.orthogonal2_limits[0] <= orthogonal2_limits[1] <= self.orthogonal2_limits[1] or
             orthogonal2_limits[0] <= self.orthogonal2_limits[0] <= orthogonal2_limits[1] or
             orthogonal2_limits[0] <= self.orthogonal2_limits[1] <= orthogonal2_limits[1]):
                return 0

        overlap_orthogonal2_length = \
            min(orthogonal2_limits[1], self.orthogonal2_limits[1]) -\
            max(orthogonal2_limits[0], self.orthogonal2_limits[0])

        return overlap_main_length * overlap_orthogonal1_length * overlap_orthogonal2_length


def orthogonal_vector(vector, tol=1e-8):
    a_vector = abs(vector)
    if len(vector) == 3:
        if a_vector[0] > tol:
            orthogonal = vector[::-1] * (1, 0, -1)
        elif a_vector[2] > tol:
            orthogonal = vector[::-1] * (-1, 0, 1)
        elif a_vector[1] > tol:
            orthogonal = vector[::-1] * (-1, 0, 0)
        else:
            raise ValueError('vector must have non-null norm')
    else:
        if a_vector[0] > tol:
            orthogonal = vector[::-1] * (-1, 1)
        elif a_vector[1] > tol:
            orthogonal = vector[::-1] * (1, -1)
        else:
            raise ValueError('vector must have non-null norm')

    orthogonal /= norm(orthogonal)
    return orthogonal


def box_cut(points, direction, mapped_points=None, max_main=None, min_main=None):
    if mapped_points is None:
        mapped_points = dot(direction, points.T)

    if max_main is None:
        max_main = mapped_points.max()
    if min_main is None:
        min_main = mapped_points.min()

    mid_main = (max_main + min_main) / 2.
    split1 = where(mapped_points <= mid_main)
    split2 = where(mapped_points > mid_main)

    return split1, split2


def all_obb_3d_along_tract(
    points, vectors, labels, tol=1e-8, level=0,
    intersection_threshold=.8, split_threshold=.2,
    box=None, clean=False, point_ids=None
):
    if point_ids is None:
        point_ids = arange(len(points))

    if (
        (box is not None) and (points is box.points) and
        (vectors is box.vectors) and (labels is box.labels)
    ):
        box_center = box
        box.level = level
    else:
        box_center = Box3D(points, point_ids, vectors, labels, level)
    level += 1

    if len(points) == 1:
        return [box_center]

    unique_labels = unique(labels)

    left, right = box_cut(
        points, box_center.orientation,
        mapped_points=box_center.points_orientation
    )

    masks = {
        'left': left,
        'right': right
    }

    split_labels = {
        'left': labels[left],
        'right': labels[right]
    }

    labels_both = intersect1d(split_labels['left'], split_labels['right'])

    if clean:
        labels_count = bincount(labels)
        labels_count = {
            side: bincount(split_labels[side])
            for side in split_labels
        }

        labels_ratio = {
            side: nan_to_num(
                labels_count[side] * 1. / labels_count[:len(labels_count(side))]
            ) for side in labels_count
        }

    new_results = [box_center]

    if (
            (len(labels_both) <= len(unique_labels) * intersection_threshold) and
            (box_center.points_orientation.ptp() / 2. < min((norm(v) for v in vectors)))
    ):
        return new_results

    for side in ('left', 'right'):
        mask = masks[side]
        new_labels = split_labels[side]
        if len(new_labels) > 0:
            new_points = points[mask]
            new_point_ids = point_ids[mask]
            new_vectors = vectors[mask]

            if clean:
                clean_labels = in1d(
                    labels[side],
                    intersect1d(labels_both, (labels_ratio[side] > .2).nonzero()[0]),
                )
                new_points = new_points[clean_labels]
                new_point_ids = new_point_ids[clean_labels]
                new_vectors = new_vectors[clean_labels]
                new_labels = new_labels[clean_labels]

            if len(new_points) > 1:
                new_tree = all_obb_3d_along_tract(
                    new_points, new_vectors, new_labels,
                    tol=tol, level=level, point_ids=new_point_ids,
                    intersection_threshold=intersection_threshold, clean=clean
                )

                setattr(box_center, side, new_tree[0])
                getattr(box_center, side).parent = box_center

                new_results += new_tree

    return new_results


def prototype_tract(
        tracts, obb_tree=None, intersection_threshold=.01, minimum_level=0,
        clean=False, return_obb_tree=False, return_leave_centers=False
):
    if obb_tree is None:
        points = vstack([t[:-1] for t in tracts])
        vectors = vstack([t[1:] - t[:-1] for t in tracts])
        labels = hstack([repeat(i, len(t) - 1) for i, t in enumerate(tracts)])

        obb_tree = all_obb_3d_along_tract(
            points, vectors, labels,
            intersection_threshold=intersection_threshold, clean=clean
        )

    if minimum_level < 0:
        max_level = max((obb.level for obb in obb_tree))
        minimum_level = max_level + 1 - minimum_level

    leave_centers = array(
        [obb.center for obb in obb_tree if obb.left is None and obb.right is None and obb.level >
            minimum_level]
    )

    mse_tract = array([
        ((t[..., None] - leave_centers[..., None].T) ** 2).sum(1).min(0).sum()
        for t in tracts
    ])

    tract_index = mse_tract.argmin()

    if return_obb_tree or return_leave_centers:
        res = (tract_index,)
        if return_obb_tree:
            res += (obb_tree,)
        if return_leave_centers:
            res += (leave_centers,)
        return res
    else:
        return tract_index
