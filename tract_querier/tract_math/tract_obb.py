from numpy import (
    vstack, where, intersect1d, in1d, unique,
    cross, abs, arccos, sign,
    dot, array, cov, nan_to_num, inf, pi,
    hstack, repeat, bincount, arange
)
from numpy.linalg import norm, solve


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



class Box3DRich(Box2D):

    def _compute_bounding_box(self, points, point_ids, vectors, labels=None, level=None, robustify=None):
        original_points = points
        original_point_ids = point_ids
        original_labels = labels
        original_vectors = vectors
        if robustify == 'points' and len(points) > 4:
            p_mean = points.mean(0)
            p_cov = cov(points.T)

            c_points = points - p_mean

            z = (solve(p_cov, c_points.T) * c_points.T).sum(0)

            cutoff = 9.3484036044961485  # chi2.ppf(.975, 3)

            points = points[z < cutoff]
            point_ids = point_ids[z < cutoff]
            print(('Discarded', (len(original_points) - len(points)) * 1. / len(points)))
            vectors = vectors[z < cutoff]
            if labels is not None:
                labels = labels[z < cutoff]

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
            orthogonal_direction1, centered_points.T)
        points_orthogonal2 = dot(
            orthogonal_direction2, centered_points.T)

        max_main, min_main = points_orientation.max(), points_orientation.min()
        max_orthogonal1, min_orthogonal1 = points_orthogonal1.max(
        ), points_orthogonal1.min()
        max_orthogonal2, min_orthogonal2 = points_orthogonal2.max(
        ), points_orthogonal2.min()

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


def all_obb_2d(points, vectors, labels, tol=1e-8, level=0, intersection_threshold=.8, split_threshold=.2, box=None):

    if (box is not None) and (points is box.points) and (vectors is box.vectors) and (labels is box.labels):
        box_center = box
        box.level = level
    else:
        box_center = Box2D(points, vectors, labels, level)
    level += 1

    if len(unique(labels)) == 1:
        return [box_center]

    # First compute the splitting across the fibers
    split_along_fiber = True

    left, right = box_cut(points, box_center.orthogonal,
                          mapped_points=box_center.points_orthogonal)
    labels_left = labels[left]
    labels_right = labels[right]

    if len(intersect1d(labels_left, labels_right)) >= len(unique(labels)) * intersection_threshold:
        split_along_fiber = True
    else:
        points_left = points[left]
        vectors_left = vectors[left]
        box_left = Box2D(points_left, vectors_left, labels_left)

        points_right = points[right]
        vectors_right = vectors[right]
        box_right = Box2D(points_left, vectors_left, labels_left)

        if (box_left.volume + box_right.volume) < (1 - split_threshold) * box_center.volume:
            split_along_fiber = False
            left = all_obb_2d(
                points_left, vectors_left, labels_left, tol=tol, level=level,
                intersection_threshold=intersection_threshold, box=box_left)
            right = all_obb_2d(
                points_right, vectors_right, labels_right, tol=tol, level=level,
                intersection_threshold=intersection_threshold, box=box_right)
        else:
            split_along_fiber = True

    if split_along_fiber:  # If we could not split across we split along

        left, right = box_cut(
            points, box_center.orientation, mapped_points=box_center.points_orientation)
        labels_left = labels[left]
        labels_right = labels[right]
        if len(intersect1d(labels_left, labels_right)) <= len(unique(labels)) * intersection_threshold:
            return [box_center]

        points_left = points[left]
        vectors_left = vectors[left]
        left = all_obb_2d(
            points_left, vectors_left, labels_left, tol=tol, level=level,
            intersection_threshold=intersection_threshold)

        points_right = points[right]
        vectors_right = vectors[right]
        right = all_obb_2d(
            points_right, vectors_right, labels_right, tol=tol, level=level,
            intersection_threshold=intersection_threshold)

    box_center.left = left[0]
    box_center.right = right[0]
    left[0].parent = box_center
    right[0].parent = box_center

    return [box_center] + left + right


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


def all_obb_3d(points, vectors, labels, tol=1e-8, level=0, intersection_threshold=.8, split_threshold=.2, box=None, clean=False, point_ids=None):
    if point_ids is None:
        point_ids = arange(len(points))

    if (
        (box is not None) and (points is box.points) and
        (vectors is box.vectors) and (labels is box.labels)
    ):
        box_center = box
        box.level = level
    else:
        box_center = Box3D(
            points, point_ids, vectors,
            labels, level
        )
    level += 1

    if len(points) == 1:
        return [box_center]

    unique_labels = unique(labels)

    for orientation in ('orthogonal1', 'orthogonal2', 'orientation'):
        left, right = box_cut(
            points, getattr(box_center, orientation),
            mapped_points=getattr(box_center, 'points_' + orientation)
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

        if len(labels_both) == 0:
            break

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
    print(level)
    if (
            orientation == 'orientation' and
            (len(labels_both) <= len(unique_labels) * intersection_threshold)  # and
            #(box_center.points_orientation.ptp() / 2. > min((norm(v) for v in vectors)))
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
                new_tree = all_obb_3d(
                    new_points, new_vectors, new_labels, tol=tol, level=level, point_ids=new_point_ids,
                    intersection_threshold=intersection_threshold, clean=clean)

                setattr(box_center, side, new_tree[0])
                getattr(box_center, side).parent = box_center

                new_results += new_tree

    return new_results


def all_obb_3d_nr(points_, vectors_, labels_, tol=1e-8, level_=0, intersection_threshold=.8, split_threshold=.2, robustify=None, point_ids_=None):
    if point_ids_ is None:
        point_ids_ = arange(len(points_))

    root = Box3D(points_, point_ids_, vectors_, labels_, level_, robustify=robustify)
    stack = [root]

    total_points = len(points_)
    points_done = 0

    while len(stack):
        box = stack.pop()
        level = box.level + 1

        if len(box.points) == 1:
            continue

        unique_labels = unique(box.labels)

        for orientation in ('orthogonal1', 'orthogonal2', 'orientation'):
            left, right = box_cut(
                box.points, getattr(box, orientation),
                mapped_points=getattr(box, 'points_' + orientation)
            )

            masks = {
                'left': left,
                'right': right
            }

            split_labels = {
                'left': box.labels[left],
                'right': box.labels[right]
            }

            labels_both = intersect1d(split_labels['left'], split_labels['right'])

            if len(labels_both) == 0:
                break

        print((level, len(unique_labels), len(box.points), total_points - points_done))
        if (
                orientation == 'orientation' and
                (len(labels_both) <= len(unique_labels) * intersection_threshold)  # and
                #(box_center.points_orientation.ptp() / 2. > min((norm(v) for v in vectors)))
        ):
            points_done += len(box.points)
            continue

        for side in ('left', 'right'):
            mask = masks[side]
            new_labels = split_labels[side]
            if len(new_labels) > 0:
                new_points = box.points[mask]
                new_point_ids = box.point_ids[mask]
                new_vectors = box.vectors[mask]

                if len(new_points) > 1 and len(new_points) < len(box.points):
                    new_box = Box3D(new_points, new_point_ids, new_vectors, new_labels, level, robustify=robustify)
                    setattr(box, side, new_box)
                    getattr(box, side).parent = box

                    print(("\tAdded to stack ", side))
                    stack.append(new_box)
                else:
                    points_done += len(new_points)

    return root


def all_obb_3d_old(points, vectors, labels, tol=1e-8, level=0, intersection_threshold=.8, split_threshold=.2, box=None, point_ids=None):
    if point_ids is None:
        point_ids = arange(len(points))

    if (box is not None) and (points is box.points) and (vectors is box.vectors) and (labels is box.labels):
        box_center = box
        box.level = level
    else:
        box_center = Box3D(points, point_ids, vectors, labels, level)
    level += 1

    if len(points) == 1:
        return [box_center]

    # First compute the splitting across the fibers
    split_along_fiber = True

    o1_left, o1_right = box_cut(
        points, box_center.orthogonal1, mapped_points=box_center.points_orthogonal1)
    o2_left, o2_right = box_cut(
        points, box_center.orthogonal2, mapped_points=box_center.points_orthogonal2)
    o1_labels_left = labels[o1_left]
    o1_labels_right = labels[o1_right]
    o2_labels_left = labels[o2_left]
    o2_labels_right = labels[o2_right]

    unique_labels = unique(labels)
    if (
        len(intersect1d(o1_labels_left, o1_labels_right)) > 0 and
        len(intersect1d(o2_labels_left, o2_labels_right)) > 0
    ):
        split_along_fiber = True
    else:
        o1_box_left = Box3D(points[o1_left], vectors[o1_left], o1_labels_left)
        o1_box_right = Box3D(points[
                             o1_right], vectors[o1_right], o1_labels_right)

        o2_box_left = Box3D(points[o2_left], vectors[o2_left], o2_labels_left)
        o2_box_right = Box3D(points[
                             o2_right], vectors[o2_right], o2_labels_right)

        if (o1_box_left.volume + o1_box_right.volume) < (o1_box_left.volume + o2_box_right.volume):
            box_left = o1_box_left
            box_right = o1_box_right
        else:
            box_left = o2_box_left
            box_right = o2_box_right

        if (box_left.volume + box_right.volume) < (1 - split_threshold) * box_center.volume:
            split_along_fiber = False
            left = all_obb_3d(
                box_left.points, box_left.vectors, box_left.labels, tol=tol, level=level,
                intersection_threshold=intersection_threshold, box=box_left)
            right = all_obb_3d(
                box_right.points, box_right.vectors, box_right.labels, tol=tol, level=level,
                intersection_threshold=intersection_threshold, box=box_right)
        else:
            split_along_fiber = True

    if split_along_fiber:  # If we could not split across we split along

        left, right = box_cut(
            points, box_center.orientation, mapped_points=box_center.points_orientation)
        labels_left = labels[left]
        labels_right = labels[right]
        if len(intersect1d(labels_left, labels_right)) <= len(unique_labels) * intersection_threshold:
            return [box_center]

        points_left = points[left]
        point_ids_left = point_ids[left]
        vectors_left = vectors[left]
        left = all_obb_3d(
            points_left, point_ids_left, vectors_left, labels_left, tol=tol, level=level,
            intersection_threshold=intersection_threshold)

        points_right = points[right]
        point_ids_right = point_ids[left]
        vectors_right = vectors[right]
        right = all_obb_3d(
            points_right, point_ids_right, vectors_right, labels_right, tol=tol, level=level,
            intersection_threshold=intersection_threshold)

    box_center.left = left[0]
    box_center.right = right[0]
    left[0].parent = box_center
    right[0].parent = box_center

    return [box_center] + left + right


def point_coverage_by_level(obbs, points):
    level = 0
    points_level = [obb.points for obb in obbs if obb.level == level]
    level_coverage = []
    while len(points_level) > 0:
        level_coverage.append(sum((len(
            points) for points in points_level)) * 1. / len(points))
        level += 1
        points_level = [obb.points for obb in obbs if obb.level == level if len(
            obb.points) > 0]

    return array(level_coverage)


def draw_boxes_2d(obbs, level, color=None, **args):
    from pylab import plot, cm

    for i, obb in enumerate(obbs):
        if obb.level != level:
            continue
        box = vstack([obb.box, obb.box[0]])
        if color is None:
            plot(box.T[0], box.T[1], lw=5, hold=True, **args)
        else:
            plot(box.T[0], box.T[
                 1], lw=5, hold=True, c=cm.jet(color[i]), **args)


def draw_box_2d(obbs, **args):
        from pylab import plot, quiver
        if isinstance(obbs, Box2D):
            obbs = [obbs]
        for obb in obbs:
            box = vstack([obb.box, obb.box[0]])
            plot(box.T[0], box.T[1], lw=5, hold=True, **args)
            quiver([obb.center[0]], [obb.center[1]], [obb.orientation[
                   0]], [obb.orientation[1]], pivot='middle', hold=True, **args)


def draw_box_3d(obbs, tube_radius=1, color=None, **kwargs):
        from mayavi.mlab import plot3d
        from numpy.random import rand
        if isinstance(obbs, Box2D):
            obbs = [obbs]
        for obb in obbs:
            if color is None:
                color_ = tuple(rand(3))
            else:
                color_ = color
            box = obb.box
            b1 = vstack([box[:4], box[0]]).T
            b2 = vstack([box[4:], box[4]]).T
            es = [vstack([b1.T[i], b2.T[i]]).T for i in range(4)]
            plot3d(b1[0], b1[1], b1[
                   2], tube_radius=tube_radius, color=color_, **kwargs)
            plot3d(b2[0], b2[1], b2[
                   2], tube_radius=tube_radius, color=color_, **kwargs)
            [plot3d(e[0], e[1], e[
                    2], tube_radius=tube_radius, color=color_, **kwargs) for e in es]


def oriented_trace(obb, positive=True, generations=2, angle_threshold=pi / 4):
    tract = [obb]
    center = obb
    candidates = center.siblings(generations)
    if positive:
        sg = 1
    else:
        sg = -1
    while len(candidates) > 0:
        next_candidate_distance = inf
        for c in candidates:
            signed_distance = sg *\
                sign(center.center_signed_orientational_distance(c)) *\
                center.center_distance(c)
            if (signed_distance <= 0) or\
                not center.overlap_orthogonal(c) or\
                    arccos(dot(center.orientation, c.orientation)) > angle_threshold:
                continue

            if signed_distance < next_candidate_distance:
                next_candidate_distance = signed_distance
                next_candidate = c

        if next_candidate_distance < inf:
            if next_candidate in tract:
                break
            tract.append(next_candidate)
            if dot(center.orientation, next_candidate.orientation) < 0:
                next_candidate.swap_direction()
            center = next_candidate
            candidates = center.siblings(generations)
        else:
            break

    return tract


def trace(obb, generations=2, angle_threshold=pi / 4):
    trace_positive = oriented_trace(obb, True, generations, angle_threshold)
    trace_negative = oriented_trace(obb, False, generations, angle_threshold)

    return trace_negative[::-1] + trace_positive


def get_most_probable_trace(obbs, generations=2, angle_threshold=pi / 4, return_all=True):
    traces_list = [trace(obb, generations=generations,
                         angle_threshold=angle_threshold) for obb in obbs]
    traces_w_set = [(t, set(t)) for t in traces_list]
    n = 1. * len(traces_w_set)

    traces_with_frequency = []
    while len(traces_w_set) > 0:
        trace_ = traces_w_set.pop()
        traces_w_set_new = []
        count = 1
        for t in traces_w_set:
            if t[1] == trace_[1]:
                count += 1
            else:
                traces_w_set_new.append(t)
        traces_with_frequency.append((count / n, trace_[0]))
        traces_w_set = traces_w_set_new

    traces_with_frequency.sort(cmp=lambda x, y: int(sign(y[0] - x[0])))
    return traces_with_frequency


def get_level(tree, level):
    if tree is None or tree.level > level:
        return []
    elif tree.level == level:
        return [tree]
    else:
        return get_level(tree.left, level) + get_level(tree.right, level)


def overlapping_boxes(tree, box, levels=None, threshold=0.):
    if tree is None:
        return []

    overlap = tree.overlap_volume(box)

    if overlap < threshold:
        return []
    else:
        left = overlapping_boxes(
            tree.left, box, levels=levels, threshold=threshold)
        right = overlapping_boxes(
            tree.right, box, levels=levels, threshold=threshold)
        if levels is None or tree.level in levels:
            return [tree] + left + right
        else:
            return left + right


def containing_boxes(tree, box, levels=None, threshold=1.):
    if tree is None or tree.level > max(levels):
        return []

    normalized_overlap = tree.overlap_volume(box) / box.volume

    if normalized_overlap < threshold:
        return []
    else:
        left = overlapping_boxes(
            tree.left, box, levels=levels, threshold=threshold)
        right = overlapping_boxes(
            tree.right, box, levels=levels, threshold=threshold)
        if levels is None or tree.level in levels:
            return [tree] + left + right
        else:
            return left + right


def min_max(vector, axis=None):
    return array((vector.min(axis), vector.max(axis)))


def overlap_vtk(self, box):
    a = self
    b = box

    axes_a = vstack((a.orientation, a.orthogonal1, a.orthogonal2))
    axes_b = vstack((b.orientation, b.orthogonal1, b.orthogonal2))

    a2b = b.center - a.center

    a_a2b_limits = min_max(dot(a2b, a.box.T))
    b_a2b_limits = min_max(dot(a2b, a.box.T))

    if (
        a_a2b_limits[0] < b_a2b_limits[1] or
        b_a2b_limits[1] < a_a2b_limits[0]
    ):
        return False


def obb_tree_dfs(obb_tree):
    for obb in obb_tree:
        if obb.level == 0:
            root = obb
            break
    else:
        raise ValueError('No root in the tree')
    return obb_tree_dfs_recursive(root)


def obb_tree_dfs_recursive(obb_node):
    if obb_node is None:
        return []
    if obb_node.left is None and obb_node.right is None:
        return [obb_node]

    return obb_tree_dfs_recursive(obb_node.left) + obb_tree_dfs_recursive(obb_node.right)


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


def obb_tree_level(obb_tree, level, include_superior_leaves=True):
    if not isinstance(obb_tree, Box3D):
        node = obb_tree[0]
        for n in obb_tree:
            if n.level < node.level:
                node = n
    else:
        node = obb_tree

    return obb_tree_level_dfs(node, level, include_superior_leaves=include_superior_leaves)


def obb_tree_level_dfs(obb_node, level, include_superior_leaves=True):
    if obb_node is None or obb_node.level > level:
        return []

    if (
        obb_node.level == level or
        (
            include_superior_leaves and
            obb_node.level < level and
            obb_node.left is None and obb_node.right is None
        )
    ):
        return [obb_node]

    return (
        obb_tree_level_dfs(obb_node.left, level, include_superior_leaves=include_superior_leaves) +
        obb_tree_level_dfs(
            obb_node.right, level, include_superior_leaves=include_superior_leaves)
    )


def obb_from_tractography(tractography, *args, **kwargs):
    along_tract = False
    if 'along_tract' in kwargs and kwargs['along_tract']:
        along_tract = True

    fibers = tractography.tracts()
    points = vstack([f[:-1] for f in fibers])
    vectors = vstack([f[1:] - f[:-1] for f in fibers])
    labels = hstack([repeat(i, len(f) - 1) for i, f in enumerate(fibers)])

    if along_tract:
        obbs3d = all_obb_3d_along_tract(
            points, vectors, labels, **kwargs
        )
    else:
        obbs3d = all_obb_3d_nr(
            points, vectors, labels, **kwargs
        )

    return obbs3d
