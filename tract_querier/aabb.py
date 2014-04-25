import numpy as np

__all__ = ['BoundingBox']


class BoundingBox(np.ndarray):

    """Bounding box assuming RAS coordinate system"""
    def __new__(cls, input_array, info=None):
        try:
            if len(input_array) == 6 and np.isscalar(input_array[0]):
                pass
            elif len(input_array) >= 1 and len(input_array[0]) == 3:
                pass
            else:
                raise ValueError(
                    "Bounding box must have 6 components or be a list of points")
        except TypeError:
            raise ValueError(
                "Bounding box must have 6 components or be a list of points")
        except ValueError:
            raise ValueError(
                "Bounding box must have 6 components or be a list of points")

        input_array = np.asanyarray(input_array)

        if len(input_array) != 6 or input_array.ndim > 1:
            input_array = np.r_[
                np.min(input_array, axis=0),
                np.max(input_array, axis=0),
            ]

        array = input_array.view(cls)
        array.setflags(write=False)
        return input_array.view(cls)

    def __str__(self):
        return ('%s:%s' % (
            super(np.ndarray, self[:3]).__str__(),
            super(np.ndarray, self[3:]).__str__()
        )).replace('BoundingBox', '')

    @property
    def volume(self):
        return np.prod(self.side_lengths)

    @property
    def left(self):
        return self[0]

    @property
    def posterior(self):
        return self[1]

    @property
    def inferior(self):
        return self[2]

    @property
    def right(self):
        return self[3]

    @property
    def anterior(self):
        return self[4]

    @property
    def superior(self):
        return self[5]

    @property
    def side_lengths(self):
        return self[3:] - self[:3]

    def union(self, bounding_box):
        if isinstance(bounding_box, BoundingBox):
            _bounding_box = bounding_box
        else:
            _bounding_box = BoundingBox(bounding_box)
        return BoundingBox(np.r_[
            np.minimum(self[:3], _bounding_box[:3]),
            np.maximum(self[3:], _bounding_box[3:]),
        ])

    def contains(self, points):
        points = np.atleast_2d(points)
        lower = np.asarray(self[:3])[None, :]
        upper = np.asarray(self[3:])[None, :]

        self_min_ge_point = (points <= upper).all(1)
        self_max_le_point = (lower <= points).all(1)

        return self_min_ge_point * self_max_le_point

    def intersection(self, bounding_box):
        bounding_boxes = np.atleast_2d(np.asarray(bounding_box))
        lower = np.asarray(self[:3])[None, :]
        upper = np.asarray(self[3:])[None, :]

        self_min_ge_other_min = (bounding_boxes[:, :3] <= lower).any(1)
        self_min_le_other_max = (lower <= bounding_boxes[:, 3:]).any(1)

        self_max_ge_other_min = (bounding_boxes[:, :3] <= upper).any(1)
        self_max_le_other_max = (lower <= bounding_boxes[:, 3:]).any(1)

        other_min_ge_self_min = (lower <= bounding_boxes[:, :3]).any(1)
        other_min_le_self_max = (bounding_boxes[:, :3] <= upper).any(1)

        other_max_ge_self_min = (lower <= bounding_boxes[:, 3:]).any(1)
        other_max_le_self_max = (bounding_boxes[:, 3:] <= upper).any(1)

        one_point_self_inside_other = (
            self_min_ge_other_min * self_min_le_other_max +
            self_max_ge_other_min * self_max_le_other_max
        )

        one_point_other_inside_self = (
            other_min_ge_self_min * other_min_le_self_max +
            other_max_ge_self_min * other_max_le_self_max
        )

        intersection_exists = (
            one_point_self_inside_other +
            one_point_other_inside_self
        )

        intersection = np.c_[
            np.maximum(bounding_boxes[:, :3], lower),
            np.minimum(bounding_boxes[:, 3:], upper),
        ]

        if bounding_box.ndim == 1:
            if any(intersection_exists):
                return BoundingBox(intersection[0])
            else:
                return None
        else:
            return intersection_exists, intersection

    def old_intersection(self, bounding_box):
        one_point_self_inside_bounding_box = np.all(
            (self[:3] >= bounding_box[:, :3]) *
            (self[3:] <= bounding_box[:, 3:]),
            1
        )
        one_point_bounding_box_inside_self = np.all(
            (self[:3] <= bounding_box[:, :3]) *
            (self[3:] >= bounding_box[:, 3:]),
            1
        )
        if one_point_self_inside_bounding_box or one_point_bounding_box_inside_self:
            return BoundingBox(np.r_[
                np.maximum(self[:3], bounding_box[:, :3]),
                np.minimum(self[3:], bounding_box[:, 3:]),
            ])
        else:
            return None

    def collides_width(self, bounding_box):
        return self.intersection(bounding_box) is not None

    def split(self, axis):
        box_minor = self.copy()
        box_major = self.copy()
        half_length = (self[axis + 3] - self[axis]) / 2.

        box_minor.setflags(write=True)
        box_major.setflags(write=True)

        box_minor[axis + 3] -= half_length
        box_major[axis] += half_length

        box_minor.setflags(write=False)
        box_major.setflags(write=False)
        return box_minor, box_major

    def __contains__(self, point):
        point = np.atleast_2d(point)
        return (self[:3] <= point).prod(1) * (point <= self[:3]).prod(1)


class AABBTree:
    tree = None
    leaves = None
    forwardMap = None
    reverseMap = None

    def __init__(self, allboxes, indices=None):
        self.tree, self.leaves = self.buildTree(allboxes, indices=indices)
        self.forwardMap = np.arange(len(allboxes))
        self.reverseMap = np.arange(len(allboxes))

    def intersect(self, box):
        return self._intersect(self.tree, box)

    def _intersect(self, tree, box):
        treeBox = tree.box
        if all(
              (box[::2] <= treeBox[1::2]) *
              (box[1::2] >= treeBox[::2])
        ):
            if isinstance(tree, self.leaf):
                return [tree]
            else:
                return self._intersect(tree.left, box) + self._intersect(tree.right, box)
        else:
            return []

    def build_tree(self, allboxes, indices=None, leafPointers=None, parent=None):
        if indices == None:
            indices = np.arange(len(allboxes), dtype=np.int)
            boxes = allboxes
        else:
            if len(indices) == 1:
                return self.leaf(allboxes[indices[0], :], indices[0], parent=parent)
            boxes = allboxes[indices, :]

    def buildTree_old(self, allboxes, indices=None, leafPointers=None, parent=None, verbose=False):
        """
        Each element of the box list is a 6-ple b where
          b = [x-,x+,y-,y+,z-,z+]
        """

        if indices == None:
            indices = np.arange(len(allboxes), dtype=np.int)
            boxes = allboxes
        else:
            if len(indices) == 1:
                return self.leaf(allboxes[indices[0], :], indices[0], parent=parent)
            boxes = allboxes[indices, :]

        if verbose:
            print '*******************************************'

        dimensions = len(boxes[0]) / 2

        box = np.empty(2 * dimensions)
        np.min(boxes[:, ::2], axis=0, out=box[::2])
        np.max(boxes[:, 1::2], axis=0, out=box[1::2])
        boxes[:, ::2].min(0, out=box[::2])
        boxes[:, 1::2].max(0, out=box[1::2])

        lengths = box[1::2] - box[::2]
        largestDimension = lengths.argmax()
        largestDimensionLength = lengths[largestDimension]
        cuttingPlaneAt = largestDimensionLength / 2.

        halfLength = (
            boxes[:, 2 * largestDimension] +
            (
                boxes[:, 2 * largestDimension + 1] - boxes[:, 2 * largestDimension]
            ) / 2.
        )

        halfLengthSortedIndices = halfLength.argsort()
        halfLengthSorted = halfLength[halfLengthSortedIndices]
        division = halfLengthSorted.searchsorted(cuttingPlaneAt)

        leftIndices = indices[halfLengthSortedIndices[:division]]
        rightIndices = indices[halfLengthSortedIndices[division:]]

        if len(leftIndices) == 0 or len(rightIndices) == 0:
            n = len(indices) / 2
            leftIndices = indices[:n]
            rightIndices = indices[n:]

        if verbose:
            print "Left: ", leftIndices
            print "Right: ", rightIndices
        n = self.node(box, indices.copy(), parent=parent)
        n.left = self.buildTree(
            allboxes, leftIndices, parent=n, leafPointers=leafPointers)
        n.right = self.buildTree(
            allboxes, rightIndices, parent=n, leafPointers=leafPointers)
        if parent != None:
            return n
        else:
            return n, leafPointers

    class node:
        box = None
        indices = None
        left = None
        right = None
        parent = None

        def __init__(self, box, indices, left=None, right=None, parent=None):
            self.box = box
            self.indices = indices
            self.left = left
            self.right = right
            self.parent = parent

        def __str__(self):
            return """
      box = %s
      indices = %s
      """ % (
                self.box,
                self.indices,
            )

        def __repr__(self):
            return self.__str__()

    class leaf:
        box = None
        indices = None
        parent = None

        def __init__(self, box, indices, parent=None):
            self.box = box
            self.indices = indices
            self.parent = parent

        def __str__(self):
            return """
      box = %s
      indices = %s
      """ % (
                self.box,
                self.indices,
            )

        def __repr__(self):
            return self.__str__()
