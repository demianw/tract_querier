import sys
import numpy as np

class BoundingBox(np.ndarray):
    """Bounding box assuming RAS coordinate system"""
    def __new__(cls, input_array, info=None):
        try:
            if len(input_array) == 6 and np.isscalar(input_array[0]):
                pass
            elif len(input_array) >= 1 and len(input_array[0]) == 3:
                pass
            else:
                raise ValueError("Bounding box must have 6 components or be a list of points")
        except TypeError:
            raise ValueError("Bounding box must have 6 components or be a list of points")
        except ValueError:
            raise ValueError("Bounding box must have 6 components or be a list of points")

        input_array = np.asanyarray(input_array)

        if len(input_array) != 6 or input_array.ndim > 1:
            input_array = np.r_[
                np.min(input_array, axis=0),
                np.max(input_array, axis=0),
            ]

        return input_array.view(cls)

    def __setitem__(self, *args, **kwargs):
        raise   NotImplementedError("Can not set item")

    def __setslice__(self, *args, **kwargs):
        raise   NotImplementedError("Can not set slice")

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

    def union(self, bounding_box):
        if isinstance(bounding_box, BoundingBox):
            _bounding_box = bounding_box
        else:
            _bounding_box = BoundingBox(bounding_box)
        return BoundingBox(np.r_[
            np.minimum(self[:3], _bounding_box[:3]),
            np.maximum(self[3:], _bounding_box[3:]),
        ])

    def intersection(self, bounding_box):
        if isinstance(bounding_box, BoundingBox):
            _bounding_box = bounding_box
        else:
            _bounding_box = BoundingBox(bounding_box)

        one_point_self_inside_bounding_box = np.all(
            (self.reshape(2, 3)>=_bounding_box[:3]) *
            (self.reshape(2, 3)<=_bounding_box[3:])
        )
        one_point_bounding_box_inside_self = np.all(
            (_bounding_box.reshape(2, 3)>=self[:3]) *
            (_bounding_box.reshape(2, 3)<=self[3:])
        )
        if one_point_self_inside_bounding_box or one_point_bounding_box_inside_self:
            return BoundingBox(np.r_[
                np.maximum(self[:3], _bounding_box[:3]),
                np.minimum(self[3:], _bounding_box[3:]),
            ])
        else:
            return None


def compute_label_bounding_boxes(image, affine_ijk_2_ras):
    labels = np.unique(image)
    linear_component = affine_ijk_2_ras[:3, :3]
    translation = affine_ijk_2_ras[:-1, -1]

    label_bounding_boxes = {}
    for i, label in enumerate(np.sort(labels)):
        if label == 0:
            continue

        coords = np.where(image == label)
        ras_coords = (
            (linear_component.dot(coords).T +
             translation)
        )

        label_bounding_boxes[label] = BoundingBox(ras_coords)

    return label_bounding_boxes


def compute_fiber_bounding_boxes(fibers, affine_transform):
    bounding_boxes = np.empty((len(fibers), 6), dtype=float)
    linear_component = affine_transform[:3, :3]
    translation = affine_transform[:-1, -1]

    for i, tract in enumerate(fibers):
        ras_coords = (
            (linear_component.dot(tract.T).T +
             translation)
        )

        bounding_boxes[i] = BoundingBox(ras_coords)

    box_array = np.empty(
        len(fibers),
        dtype=[(name, float) for name in (
            'left', 'posterior', 'inferior',
            'right', 'anterior', 'superior'
        )])
    bounding_boxes = bounding_boxes.T
    for i, name in enumerate(box_array.dtype.names):
        box_array[name] = bounding_boxes[i]

    return box_array


def compute_label_crossings(fiber_cumulative_lengths, point_labels, threshold):
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


def compute_label_endings(fiber_cumulative_lengths, point_labels):
    fibers_labels = {}
    for i in xrange(len(fiber_cumulative_lengths) - 1):
        start = fiber_cumulative_lengths[i]
        end = fiber_cumulative_lengths[i + 1]
        fibers_labels[i] = set((int(point_labels[start]), int(point_labels[end - 1])))

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

    crossing_fibers_labels, crossing_labels_fibers = compute_label_crossings(
        fiber_cumulative_lengths, point_labels, crossing_threshold
    )

    ending_fibers_labels, ending_labels_fibers = compute_label_endings(
        fiber_cumulative_lengths, point_labels
    )

    return crossing_fibers_labels, crossing_labels_fibers, ending_fibers_labels, ending_labels_fibers

def compute_fiber_occupation_image(affine_ras_2_ijk, img, fibers, length_threshold):
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

    image_points_to_traverse = np.empty(img.shape, dtype=bool)
    coords = tuple(all_points_ijk_rounded.T)
    image_points_to_traverse[coords] = True
    points_to_traverse = np.transpose(image_points_to_traverse.nonzero())

    label_occupation_image = np.empty(img.shape, dtype='object')
    fiber_numbers = np.empty(len(all_points), dtype=int)

    last_index = 0
    for i, fiber in enumerate(fibers):
        n = len(fiber)
        fiber_numbers[last_index: last_index + n] = i
        last_index += n

    label_occupation_image[:] = set()
    for point in points_to_traverse:
        print i
        i, j, k = point
        points_in_fibers = (all_points_ijk_rounded == point).all(1)
        label_occupation_image[i, j, k] = set(
            fiber_numbers[points_in_fibers]
        )

    return label_occupation_image




    point_labels = img[tuple(all_points_ijk_rounded.T)]
    fiber_cumulative_lengths = np.cumsum([0] + [len(f) for f in fibers])

    crossing_fibers_labels, crossing_labels_fibers = compute_label_crossings(
        fiber_cumulative_lengths, point_labels, crossing_threshold
    )

    ending_fibers_labels, ending_labels_fibers = compute_label_endings(
        fiber_cumulative_lengths, point_labels
    )

    return crossing_fibers_labels, crossing_labels_fibers, ending_fibers_labels, ending_labels_fibers


#import collections
#BinaryTreeNode = collections.namedtuple('key value left right')

#class BinaryTreeDict:
#    go_left = 1
#    go_right = 2
#    def __init__(self):
#        self.root = None

#    def __setitem__(self, key, value):
#        new_node = BinaryTreeNode(key, value, None, None)

#        if self.root is None:
#            self.root = new_node
#        else:
#            current_node = self.root
#            while True:
#                node_key = current_node.key
#                if key == node_key:
#                    current_node.value = value
#                    return
#                elif key < node_key:
#                    selected_child_attr = 'left'
#                elif key > node_key:
#                    selected_child_attr = 'right'

#                selected_child = getattr(
#                    current_node,
#                    selected_child_attr
#                )

#                if selected_child is not None:
#                    current_node = selected_child
#                else:
#                    break

#            setattr(current_node, selected_child_attr, new_node)

#    def update_item(self, key, value):
#        new_node = BinaryTreeNode(key, value, None, None)

#        if self.root is None:
#            self.root = new_node
#        else:
#            current_node = self.root
#            while True:
#                node_key = current_node.key
#                if key == node_key:
#                    current_node.value.update(value)
#                    return
#                elif key < node_key:
#                    selected_child_attr = 'left'
#                elif key > node_key:
#                    selected_child_attr = 'right'

#                selected_child = getattr(
#                    current_node,
#                    selected_child_attr
#                )

#                if selected_child is not None:
#                    current_node = selected_child
#                else:
#                    break

#            setattr(current_node, selected_child_attr, new_node)

#    def __getitem__(self, key):
#        current_node = self.root
#        while current_node != None:
#            node_key = current_node.key
#            if key == node_key:
#                return current_node.value
#            elif key < node_key:
#                current_node = current_node.left
#            elif key > node_key:
#                current_node = current_node.right

#        raise KeyError("Key not found")

#    def smaller_items(self, key):
#        current_node = self.root
#        def action(node, key, result):
#            action_result = self.go_left
#            if node.key < key:
#                result.update(node.value)
#                action_result |= self.go_right

#            return action_result

#        while current_node != None:
#            node_key = current_node.key
#            if key == node_key:
#                break
#            elif key < node_key:
#                current_node = current_node.left
#            elif key > node_key:
#                current_node = current_node.right
#        else:
#            raise KeyError("key not found")

#    def walk_tree_(self, node, action, args):
#        if node is not None:
#            action_result = action(node, *args)
#            if action_result & self.go_left > 0:
#                self.walk_tree_(node.left, action)
#            if action_result & self.go_right > 0:
#                self.walk_tree_(node.right, action)



#class SixTreeDict:
#    def __init__(self):
#        pass

#    def __setitem__(self, key, value):
#        if len(key) != 6:
#            raise ValueError("Keys must be 6 components")
