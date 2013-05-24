import sys
import numpy as np
from collections import namedtuple

from .aabb import BoundingBox

TractLabelIndices = namedtuple(
    'TractLabelIndices',
    [
        'crossing_tracts_labels',
        'crossing_labels_tracts',
        'ending_tracts_labels',
        'ending_labels_tracts'
    ]
)


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


def compute_tract_bounding_boxes(tracts, affine_transform):
    bounding_boxes = np.empty((len(tracts), 6), dtype=float)
    linear_component = affine_transform[:3, :3]
    translation = affine_transform[:-1, -1]

    for i, tract in enumerate(tracts):
        ras_coords = (
            (linear_component.dot(tract.T).T +
             translation)
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
    for i in xrange(len(tract_cumulative_lengths) - 1):
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
    for i in xrange(len(tract_cumulative_lengths) - 1):
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


def compute_tract_label_indices(
    affine_ras_2_ijk, img,
    tracts, length_threshold, crossing_threshold
):
    if length_threshold > 0:
        tract_length = lambda tract: ((((tract[
                                      1:] - tract[:-1]) ** 2).sum(1)) ** .5).sum()
        tracts = [f for f in tracts if tract_length(f) >= length_threshold]

    all_points = np.vstack(tracts)
    all_points_ijk = (np.dot(affine_ras_2_ijk[:-1, :-1], all_points.T).T +
                      affine_ras_2_ijk[:-1, -1])
    all_points_ijk_rounded = np.round(all_points_ijk).astype(int)

    if any(((all_points_ijk_rounded[:, i] >= img.shape[i]).any() for i in xrange(3))) or (all_points_ijk_rounded < 0).any():
        print >>sys.stderr, "Warning tract points fall outside the image"

    for i in xrange(3):
        all_points_ijk_rounded[:, i] = all_points_ijk_rounded[
            :, i].clip(0, img.shape[i] - 1)

    point_labels = img[tuple(all_points_ijk_rounded.T)]
    tract_cumulative_lengths = np.cumsum([0] + [len(f) for f in tracts])

    crossing_tracts_labels, crossing_labels_tracts = compute_label_crossings(
        tract_cumulative_lengths, point_labels, crossing_threshold
    )

    ending_tracts_labels, ending_labels_tracts = compute_label_endings(
        tract_cumulative_lengths, point_labels
    )

    return TractLabelIndices(
        crossing_tracts_labels,
        crossing_labels_tracts,
        ending_tracts_labels,
        ending_labels_tracts
    )


# import collections
# BinaryTreeNode = collections.namedtuple('key value left right')

# class BinaryTreeDict:
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



# class SixTreeDict:
#    def __init__(self):
#        pass

#    def __setitem__(self, key, value):
#        if len(key) != 6:
#            raise ValueError("Keys must be 6 components")
