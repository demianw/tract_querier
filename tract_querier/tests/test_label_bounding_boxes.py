import numpy as np

from ..tract_label_indices import compute_label_bounding_boxes


def test_label_bounding_box_spans_voxel_centres():
    image = np.zeros((6, 6, 6))
    image[2:4, 2:4, 2:4] = 1
    affine = np.diag([-2., 2., 2., 1.])
    bounding_boxes = compute_label_bounding_boxes(image, affine)
    assert list(bounding_boxes) == [1]
    np.testing.assert_array_equal(bounding_boxes[1], [-6, 4, 4, -4, 6, 6])


def test_label_bounding_boxes_match_voxel_coordinates():
    rng = np.random.default_rng(0)
    image = rng.integers(-2, 6, size=(7, 8, 9))
    image[image == 4] = 0
    affine = np.array([
        [0., -1.5, .3, 10.],
        [2., 0., 0., -5.],
        [.2, 0., 1., 3.],
        [0., 0., 0., 1.],
    ])
    bounding_boxes = compute_label_bounding_boxes(image, affine)
    assert set(bounding_boxes) == {-2, -1, 1, 2, 3, 5}
    for label, bounding_box in bounding_boxes.items():
        ijk = np.argwhere(image == label)
        ras = ijk @ affine[:3, :3].T + affine[:3, 3]
        np.testing.assert_allclose(
            bounding_box, np.r_[ras.min(0), ras.max(0)]
        )


def test_no_labels():
    assert compute_label_bounding_boxes(np.zeros((3, 3, 3)), np.eye(4)) == {}
