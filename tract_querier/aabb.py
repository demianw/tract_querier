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
