import numpy as np

try:
    from . import fiber_reparametrization as fr
except:
    pass


class Tractography:

    _original_tracts = []
    _original_lines = []
    _original_data = {}
    _tract_data = {}
    _tracts = []
    _tract_map = []
    _subsampled_tracts = []
    _quantity_of_points_per_tract = None
    _interpolated = False

    def __init__(self, **args):
        self._original_tracts = []
        self._tract_data = {}
        self._original_lines = []
        self._original_data = {}
        self._tracts = []
        self._tract_map = []
        self._quantity_of_points_per_tract = None

        self._interpolated = False
        self._fiber_keep_ratio = None

        self._subsampled_tracts = []
        self._subsampled_lines = []
        self._subsampled_data = []

        if isinstance(args[0], dict):
            self.from_dictionary(args[0])
        elif hasattr('len', args[0]) and hasattr('__setitem__', args[0]):
            self._tracts = args[0]
            if any(
                not (
                    hasattr('shape', t) and
                    len(t.shape != 2) and
                    t.shape[1] != 3
                ) for t in self._tracts
            ):
                raise ValueError('First argument is not a list of tracts')


    def from_dictionary(self, dictionary, append=False):
        dictionary_keys = set('lines', 'points', 'numberOfLines')
        if not dictionary_keys.issuperset(dictionary.keys()):
            raise ValueError("Dictionary must have the keys lines and points")

        #Tracts and Lines are the same thing
        if not append:
            self._original_tracts = []
            self._tract_data = {}
            self._original_lines = []
            self._original_data = {}
            self._tracts = []
            self._tract_map = []
            self._quantity_of_points_per_tract = None

        self._interpolated = False
        self._fiber_keep_ratio = 1

        self._subsampled_tracts = []
        self._subsampled_lines = []
        self._subsampled_data = []

        lines = np.asarray(dictionary['lines']).squeeze()
        points = dictionary['points']

        actual_line_index = 0
        number_of_tracts = dictionary['numberOfLines']
        for l in xrange(number_of_tracts):
            self._tracts.append(
                points[
                       lines[
                           actual_line_index + 1:
                           actual_line_index + lines[actual_line_index] + 1
                       ]
                      ]
            )
            self._original_lines.append(
                np.array(
                    lines[
                        actual_line_index + 1:
                        actual_line_index + lines[actual_line_index] + 1],
                    copy=True
                ))
            actual_line_index += lines[actual_line_index] + 1

        if 'pointData' in dictionary:
            point_data_keys = [
                it[0] for it in dictionary['pointData'].items()
                if isinstance(it[1], np.ndarray)
            ]
            if (
                len(self._original_data.keys()) > 0 and
                (self._original_data.keys() != point_data_keys)
            ):
                raise ValueError('PointData not compatible')

            for k in point_data_keys:
                array_data = dictionary['pointData'][k]
                if not k in self._original_data:
                    self._original_data[k] = array_data
                    self._tract_data[k] = [
                        array_data[f]
                        for f in self._original_lines
                    ]
                else:
                    np.vstack(self._original_data[k])
                    self._tract_data[k].extend(
                        [
                            array_data[f]
                            for f in self._original_lines[-number_of_tracts:]
                        ]
                    )

        self._original_tracts = list(self._tracts)

        if self._quantity_of_points_per_tract != None:
            self.subsample_tracts(self._quantity_of_points_per_tract)

    def unsubsample_tracts(self):
        self._subsampled_tracts = []

    def unfilter_tracts(self):
        self._tract_map = []

    def subsample_tracts(self, quality_of_points_per_tract):
        self._quantity_of_points_per_tract = quality_of_points_per_tract
        self._subsampled_tracts = []
        self._subsampled_lines = []
        self._subsampled_data = {}

        for k in self._tract_data:
            self._subsampled_data[k] = []

        for i in xrange(len(self._tracts)):
            f = self._tracts[i]
            s = np.linspace(
                0,
                f.shape[0] - 1,
                min(f.shape[0], self._quantity_of_points_per_tract)
            ).round().astype(int)

            self._subsampled_tracts.append(f[s, :])
            self._subsampled_lines.append(s)

            for k in self._tract_data:
                self._subsampled_data[k].append(self._tract_data[k][i][s])

        self._interpolated = False

    def subsample_tracts_equal_step(self, step_ratio):
        self._step_ratio = step_ratio
        self._subsampled_tracts = []
        self._subsampled_lines = []
        self._subsampled_data = {}

        for k in self._tract_data:
            self._subsampled_data[k] = []

        for i in xrange(len(self._tracts)):
            f = self._tracts[i]
            s = np.arange(0, len(f), step_ratio)
            self._subsampled_tracts.append(f[s, :])
            self._subsampled_lines.append(s)
            for k in self._tract_data:
                self._subsampled_data[k].append(self._tract_data[k][i][s])

        self._interpolated = False

    def subsample_interpolated_tracts(self, step):
        self._quantity_of_points_per_tract = step
        self._subsampled_tracts = []
        for i, f in enumerate(self._tracts):
            reparametrized_fiber = fr.arc_length_fiber_parametrization(
                f, step=step
            )
            if f is not None:
                self._subsampled_tracts.append(reparametrized_fiber[1:].T)
        self._interpolated = True

    def filter_tracts(self, min_num_of_samples):
        if len(self._original_tracts) == 0:
            self._original_tracts = self._tracts

        self._tracts = filter(
            lambda f: f.shape[0] >= min_num_of_samples,
            self._original_tracts
        )
        self._tract_map = filter(
            lambda i: self._original_tracts[i].shape[0] >= min_num_of_samples,
            xrange(len(self._original_tracts))
        )

        if self._quantity_of_points_per_tract != None:
            if self._interpolated:
                self.subsample_interpolated_tracts(
                    self._quantity_of_points_per_tract
                )
            else:
                self.subsample_tracts(self._quantity_of_points_per_tract)

    def are_tracts_filtered(self):
        return self._tract_map != []

    def are_tracts_subsampled(self):
        return self._subsampled_tracts != []

    def are_subsampled_tracts_interpolated(self):
        return self._interpolated

    def original_tracts_data(self):
        return self._tract_data

    def original_tracts(self):
        return self._original_tracts

    def original_lines(self):
        return self._original_lines

    def original_data(self):
        return self._original_data

    def filtered_tracts_map(self):
        return self._tract_map

    def tracts_to_process(self):
        if self._subsampled_tracts != []:
            return self._subsampled_tracts
        elif self._tracts != []:
            return self._tracts

    def tracts_data_to_process(self):
        if self._subsampled_data != []:
            return self._subsampled_data
        elif self._tract_data != []:
            return self._tract_data
            return self._tract_data
