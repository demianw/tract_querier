import numpy as np

__all__ = ['Tractography']


class Tractography:

    r"""
    Class to represent a tractography dataset

    Parameters
    ----------
    tracts : list of float array :math:`N_i\times 3`
        Each element of the list is a tract represented as point array,
        the length of the i-th tract is :math:`N_i`
    tracts_data : dict of <data name>= list of float array of :math:`N_i\times M`
        Each element in the list corresponds to a tract,
        :math:`N_i` is the length of the i-th tract and M is the
        number of components of that data type.
    validate : bool
        Check that tracts and tracts_data are valid
    """

    def __init__(self, tracts=None, tracts_data=None, validate=True, **kwargs):
        if tracts is not None and tracts_data is None:
            tracts_data = {}
        self._tracts = []
        self._quantity_of_points_per_tract = None

        self._tract_map = None
        self._subsampled_tracts = None
        self._subsampled_data = None

        self._extra_args = []
        for k, v in kwargs.items():
            if k[0] != '_':
                setattr(self, k, v)
                self._extra_args.append(k)

        if tracts is not None:
            self.append(tracts, tracts_data, validate=validate)

    @property
    def extra_args(self):
        ret = {}
        for k in self._extra_args:
            ret[k] = getattr(self, k)
        return ret

    def append(self, tracts, tracts_data=None, validate=True):
        r"""
        Append tracts and corresponding data to the current set

        Parameters
        ----------
        tracts : list of float array :math:`N_i\times 3`
            Each element of the list is a tract represented as point array,
            the length of the i-th tract is :math:`N_i`
        tracts_data : dict of <data name>= list of float array of :math:`N_i\times M`
            Each element in the list corresponds to a tract,
            :math:`N_i` is the length of the i-th tract and M is the
            number of components of that data type.
        validate : bool
            Check that tracts and tracts_data are valid
        """
        if tracts_data is None:
            tracts_data = {}

        if len(self._tracts) == 0:
            self._tracts = tracts
            self._tracts_data = tracts_data
            appending = False
        else:
            appending = True

        if validate:
            if tracts is not None:
                try:
                    if any(
                        not (
                            t.ndim == 2 and
                            t.shape[1] == 3
                        ) for t in tracts
                    ):
                        raise ValueError(
                            'First argument is not a list of tracts')
                except AttributeError:
                    raise ValueError('First argument is not a list of tracts')

                if tracts_data is not None and hasattr(tracts_data, 'iteritems'):
                    for k, v in tracts_data.iteritems():
                        if isinstance(v, str):
                            continue
                        if len(v) != len(tracts):
                            raise ValueError(
                                'Number of elements in attribute %s must '
                                'be the same as the number of tracts' % k
                            )
                        _, M = v[0].shape
                        for i, tract_v in enumerate(v):
                            N, tract_M = tract_v.shape
                            if (
                                (N != len(tracts[i])) or
                                (tract_M != M)
                            ):
                                raise ValueError(
                                    "Data for tract %s: %d is inconsistent" % (
                                        k, i)
                                )
        if appending:
            if tracts_data.keys() != self._tracts_data.keys():
                raise ValueError("Tract data to append not compatible")
            if any(
                self._tracts_data[k][0].shape[1] != v[0].shape[1]
                for k, v in tracts_data.iteritems()
                if not isinstance(v, str)
            ):
                raise ValueError("Tract data to append not compatible")

            for k, v in tracts_data.iteritems():
                self._tracts_data[k] += v

            self._tracts += tracts

            if self.are_tracts_subsampled():
                self.subsample_tracts(self._quantity_of_points_per_tract)
            if self.are_tracts_filtered():
                self.filter_tracts(self._criterium)

    def unsubsample_tracts(self):
        r"""
        Reset any subsampling applied to the tracts
        """
        self._subsampled_tracts = None
        self._subsampled_data = None

    def unfilter_tracts(self):
        r"""
        Reset any filtering applied to the tracts
        """

        self._tract_map = None

    def subsample_tracts(self, points_per_tract):
        r"""
        Subsample the tracts in the dataset to a maximum number of
        points per tract

        Parameters
        ----------
        points_per_tract: int
            Maximum number of points per tract after the operation
            is executed
        """
        self._quantity_of_points_per_tract = points_per_tract
        self._subsampled_tracts = []
        self._subsampled_data = {}

        for k in self._tracts_data:
            self._subsampled_data[k] = []

        for i in xrange(len(self._tracts)):
            f = self._tracts[i]
            s = np.linspace(
                0,
                f.shape[0] - 1,
                min(f.shape[0], self._quantity_of_points_per_tract)
            ).round().astype(int)

            self._subsampled_tracts.append(f[s, :])

            for k, v in self._tracts_data.iteritems():
                if not isinstance(v, str):
                    self._subsampled_data[k].append(v[i][s])

        self._interpolated = False

    def filter_tracts(self, criterium):
        r"""
        Filter the tracts in the set according to a criterium function

        Parameters
        ----------

        criterium : function of array :math:`N\times 3` -> Bool
            A function taking a tract as an array of
            3D points and returning True or False with
            specifying if it should be included
        """
        if len(self._subsampled_tracts) > 0:
            tracts = self._subsampled_tracts
            data = self._subsampled_data
        else:
            tracts = self._tracts
            data = self._data

        self._tract_map = filter(
            lambda i: criterium(tracts),
            xrange(len(tracts))
        )

        self._filtered_tracts = [tracts[i] for i in self._tract_map]
        self._filtered_data = {}
        for k, v in data.iteritems():
            self._filtered_data[k] = [
                v[i] for i in self._tract_map
            ]

        self._criterium = criterium

    def are_tracts_filtered(self):
        return self._tract_map is not None

    def are_tracts_subsampled(self):
        return self._subsampled_tracts is not None

    def original_tracts(self):
        r"""
        Tract set used to original construct this
        tractography object, no subsampling or filtering
        applied

        Returns
        -------
        tracts : list of float array :math:`N_i\times3`
            Each element of the list is a tract represented as point array,
            the length of the i-th tract is :math:`N_i`
        """
        return self._tracts

    def original_tracts_data(self):
        r"""
        Tract data contained of the original dataset of this tractography object

        Returns
        -------
        tract data : dict of <data name>= list of float array of :math:`N_i\times M`
                     Each element in the list corresponds to a tract,
                     :math:`N_i` is the length of the i-th tract and M is the
                     number of components of that data type.
        """
        return self._tracts_data

    def filtered_tracts_map(self):
        r"""
        Tract indices included after the filtering

        Returns
        -------
        List of tract indices included after the filtering
        """
        return self._tract_map

    def tracts(self):
        r"""
        Tracts contained in this tractography object after filtering and
        subsampling if these operations have been applied

        Returns
        -------
        tracts : list of float array :math:`N_i\times 3`
            Each element of the list is a tract represented as point array,
            the length of the i-th tract is :math:`N_i`
        """
        if self._tract_map is not None:
            return self._filtered_tracts
        elif self._subsampled_tracts is not None:
            return self._subsampled_tracts
        else:
            return self._tracts

    def tracts_data(self):
        r"""
        Tract data contained in this tractography object after filtering and
        subsampling if these operations have been applied

        Returns
        -------
        tract data : dict of <data name>= list of float array of :math:`N_i\times M`
                     Each element in the list corresponds to a tract,
                     :math:`N_i` is the length of the i-th tract and M is the
                     number of components of that data type.
        """
        if self._tract_map is not None:
            return self._filtered_data
        elif self._subsampled_data is not None:
            return self._subsampled_data
        else:
            return self._tracts_data

    def add_tract_data_from_array(self, name, array):
        r"""
        Add a new data element reproducing a constant data
        value for each of the :math:`$M$` tracts.

        After execution, the tract data will have a new set
        original_tracts_data()[name][i][:] == array[i]

        Parameters
        ----------
        name : str
            Name of the new data element
        array : array of length :math:`$M$`
            Data value for each tract
       """
        data = [
            np.ones((len(self.original_tracts()[i]), 1)) * array[i]
            for i in xrange(len(self.tracts()))
        ]

        self.original_tracts_data()[name] = data

        if self._subsampled_tracts is not None:
            self.subsample_tracts(self._quantity_of_points_per_tract)
