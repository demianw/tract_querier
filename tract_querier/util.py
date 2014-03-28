class LabelsBundles:

    def __init__(self):
        self.key_tracts_value_labels = {}
        self.key_labels_value_tracts = {}

    def append_tract(self, tract, labels):
        if tract not in self.key_tracts_value_labels:
            self.key_tracts_value_labels[tract] = set(labels)
        else:
            self.key_tracts_value_labels[tract].update(labels)

        for label in labels:
            self.key_labels_value_tracts[label].append(tract)

    def append_label(self, label, tracts):
        if label not in self.key_labels_value_tracts:
            self.key_labels_value_tracts[label] = set(tracts)
        else:
            self.key_labels_value_tracts[label].update(tracts)

        for tract in tracts:
            self.key_tracts_value_labels[tract].append(label)

    def get_tract(self, tract):
        return self.key_tracts_value_labels[tract]

    def get_labels(self, label):
        return self.key_labels_value_tracts[label]
