class LabelsBundles:
    def __init__(self):
        self.key_fibers_value_labels = {}
        self.key_labels_value_fibers = {}

    def append_fiber(self, fiber, labels):
        if fiber not in self.key_fibers_value_labels:
            self.key_fibers_value_labels[fiber]= set(labels)
        else:
            self.key_fibers_value_labels[fiber].update(labels)

        for label in labels:
            self.key_labels_value_fibers[label].append(fiber)

    def append_label(self, label, fibers):
        if label not in self.key_labels_value_fibers:
            self.key_labels_value_fibers[label]= set(fibers)
        else:
            self.key_labels_value_fibers[label].update(fibers)

        for fiber in fibers:
            self.key_fibers_value_labels[fiber].append(label)


    def get_fiber(self, fiber):
        return self.key_fibers_value_labels[fiber]

    def get_labels(self, label):
        return self.key_labels_value_fibers[label]
