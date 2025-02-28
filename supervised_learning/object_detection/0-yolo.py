#!/usr/bin/env python3
'''NOTES:
    This class only initializes
    objects to perform object
    detection
'''

import tensorflow.keras as K #type: ignore


class Yolo:
    '''This class initializes objects to
    perform object detection.
    '''

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo model"""

        self.model = K.models.load_model(model_path)
        self.class_names = self._load_class_names(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_class_names(self, classes_path):
        """Load class names from a file"""

        with open(classes_path, 'r') as file:
            class_names = file.read().splitlines()
        return class_names
