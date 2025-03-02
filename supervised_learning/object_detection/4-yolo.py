#!/usr/bin/env python3
'''NOTES:
    This class only initializes
    objects to perform object
    detection
'''

import os
import numpy as np
import tensorflow.keras as K  # type: ignore
from keras.activations import sigmoid
import cv2


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

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs: list of numpy.ndarrays containing the predictions from the
                Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width: the height and width of the
                    grid used
                        for the output
                    anchor_boxes: the number of anchor boxes used
                    4: (t_x, t_y, t_w, t_h)
                    1: box_confidence
                    classes: class probabilities for all classes
            image_size: numpy.ndarray containing the image’s original size
                [image_height, image_width]
        Returns: tuple of (boxes, box_confidences, box_class_probs):
            boxes: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, 4) containing the processed boundary boxes
                for each
                output, respectively:
                4: (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                    original image
            box_confidences: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, 1) containing the box confidences for
                each output,
                respectively
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, classes) containing the box’s class
                probabilities for
                each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box = output[..., :4]

            t_x = box[..., 0]
            t_y = box[..., 1]
            t_w = box[..., 2]
            t_h = box[..., 3]

            c_x = np.arange(grid_width).reshape(1, grid_width)

            c_x = np.repeat(c_x, grid_height, axis=0)

            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)

            c_y = np.arange(grid_width).reshape(1, grid_width)

            c_y = np.repeat(c_y, grid_height, axis=0).T

            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)

            b_x = (sigmoid(t_x) + c_x) / grid_width
            b_y = (sigmoid(t_y) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            x1 = (b_x - b_w / 2)
            y1 = (b_y - b_h / 2)

            x2 = (b_x + b_w / 2)
            y2 = (b_y + b_h / 2)

            x1 = x1 * image_size[1]
            y1 = y1 * image_size[0]
            x2 = x2 * image_size[1]
            y2 = y2 * image_size[0]

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

            box_confidence = output[..., 4:5]
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_confidences.append(box_confidence)

            box_class_prob = output[..., 5:]
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        '''filters boxes and returns them filtered by the threshold

        Args:
            boxes (nd_array): of shape (grid_height,
                grid_width, anchor_boxes, 4)
                containing the processed boundary boxes for
                each output, respectively
            box_confidences (nd_array): containing the processed box
                confidences for each output, respectively
            box_class_probs (nd_array):  containing the processed box
                class probabilities for each output, respectively

        Returns:
           tuple : returns the bounding boxes filtered by the threshold
            class number of each box and the box score for each box
        '''

        filtered_boxes_all = []
        box_classes_all = []
        box_scores_all = []

        for box, conf, prob in zip(boxes, box_confidences, box_class_probs):
            box_scores = conf * prob

            max_scores = np.max(box_scores, axis=-1)
            max_classes = np.argmax(box_scores, axis=-1)

            max_scores = max_scores.reshape(-1)
            max_classes = max_classes.reshape(-1)
            box = box.reshape(-1, 4)

            keep_indices = np.where(max_scores >= self.class_t)

            filtered_boxes_all.append(box[keep_indices])
            box_classes_all.append(max_classes[keep_indices])
            box_scores_all.append(max_scores[keep_indices])

        filtered_boxes_all = np.concatenate(filtered_boxes_all, axis=0)
        box_classes_all = np.concatenate(box_classes_all, axis=0)
        box_scores_all = np.concatenate(box_scores_all, axis=0)

        return filtered_boxes_all, box_classes_all, box_scores_all

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max suppression
            Args:
                filtered_boxes: numpy.ndarray of shape (?, 4) containing all of
                the filtered bounding boxes:
                    4: (x1, y1, x2, y2)
                box_classes: numpy.ndarray of shape (?,) containing the class
                number for the class that filtered_boxes predicts
                box_scores: numpy.ndarray of shape (?) containing the
                box scores
                for each box in filtered_boxes
            Returns: tuple of (box_predictions, predicted_box_classes,
            """
        # Initialize lists to hold the final predictions,
        # their classes, and scores
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Iterate over each unique class found in box_classes
        for box_class in np.unique(box_classes):
            # Find indices of all boxes belonging to the current class
            indices = np.where(box_classes == box_class)[0]

            # Extract subsets for the current class
            filtered_boxes_subset = filtered_boxes[indices]
            box_classes_subset = box_classes[indices]
            box_scores_subset = box_scores[indices]

            # Calculate the area of each box in the subset
            x1 = filtered_boxes_subset[:, 0]
            y1 = filtered_boxes_subset[:, 1]
            x2 = filtered_boxes_subset[:, 2]
            y2 = filtered_boxes_subset[:, 3]
            box_areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            # Sort boxes by their scores in descending order
            ranked = np.argsort(box_scores_subset)[::-1]

            # Initialize a list to keep track of boxes that
            # pass the suppression
            pick = []

            # Continue until all boxes are either picked or suppressed
            while ranked.size > 0:
                # Always pick the first box in the ranked list
                pick.append(ranked[0])

                # Compute the intersection over union (IOU) between
                # the picked box and all other boxes
                xx1 = np.maximum(x1[ranked[0]], x1[ranked[1:]])
                yy1 = np.maximum(y1[ranked[0]], y1[ranked[1:]])
                xx2 = np.minimum(x2[ranked[0]], x2[ranked[1:]])
                yy2 = np.minimum(y2[ranked[0]], y2[ranked[1:]])
                inter_areas = np.maximum(0, xx2 - xx1 + 1) * np.maximum(
                    0, yy2 - yy1 + 1)
                union_areas = box_areas[ranked[0]] + box_areas[
                    ranked[1:]] - inter_areas
                IOU = inter_areas / union_areas

                # Keep only boxes with IOU below the threshold
                updated_indices = np.where(IOU <= self.nms_t)[0]
                ranked = ranked[updated_indices + 1]

            # Update the final lists with the picks for this class
            pick = np.array(pick)
            box_predictions.append(filtered_boxes_subset[pick])
            predicted_box_classes.append(box_classes_subset[pick])
            predicted_box_scores.append(box_scores_subset[pick])

        # Concatenate the lists into final arrays
        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
            Loads all images from a given folder.

            Args:
                folder_path (str): path to the folder holding the images

            Returns:
                tuple (images, image_paths):
                  - images is a list of all images loaded as numpy.ndarrays
                  - image_paths is a list of the actual file paths
        """

        images = []
        images_path = []

        for file_name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file_name)

            if os.path.isfile(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    images.append(img)
                    images_path.append(full_path)

        return images, images_path
