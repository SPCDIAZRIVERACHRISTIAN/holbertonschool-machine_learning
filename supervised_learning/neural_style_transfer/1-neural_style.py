#!/usr/bin/env python3
'''NOTES:
    this class works to transfer styles
    in images it takes images
    grabs the style of one and
    implements it in the other.
'''

import numpy as np
import tensorflow as tf


class NST():
    '''
        class for neural style transfer
    '''

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
        ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        '''initializes neural style transfer

        Args:
            style_image (ndarray): image that has the desired style
            content_image (ndarray): image to be transformed
            alpha (float, optional): weight for the content
                cost. Defaults to 1e4.
            beta (int, optional): weight for the style
                cost. Defaults to 1.
        '''

        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or \
           style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or \
           content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        '''scales the image with values between 0 to 1
            and its largest side is 512 pixels

        Args:
            image (ndarray): image to be scaled

        Raises:
            TypeError: image not of shape [h, w, 3]

        Returns:
            ndarray: scaled image
        '''

        if not isinstance(image, np.ndarray) or \
                image.ndim != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image = tf.image.resize(
            image,
            (h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
            )
        image = tf.expand_dims(image, axis=0)
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image

    def load_model(self):
        """Loads the model for Neural Style Transfer"""
        # Initialize VGG19 as the base model, excluding the
        # top layer (classifier)
        # The model uses the default input size of 224x224 pixels
        base_vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )

        custom_object = {"MaxPooling2D": tf.keras.layers.AveragePooling2D}
        base_vgg.save("base_vgg")

        vgg = tf.keras.models.load_model("base_vgg",
                                         custom_objects=custom_object)

        for layer in vgg.layers:
            layer.trainable = False

        style_outputs = \
            [vgg.get_layer(name).output for name in self.style_layers]

        content_output = vgg.get_layer(self.content_layer).output

        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
