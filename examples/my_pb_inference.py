# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
import os
import sys
import time
import cv2
import glob

import tensorflow as tf
import logging

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = 'keras_retinanet.bin'

from keras_retinanet.utils.visualization import draw_detections, draw_annotations


class TextDetectionParams(object):
    """
    Inference parameters for Text Detection Model.
    """
    MODEL_PATH = '/path/to/pb'
    GPU_MEMORY_FRACTION = 0.3

    # the input image size
    IMAGE_MIN_SIDE  = 480
    IMAGE_MAX_SIDE  = 1024

    # post processing
    SCORE_THRESHOLD = 0.3
    MAX_DETECTIONS  = 100


class TextDetectionInference(object):
    """
    Text detection Inference interface.
    """

    def __init__(self, params, session=None):
        """
        Initialize model inference parameters and load model from pb file.
        :param params:
        :param session:
        """
        self.image_min_side  = params.IMAGE_MIN_SIDE
        self.image_max_side  = params.IMAGE_MAX_SIDE
        self.score_threshold = params.SCORE_THRESHOLD
        self.max_detections  = params.MAX_DETECTIONS

        if session is None:
            self.load_model(params.MODEL_PATH, params.GPU_MEMORY_FRACTION)
        else:
            self.session = session

        logging.info('Module initialize finished...')

    def load_model(self, model_path, gpu_memory_fraction):
        """
        Load pb model file from file.
        """
        logging.info('Load text detection model from pb start .......')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        config.gpu_options.allow_growth = True

        with tf.Graph().as_default():
            with tf.gfile.FastGFile(model_path, 'rb') as f_handle:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f_handle.read())
                tf.import_graph_def(graph_def, name='')

            self.session = tf.Session(config=config)

        logging.info('Load text detection model from pb end .......')

    def resize_image(self, image):
        (rows, cols, _) = image.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        scale = float(self.image_min_side) / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > self.image_max_side:
            scale = self.image_max_side / largest_side
        image = cv2.resize(image, None, fx=scale, fy=scale)
        return image, scale

    def preprocess_image(self, x):
        x = x.astype(np.float32)
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.680
        return x

    def predict(self, image):
        image = self.preprocess_image(image)
        image, scale = self.resize_image(image)
        data = np.array([image], np.float32)

        input_feed = {'input_1:0': data}
        output_fetch = ['output_boxes:0', 'output_scores:0']

        boxes, scores = self.session.run(output_fetch, input_feed)

        # correct boxes for image scale
        boxes /= scale
        indices = np.where(scores[0, :] > self.score_threshold)[0]
        scores = scores[0][indices]
        scores_sort = np.argsort(-scores)[:self.max_detections]

        boxes = boxes[0, indices[scores_sort], :]
        scores = scores[scores_sort]
        return boxes, scores


def main():
    detection_params = TextDetectionParams()
    text_detections  = TextDetectionInference(detection_params)

    image_files = '/path/to/test/images/'
    image_names = glob.glob(os.path.join(image_files, '*.jpg'))
    for image_file in image_names:
        image_data = cv2.imread(image_file)
        boxes, scores = text_detections.predict(image_data)
        print(boxes)

    print('all images in, {} finished.'.format(image_files))


if __name__ == '__main__':
    main()
