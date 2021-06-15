#!/usr/bin/env python
from collections import namedtuple

#------- The Input for the model -------#
INPUT_IMAGE = "../sharon.jpg"

#------- Set the dimension for output image of the model -------#
OUTPUT_IMAGE_SIZE = (160,160)

#------- Select the processing device for the algorithm -------#
DEVICE = "cuda:0"

#------- Boolean to get the bounding box -------#
GET_BOUNDING_BOX = True

#------- Boolean to get the Probability of the result -------#
GET_PROBABILITY = True

#------- Boolean to get the landmarks on the face -------#
GET_LANDMARKS = True

class DetectorConfig:

    def configuration(self):

        #------- Required settings for the MTCNN Model -------#
        settings_dict = {
            "INPUT_IMAGE" : INPUT_IMAGE,
            "OUTPUT_IMAGE_SIZE" : OUTPUT_IMAGE_SIZE,
            "DEVICE" : DEVICE,
            "GET_BOUNDING_BOX" : GET_BOUNDING_BOX,
            "GET_PROBABILITY" : GET_PROBABILITY,
            "GET_LANDMARKS" : GET_LANDMARKS
        }

        #------- Convert Config dict to python object -------#
        SETTINGS = namedtuple("SETTINGS", settings_dict.keys())(*settings_dict.values())

        return SETTINGS