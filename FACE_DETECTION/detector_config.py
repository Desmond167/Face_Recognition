#!/usr/bin/env python
from collections import namedtuple
import os 

#_______ THE CURRENT WORKING DIRECTORY _______#
BASE_DIR = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')

IMAGE_NAME = "sharon.jpg"

INPUT_IMAGE = "{BASE_DIR}/{IMAGE_NAME}".format(BASE_DIR=BASE_DIR,
                                                IMAGE_NAME=IMAGE_NAME)

OUTPUT_IMAGE_FOLDER = "DETECTED_FACES"

OUTPUT_IMAGE = OUTPUT_IMAGE = "{BASE_DIR}/{OUTPUT_IMAGE_FOLDER}/{IMAGE_NAME}".format(
                                                BASE_DIR=BASE_DIR,                                     OUTPUT_IMAGE_FOLDER=OUTPUT_IMAGE_FOLDER,
                                                IMAGE_NAME=IMAGE_NAME)

OUTPUT_IMAGE_SIZE = (160,160)

DEVICE = "cuda:0"

BOOL_GET_BOUNDING_BOX = True

BOOL_GET_PROBABILITY = True

BOOL_GET_LANDMARKS = True

class DetectorConfig:

    def configuration(self):

        #------- Required settings for the MTCNN Model -------#
        settings_dict = {
            "INPUT_IMAGE" : INPUT_IMAGE,
            "OUTPUT_IMAGE" : OUTPUT_IMAGE,
            "OUTPUT_IMAGE_SIZE" : OUTPUT_IMAGE_SIZE,
            "DEVICE" : DEVICE,
            "BOOL_GET_BOUNDING_BOX" : BOOL_GET_BOUNDING_BOX,
            "BOOL_GET_PROBABILITY" : BOOL_GET_PROBABILITY,
            "BOOL_GET_LANDMARKS" : BOOL_GET_LANDMARKS
        }

        #------- Convert Config dict to python object -------#
        SETTINGS = namedtuple("SETTINGS", settings_dict.keys())(*settings_dict.values())

        return SETTINGS