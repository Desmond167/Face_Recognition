#!/usr/bin/env python
from collections import namedtuple


OUTPUT_IMAGE_SIZE = (160,160)

DEVICE = "cuda"

class DetectorConfig:

    def configuration(self):
        settings_dict = {
            "INPUT_IMAGE" : "../sharon.jpg",
            "OUTPUT_IMAGE_SIZE" : OUTPUT_IMAGE_SIZE,
            "DEVICE" : DEVICE
        }

        SETTINGS = namedtuple("SETTINGS", settings_dict.keys())(*settings_dict.values())

        return SETTINGS