#!/usr/bin/env python


OUTPUT_IMAGE_SIZE = (160,160)

DEVICE = "cuda:0"

class DetectorConfig:

    def configuration(self):
        settings = {
            "INPUT_IMAGE" : "../sharon.jpg",
            "OUTPUT_IMAGE_SIZE" : OUTPUT_IMAGE_SIZE,
            "DEVICE" : DEVICE
        }

        return settings