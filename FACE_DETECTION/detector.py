from PIL import Image
from facenet_pytorch import MTCNN
from detector_config import DetectorConfig
from collections import namedtuple
from matplotlib import pyplot
import numpy as np


class DetectFace:
    def __init__(self, input_image, device, output_image, output_image_size):
        self.INPUT_IMAGE = input_image
        self.DEVICE = device
        self.OUTPUT_IMAGE = output_image
        self.OUTPUT_IMAGE_SIZE = output_image_size
        self.INPUT_IMAGE_ARRAY = []

    ################ Extract a face from a given photograph ################
    def run_MTCNN(self, bool_get_bounding_box=True, bool_get_probability=True, bool_get_landmarks=True):

        BOUNDING_BOX = []
        PROBABILITY = []
        LANDMARKS = []

        #------- Initialize the MTCNN Model using default weights -------
        MTCNN_MODEL = MTCNN(keep_all=True, device=self.DEVICE)

        # _______ Load the Image _______ #
        LOADED_INPUT_IMAGE = Image.open(self.INPUT_IMAGE)

        #_______ Convert image to Numpy Array _______#
        self.INPUT_IMAGE_ARRAY = np.array(LOADED_INPUT_IMAGE)
        
        #________________ Get the bounding box co-ordinates, Probability and Landmarks on the face________________#
        boxes, prob, landmarks = MTCNN_MODEL.detect(LOADED_INPUT_IMAGE, landmarks=bool_get_landmarks)

        #_______ Check if GET_BOUNDING_BOX is True in Config file _______#
        if bool_get_bounding_box == True:
            BOUNDING_BOX = boxes

        #_______ Check if GET_PROBABILITY is True in Config file _______#
        if bool_get_probability == True:
            PROBABILITY = prob

        #_______ Check if GET_LANDMARKS is True in Config file _______#
        if bool_get_landmarks == True:
            LANDMARKS = landmarks

        #________________Face Dict containing Bounding Box, Probability and Landmarks________________#
        face_dict ={
                "BOUNDING_BOX" : BOUNDING_BOX,
                "PROBABILITY" : PROBABILITY,
                "LANDMARKS" : LANDMARKS
            }

        #________________Convert Face Info dict to python object________________#
        FACE_DETAILS = namedtuple("FACE_DETAILS", face_dict.keys())(*face_dict.values())

        print("========================= DETECTED FACIAL DETAILS SUCCESSFULLY =========================")


        return FACE_DETAILS

    ################ Extract a face from a given photograph ################
    def get_face(self, bounding_box):

        for box in bounding_box:
            COORDINATE_X1 = int(box[0])
            COORDINATE_X2 = int(COORDINATE_X1 + box[2])
            COORDINATE_Y1 = int(box[1])
            COORDINATE_Y2 = int(COORDINATE_Y1 + box[3])

            CROPPED_FACE_IMAGE = self.INPUT_IMAGE_ARRAY[COORDINATE_Y1:COORDINATE_Y2, COORDINATE_X1:COORDINATE_X2]
            CROPPED_FACE_IMAGE = Image.fromarray(CROPPED_FACE_IMAGE)

        if CROPPED_FACE_IMAGE != 'RGB':
            CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.convert('RGB')

        #________________Resize pixels to required size________________#
        CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.resize(self.OUTPUT_IMAGE_SIZE)

        #________________Save the cropped face image________________#
        CROPPED_FACE_IMAGE.save(self.OUTPUT_IMAGE)

        print("========================= DETECTED FACE IMAGE SAVED =========================")