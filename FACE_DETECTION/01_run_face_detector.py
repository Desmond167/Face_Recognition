from PIL import Image
from detect_face import extract_face
from detector_config import DetectorConfig

import matplotlib.pyplot as plt
import numpy as np

#------- Initialize the CONFIGURE Class -------#
CONFIG = DetectorConfig()

DETECTOR_SETTINGS = CONFIG.configuration()


#------- Run the main Face Detection function -------#
GET_FACE_DETAILS = extract_face(image_path=DETECTOR_SETTINGS.INPUT_IMAGE ,
                output_image_size=DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE ,
                    device=DETECTOR_SETTINGS.DEVICE)

fig, ax = plt.subplots(figsize=(160, 160))

for box, landmark in zip(GET_FACE_DETAILS.BOUNDING_BOX, GET_FACE_DETAILS.LANDMARKS):
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
    # ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
fig.show()

# CROPPED_FACE_IMAGE = Image.fromarray(GET_FACE_DETAILS.BOUNDING_BOX[0])

# if CROPPED_FACE_IMAGE != 'RGB':
#     CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.convert('RGB')

# # ------- Resize pixels to required size ------- #
# CROPPED_FACE_IMAGE = CROPPED_FACE_IMAGE.resize(DETECTOR_SETTINGS.OUTPUT_IMAGE_SIZE)
# CROPPED_FACE_IMAGE.save(DETECTOR_SETTINGS.OUTPUT_IMAGE)
