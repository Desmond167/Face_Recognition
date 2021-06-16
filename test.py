import os 
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent).replace('\\','/')
# OUTPUT_IMAGE_FOLDER = "DETECTED_FACES"
# IMAGE_NAME = "sharon.jpg"

# INPUT_IMAGE = "{BASE_DIR}/{IMAGE_NAME}".format(BASE_DIR=BASE_DIR,
#                                                 IMAGE_NAME=IMAGE_NAME)
                                                
# OUTPUT_IMAGE = "{BASE_DIR}/{OUTPUT_IMAGE_FOLDER}/{IMAGE_NAME}".format(
#                                                         BASE_DIR=BASE_DIR,
#                                                             OUTPUT_IMAGE_FOLDER=OUTPUT_IMAGE_FOLDER,                 IMAGE_NAME=IMAGE_NAME)

# print(INPUT_IMAGE)
# print(OUTPUT_IMAGE)
print(BASE_DIR)