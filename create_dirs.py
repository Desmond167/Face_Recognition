#!/usr/bin/env python
from pathlib import Path
import os 

#_______ THE CURRENT PROJECT WORKING DIRECTORY _______#
ROOT_DIR = str(Path(__file__).resolve().parent.parent).replace("\\","/")
FOLDER_STRUCTURE = "/"
BASE_DIR = "{ROOT_DIR}{FOLDER_STRUCTURE}".format(ROOT_DIR=ROOT_DIR,
                                                FOLDER_STRUCTURE=FOLDER_STRUCTURE)

##########################################################################################################
########################################## CREATING DIRECTORIES ##########################################
##########################################################################################################

#________________ DIRECTORY FOR DETECTED FACE IMAGES ________________#

DETECTED_FACE_FOLDER = "DETECTED_FACES"

DETECTED_FACE_DIRECTORY = os.path.join(BASE_DIR, DETECTED_FACE_DIRECTORY)

try:
    os.makedirs(DETECTED_FACE_DIRECTORY)

except OSError as error:
    print(error)
#_____________________________________________________________________#