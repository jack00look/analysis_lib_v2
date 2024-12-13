import importlib

import camera_settings
importlib.reload(camera_settings)
from camera_settings import cam_vert1,cam_hor1,cam_hor3

import SVD_analysis.SVD_training
importlib.reload(SVD_analysis.SVD_training)
from SVD_analysis.SVD_training import SVD_training_save


try:
    year = 2024
    month = 12
    day = 4
    sequences = [29]
    what_images = ['PTAI_probe']
    SVD_training_save(year,month,day,sequences,cam_vert1,what_images=what_images)
except Exception as e:
    raise(e)

