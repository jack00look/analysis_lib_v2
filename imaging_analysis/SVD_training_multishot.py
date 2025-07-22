import importlib

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_training", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_training.py")
SVD_training_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_training_mod)


cam_vert1 = camera_settings.cam_vert1
cam_hor1 = camera_settings.cam_hor1


try:
    year = 2024
    month = 12
    day = 17
    sequences = [9,10,11,12,13,14,15,16]
    what_images = ['PTAI_probe']
    SVD_training_mod.SVD_training_save(year,month,day,sequences,cam_vert1,what_images=what_images)
except Exception as e:
    raise(e)

