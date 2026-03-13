import importlib

spec = importlib.util.spec_from_file_location("imaging_analysis_lib.camera_settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/imaging_analysis_lib/camera_settings.py")
camera_settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera_settings)

spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_training", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_training.py")
SVD_training_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_training_mod)


cam_vert1 = camera_settings.cameras['cam_vert1']
cam_hor1 = camera_settings.cameras['cam_hor1']


try:
    year = 2026
    month = 3
    day = 10
    sequences = [1]
    what_images = ['PTAI_probe','PTAI_m1']
    # Use corners with custom size
    reduce_func_name = 'get_corners_custom'
    reduce_func_params = {'vert_size': 10, 'hor_size': 700}
    SVD_training_mod.SVD_training_save(year,month,day,sequences,cam_vert1,what_images=what_images,reduce_func_name=reduce_func_name,reduce_func_params=reduce_func_params)
except Exception as e:
    raise(e)

