import importlib
import numpy as np
import h5py
from pathlib import Path
import json
import inspect



spec = importlib.util.spec_from_file_location("SVD_analysis.SVD_analysis_main", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/SVD_analysis/SVD_analysis_main.py")
SVD_main_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SVD_main_mod)

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)


def SVD_training_save(year,month,day,sequences,camera,NAS_path=settings.bec2path,what_images=None,reduce_func_name='get_two_lateral_regions',reduce_func_params=None):
    invU, invS, Vh, all_probes_full,shape_x,shape_y,filenames_list,keys_list,reduce_image_func,reduce_func_name,reduce_func_params = SVD_main_mod.SVD_training(year,month,day,sequences,camera,NAS_path,what_images,reduce_func_name=reduce_func_name,reduce_func_params=reduce_func_params)
    save_path_NAS = SVD_main_mod.make_SVD_savepath(year,month,day,NAS_path)
    infos_txt = 'year: ' + str(year) + '\n' \
                + 'month: ' + str(month) + '\n' \
                + 'day: ' + str(day) + '\n' \
                + 'sequences: ' + str(sequences) + '\n' \
                + 'camera: ' + camera['name'] + '\n' \
                + 'ROI:' + (str(shape_y) + 'x' + str(shape_x))

    infos_json = {'year': year,
                  'month': month,
                  'day': day,
                  'sequences': sequences,
                  'camera': camera['name'],
                  'ROI': (shape_y,shape_x),
                  'reduce_func_name': reduce_func_name,
                  'reduce_func_params': reduce_func_params}

    with open(save_path_NAS / 'infos.txt', 'w') as f:
        f.write(infos_txt)
    with open(save_path_NAS / 'infos.json', 'w') as f:
        json.dump(infos_json, f)
    
    save_path_NAS_func = save_path_NAS / 'reduce_image_module'
    save_path_NAS_func.mkdir()
    
    # Save the reduce function based on what was used
    if reduce_func_name == 'get_corners_custom' and reduce_func_params is not None:
        # For custom corners, save with parameters
        vert_size = reduce_func_params.get('vert_size', 20)
        hor_size = reduce_func_params.get('hor_size', 100)
        function_code = f'''import numpy as np

def reduce_image_func(image):
    """Extract the four corners of the image with configurable size."""
    vert_size = {vert_size}
    hor_size = {hor_size}
    image_cut = np.zeros((2*vert_size, 2*hor_size))
    # Top-left corner
    image_cut[:vert_size, :hor_size] = image[:vert_size, :hor_size]
    # Top-right corner
    image_cut[:vert_size, hor_size:] = image[:vert_size, -hor_size:]
    # Bottom-left corner
    image_cut[vert_size:, :hor_size] = image[-vert_size:, :hor_size]
    # Bottom-right corner
    image_cut[vert_size:, hor_size:] = image[-vert_size:, -hor_size:]
    return image_cut
'''
    else:
        # For other functions, extract the source code
        try:
            function_original = inspect.getsource(reduce_image_func)
            first_parenthesis_position = function_original.find('(')
            function_code = function_original[:4] + 'reduce_image_func' + function_original[first_parenthesis_position:]
            function_code = 'import numpy as np\n' + function_code
        except:
            print(f"Warning: Could not extract source for {reduce_func_name}, using fallback")
            function_code = 'import numpy as np\n\ndef reduce_image_func(image):\n    return image\n'
    
    with open(save_path_NAS_func / 'reduce_image_func.py','w') as f:
        f.write(function_code)
    with open(save_path_NAS_func / '__init__.py','w') as f:
        f.write('')
    
    np.save(save_path_NAS / 'invU.npy', invU)
    np.save(save_path_NAS / 'invS.npy', invS)
    np.save(save_path_NAS / 'Vh.npy', Vh)
    np.save(save_path_NAS / 'filename_list.npy', filenames_list)
    np.save(save_path_NAS / 'keys_list.npy', keys_list)
    np.save(save_path_NAS / 'all_probes_full.npy', all_probes_full)
    with open(NAS_path + '/SVD_allprobefiles_list.txt', 'a') as f:
        f.write(str(save_path_NAS / 'all_probes_full.npy') + '\n')
