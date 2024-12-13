import importlib
import os

spec = importlib.util.spec_from_file_location("general_lib.settings", "/home/rick/labscript-suite/userlib/analysislib/analysislib_v2/general_lib/settings.py")
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)

def delete_allprobes_full(path = settings.bec2path):
    with open(path + '/SVD_allprobefiles_list.txt', 'r') as file:
    # Read each line in the file
        for line in file:
        # Print each line
            os.remove(line[:-1])
    with open(path + '/SVD_allprobefiles_list.txt', 'w') as file:
        file.write('')