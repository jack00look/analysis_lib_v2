import h5py
from pathlib import Path
import numpy as np
import datetime
import pandas as pd
import os



# save dataset in h5file, if dataset already exists, it overwrites it
def save_data(h5file, dataset_name,data, comment = None):
    if dataset_name in h5file:
        h5file[dataset_name][...] = data
    else:
        h5file.create_dataset(dataset_name, data=data)
    if comment is not None:
        h5file[dataset_name].attrs['comment'] = comment
    return

# return Path object (pathlib) of a given h5 file
# bec2path is the path to the bec2 folder containing the years folders of the sequence files

def get_h5file_infos(filename):
    try:
        NAS_folder = 'NAS542_dataBEC2'
        filename = filename.split(NAS_folder)[1]
        year = (filename[1:5])
        month = (filename[6:8])
        day = (filename[9:11])
        sequence = (filename[12:16])
        filename = filename[28:]
        if 'rep' in filename:
            rep = (filename[-8:-3])
            it = (filename[-16:-12])
            program_name = filename[:-17]
        else:
            rep = 'None'
            it = (filename[-7:-3])
            program_name = filename[:-8]
        dict = {'year':year,'month':month,'day':day,'seq':sequence,'it':it,'rep':rep,'program_name':program_name}
        return dict
    except Exception as e:
        print("Error in get_h5file_infos_from_filename: " + str(e))
        return
    
def get_title(h5file,cam):
    detuning = h5file['globals'].attrs['probe_detuning_debug']/9.7946
    tof = h5file['globals'].attrs['tof_debug']
    title = 'CAM' + cam['name']
    title += '\nDetuning. {:.2f} [Gamma], TOF: {:.2f} ms'.format(detuning,tof)
    h5file_infos = get_h5file_infos(h5file.filename)
    title += '\n' + h5file_infos['year'] + '/' + h5file_infos['month'] + '/' + h5file_infos['day']
    title += '\n' + 'seq:' + h5file_infos['seq'] + '   ' + 'it:' + h5file_infos['it'] + '   ' + 'rep:' + h5file_infos['rep']
    title += '\n' + h5file_infos['program_name']
    return title

def get_h5filename(year, month, day, sequence, it, bec2_path, rep=0,show_errs = False):
    try:
        path = Path(bec2_path)
        if not(path.exists()):
            print("get_h5file: main_path does not exist (main_path = " + str(bec2_path) + ")")
            return
        path = path / str(year)
        if not(path.exists()):
            print("get_h5file: Year folder does not exist")
            return
        path = path / str(month).zfill(2)
        if not(path.exists()):
            print("get_h5file: Month folder does not exist")
            return
        path = path / str(day).zfill(2)
        if not(path.exists()):
            print("get_h5file: Day folder does not exist")
            return
        path = path / str(sequence).zfill(4)
        if not(path.exists()):
            print("get_h5file: Sequence folder does not exist")
            return
        it = str(it).zfill(4)
        for path_temp in path.iterdir():
            path_temp = str(path_temp)
            if path_temp[-2:] == 'h5':
                if rep == 0:
                    if path_temp[-7:-3] == it:
                        return Path(path_temp)
                else:
                    rep = str(rep).zfill(5)
                    if path_temp[-16:-12] == it and path_temp[-8:-3] == rep:
                        return Path(path_temp)
        print("File h5 does not exist in the sequence folder")
        return
    except Exception as e:
        if show_errs:
            raise("Error in get_h5file: " + str(e))
        return

# gets Path objects ordered by iterations (01,02,...) and by reps (rep0001,....), with reps at the end of the sequence
# can decide to omit reps
def get_ordered_sequence(year,month,day,sequence,bec2_path,show_reps = True,show_errs = False):
    
    try:
        path = Path(bec2_path)
        if not(path.exists()):
            print("get_h5file: main_path does not exist (main_path = " + str(bec2_path) + ")")
            return
        path = path / str(year)
        if not(path.exists()):
            print("get_h5file: Year folder does not exist")
            return
        path = path / str(month).zfill(2)
        if not(path.exists()):
            print("get_h5file: Month folder does not exist")
            return
        path = path / str(day).zfill(2)
        if not(path.exists()):
            print("get_h5file: Day folder does not exist")
            return
        path = path / str(sequence).zfill(4)
        if not(path.exists()):
            print("get_h5file: Sequence folder does not exist")
            return
        reps = np.array([f for f in path.iterdir() if (f.suffix == '.h5' and f.name[-11:-8] == 'rep')])
        non_reps = np.array([f for f in path.iterdir() if (f.suffix == '.h5' and f.name[-11:-8] != 'rep')])
        reps_ind_sorted = np.argsort(np.array([int(f.name[-8:-3]) for f in reps]))
        non_reps_ind_sorted = np.argsort(np.array([int(f.name[-7:-3]) for f in non_reps]))
        reps_ordered = reps[reps_ind_sorted]
        non_reps_ordered = non_reps[non_reps_ind_sorted]
        if show_reps:
            return np.concatenate((non_reps_ordered,reps_ordered))
        else:
            return non_reps_ordered
    except Exception as e:
        if show_errs:
            raise("Error in get_ordered_sequence: " + str(e))
        return

def get_ordered_multiple_sequences(year,month,day,sequences,bec2_path,show_reps = True,show_errs = False):
    try:
        filenames = get_ordered_sequence(year,month,day,sequences[0],bec2_path,show_reps,show_errs)
        for sequence in sequences[1:]:
            filenames = np.concatenate((filenames,get_ordered_sequence(year,month,day,sequence,bec2_path,show_reps,show_errs)))
        return filenames
    except Exception as e:
        if show_errs:
            raise("Error in get_ordered_multiple_sequences: " + str(e))
        return