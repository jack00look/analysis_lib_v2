cam_vert1 = {
    'h5group': 'data/cam_vert1/images',
    'name': 'vert1',
    'px_size': 1.019e-6,
    'alpha': 1.23,
    'chi_sat': 215e6,
    'pulse_time': 5e-6,
    'roi_back': [0,10,0,50],
    'atoms_images': ['PTAI_m1', 'PTAI_m2'],
    'probe_image': 'PTAI_probe',
    'background_image': 'PTAI_back',
    'dynamic_range': 30000#2**16
}

cam_hor3 = {
            'h5group': "data/cam_hor3/images",
            'px_size':9.15e-6, # measured on 2020 Jan 14
            'name': 'hor1',
            'alpha': 2.86,
            'chi_sat': 175e8,
            'pulse_time': 5e-6,
            'roi_back' : [0,300,0,300],
            'atoms_images': ['atoms'],
            'probe_image': 'probe',
            'background_image': 'background',
            'dynamic_range': 2**16
            }

cam_hor1 = {
    'h5group': 'data/cam_hor1/images',
    'name': 'hor1',
    'px_size': 4.34e-6,
    'alpha': 2.86,
    'chi_sat': 175e8,
    'pulse_time': 5e-6,
    'roi_back' : [0,300,0,300],
    'atoms_images': ['atoms'],
    'probe_image': 'probe',
    'background_image': 'back',
    'dynamic_range': 2**16
}

cameras = [cam_vert1, cam_hor1, cam_hor3]