import numpy as np

cameras = {
    'cam_vert1': {
        'h5group': 'data/cam_vert1/images',
        'name': 'vert1',
        'px_size': 1.019e-6,
        #'px_size': 1,
        'alpha': 2.5,
        'chi_sat': 215e6,
        'pulse_time': 5e-6,
        'roi_back': np.s_[0:10,0:100],
        'roi_integration': np.s_[:,:],
        'atoms_images': ['PTAI_m2', 'PTAI_m1','PTAI_0','PTAI_p1'],
        'probe_image': 'PTAI_probe',
        'background_image': 'PTAI_back',
        'dynamic_range': 60000,#2**16,
        'roi': np.s_[:, :],
        'axis_matrix': np.array([[0, 1], [1, 0],[0, 0]])
        },

    'cam_hor3': {
        'h5group': "data/cam_hor3/images",
        'px_size':9.89e-6, # measured on 2020 Jan 14
        'name': 'hor3',
        'alpha': 2.86,
        'chi_sat': 175e8,
        'pulse_time': 5e-6,
        'roi_back' : np.s_[-100:,-100:],
        'roi_integration': np.s_[:, :],
        'atoms_images': ['atoms'],
        'probe_image': 'probe',
        'background_image': 'back',
        'dynamic_range': 2**16,
        'roi': np.s_[300:, :950],
        'axis_matrix': np.array([[1, 0], [0, 0],[0, 1]])
        },
    'cam_hor1': {
        'h5group': 'data/cam_hor1/images',
        'name': 'hor1',
        'px_size': 4.34e-6,
        'alpha': 2.86,
        'chi_sat': 175e8,
        'pulse_time': 5e-6,
        'roi_back' : np.s_[-50:,-50:],
        'roi_integration': np.s_[:, :],
        'atoms_images': ['atoms'],
        'probe_image': 'probe',
        'background_image': 'back',
        'dynamic_range': 2**16,
        'roi': np.s_[:, :],
        'axis_matrix': np.array([[1, 0], [0, 0],[0, -1]])
        }
}