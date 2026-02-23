import numpy as np

cameras = {
    'cam_vert1': {
        'h5group': 'data/cam_vert1/images',
        'name': 'vert1',
        'method': 'absorption',
        #'px_size': 1.019e-6,
        'px_size': 1,
        'alpha': 2.33,
        #'chi_sat': 215e6,  old
        'chi_sat': 4.7e7, # updated 2026 Feb 03
        'pulse_time': 5e-6,
        'roi_back': np.s_[110:,-300:],
        'roi_integration': np.s_[:,:],
        'atoms_images': ['PTAI_m2', 'PTAI_m1','PTAI_0','PTAI_p1'],
        'probe_image': 'PTAI_probe',
        'background_image': 'PTAI_back',
        'dynamic_range': 10000,#2**16,
        'roi': np.s_[:, :],
        'axis_matrix': np.array([[1, 0], [0, 1],[0, 0]])
        },

    'cam_vert2_PHC': {
        'h5group': 'data/cam_vert2/images',
        'name': 'vert2_PHC',
        #'method': 'absorption',
         'method': 'phase_contrast',
        'px_size': 1.019e-6,
        #'px_size': 1,
        'alpha': 2.5,
        'chi_sat': 215e6,
        'pulse_time': 5e-6,
        'roi_back': np.s_[100:200,100:400],
        'roi_integration': np.s_[25:,:],
        'atoms_images': ['PHC_m2', 'PHC_m1_1', 'PHC_m1_2' ,'PHC_0','PHC_p1'],
        'probe_image': 'PHC_probe',
        'background_image': 'PHC_back',

        'dynamic_range': 1000, #60000, #2**16, 
        'roi': np.s_[:, :],
        'axis_matrix': np.array([[0, 1], [1, 0],[0, 0]])
        },

    # 'cam_vert2_PHC': {
    #     'h5group': 'data/cam_vert2/images',
    #     'name': 'vert2_PHC',
    #     'method': 'absorption',
    #     #'method': 'phase_contrast',
    #     'px_size': 1.019e-6,
    #     #'px_size': 1,
    #     'alpha': 2.5,
    #     'chi_sat': 215e6,
    #     'pulse_time': 5e-6,
    #     'roi_back': np.s_[100:200,100:400],
    #     'roi_integration': np.s_[45:70,:],
    #     'atoms_images': ['PTAI_m1','PTAI_m2'],
    #     'probe_image': 'PTAI_probe',
    #     'background_image': 'PTAI_back',
    #     'dynamic_range': 60000,
    #     'roi': np.s_[:, :],
    #     'axis_matrix': np.array([[0, 1], [1, 0],[0, 0]])
    #     },

    'cam_hor3': {
        'h5group': "data/cam_hor3/images",
        'px_size':9.89e-6, # measured on 2020 Jan 14
        'name': 'hor3',
        'method': 'absorption',
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
        'method': 'absorption',
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
        },

    'cam_vert_up': {
        'h5group': "data/cam_vert_up/images",
        'px_size':4.34e-6, # measured on 2020 Jan 14
        'name': 'vert_up',
        'method': 'absorption',
        'alpha': 2.86,
        'chi_sat': 175e8,
        'pulse_time': 5e-6,
        'roi_back' : np.s_[-100:,-100:],
        'roi_integration': np.s_[:, 900:950],
        'atoms_images': ['atoms','image0'],
        'probe_image': 'probe',
        'background_image': 'back',
        'dynamic_range': 60000,
        'roi': np.s_[:, :],
        'axis_matrix': np.array([[1, 0],
          [0, 0],
          [0, 1]])
        # 'axis_matrix': np.array([[0, -1],
        #   [0, 0],
        #   [1, 0]])
        }

}