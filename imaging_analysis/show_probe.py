#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import lyse
import numpy as np
import h5py
from settings import cameras_db
from OD import calculate_OD, get_OD
from plotting.cam_like_plots import plot_show
from scipy.ndimage import center_of_mass


run = lyse.Run(lyse.path)
roi_back = np.s_[-50:, 400:600]
# roi_back = np.s_[400:500,1300:1500]
# roi_back = np.s_[400:500,200:400]

#Only with Cam_hor3
roi = np.s_[50:1000, 200:1200]
roi = np.s_[:,:]

for name, cam in cameras_db.items():
    try:
        with h5py.File(lyse.path) as h5file:
            img_names = [s.decode('utf-8') for s in h5file[cam['h5group']].attrs['img_names']]
            imgs = h5file[cam['h5group']]
            images = dict(zip(img_names, imgs))

            atoms = images['atoms'][roi]
            probe = images['probe'][roi]
            background = images['back'][roi]

            # atoms = images['PTAI_p1']
            # probe = images['PTAI_probe']
            # background = images['PTAI_back']

        #OD = get_OD(lyse.path, cam, roi_back=np.s_[100:200, 200:300])
        OD = calculate_OD(atoms, probe, back1=background, roi_background=roi_back,alpha=1)
        print(f"Dimension: {OD.shape}")

        # OD_FFT = np.log(np.abs(np.fft.fftshift(np.fft.fft2(OD))))
        # fig0, ax0 = plt.subplots()
        # ax0.imshow(OD_FFT, vmin=-6, vmax=5)
        # ax0.set_aspect('auto')

        # fig3,ax3 = plt.subplots(layout='constrained',figsize=(3,3))
        # img = ax3.imshow(OD, aspect='auto',cmap='gist_stern',vmin=0,vmax=1.5)
        # fig3.colorbar(img)
        # ax3.set_xlim(750,950)
        # ax3.set_ylim(400,600)
        # ax3.axhline(506, color='g')
        
        fig, ax  = plot_show(OD,
                  #cmap="gist_yarg",
                  cmap="gist_stern",
                  vmin=0.,
                  vmax = 1.5)
        
        print(np.max(probe),np.min(probe),np.mean(probe))
        fig, ax  = plot_show(probe,
                  #cmap="gist_yarg",
                  cmap="gist_stern",
                  vmin=0,
                  vmax = np.max(probe))
        ax[0, 1].set_title(name)

        ax[0,1].axhline(331)
        #ax[0,1].axvline(836)
        
        OD_sum = np.sum(OD[100:600,400:1000])
        run.save_result('{}_ODmax'.format(name), OD.max())
        run.save_result('{}_ODsum'.format(name), OD_sum)

    except (KeyError, ValueError) as e:
        print(f'no images from {name}')
        continue
