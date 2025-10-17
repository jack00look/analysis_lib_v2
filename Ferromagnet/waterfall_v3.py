import numpy as np
import matplotlib.pyplot as plt
import lyse
import pandas as pd
import h5py
import importlib
import sys
from lmfit import Model
import getpass
username = getpass.getuser()
analysis_lib_v2_folder = "/home/{}/labscript-suite/userlib/analysislib/analysis_lib_v2".format(username)
print("{}/general_lib/main_lyse.py".format(analysis_lib_v2_folder))

spec = importlib.util.spec_from_file_location("general_lib.main", "{}/general_lib/main_lyse.py".format(analysis_lib_v2_folder))
print(spec)
general_lib_lyse_mod = importlib.util.module_from_spec(spec)
print(general_lib_lyse_mod)
spec.loader.exec_module(general_lib_lyse_mod)
print('general_lib_lyse_mod imported')


'''Plots a waterfall plot of the magnetization of the cloud as a function of a certain variable (y_axis) discarding repetitions'''
''' scan can be 'ARP_Forward', 'ARP_Backward' or 'bubbles' '''
''' data_origin can be 'od_remove_thpart' or 'spin_waves' '''

def edge_func(x, x0, A,B, m,c):
       val = A/np.pi*np.arctan(B*(x-x0))+m*(x-x0)+c
       return val

edge_model = Model(edge_func,independent_vars = ['x'],nan_policy = 'omit')
params_edge = edge_model.make_params()
params_edge['A'].set(value=0.5, vary=False)


def edge_func():
    return


def waterfall_plot(df,seqs, scan, data_origin,title_labels = None,constraints= None,average = False):

    # selects the sequences to be analyzed
    df = df[(df['sequence_index'].isin(seqs))]

    if constraints is not None:
        for key in constraints.keys():
            try:
                len(constraints[key])
                df = df[(df[key].isin(constraints[key]))]
            except:
                df = df[(df[key]==constraints[key])]

    # selects the y axis
    if scan == 'ARP_Forward':
        y_axis = 'ARPF_final_set_field'
        title = 'ARP Forward'
    elif scan == 'ARP_Backward':
        y_axis = 'ARPB_final_set_field'
        title = 'ARP Backward'
    elif scan == 'bubbles':
        y_axis = 'uW_pulse'
        title = 'Bubbles'

    what = data_origin
    m1_lbl = (what, 'm1_1d')
    m2_lbl = (what, 'm2_1d')

    

    if what == 'od_remove_thpart':
        # builds the ROI index tuple, with center at 400 and width 400
        rx_s = df[(what,'rx')].values
        rx_max = np.max(rx_s)
        w = rx_max
        c = 1024 #int(M.shape[1]//2)
        roi = np.s_[:, int(np.round(c-rx_max)):int(np.round(c+rx_max))]
    elif what == 'spin_waves':
        c = 700
        w = 400
        roi = np.s_[:, :]
    elif what == 'spin_waves_FLAT_20250129':
        c = 300
        w = 200
        roi = np.s_[:, :]

    elif what == 'show_ODs':
        c = 300
        w = 200
        roi = np.s_[:, :]

    
    # sorts values according to the y axis and discards repetitions
    sorted_df = df.sort_values(y_axis)
    y_orig = sorted_df[y_axis].values    #All valuew
    print(y_axis, y_orig)
    unique,unique_indexes = np.unique(sorted_df[y_axis], return_index = True)
    y_raw = (sorted_df[y_axis])[unique_indexes].values    #Unique values
    print(sorted_df[m1_lbl].values[0])
    img_dimensions = sorted_df[m1_lbl].values[0].shape[0]

    if average:
        m1_raw = np.zeros((len(y_raw),img_dimensions))
        m2_raw = np.zeros((len(y_raw),img_dimensions))
        for i in range(len(y_raw)):
            mask = (y_orig == y_raw[i])
            m1_raw[i] = np.mean(sorted_df[m1_lbl][mask])
            m2_raw[i] = np.mean(sorted_df[m2_lbl][mask])

    else:
        m1_raw = (sorted_df[m1_lbl])[unique_indexes].values
        m2_raw = (sorted_df[m2_lbl])[unique_indexes].values
    # selects shots with 800 points of the density along the x direction, and with type of the density array being np.ndarray
    image_px_width = 2048
    shape = (image_px_width, )
    check = type(np.ndarray(1))
    type_check = np.ones(len(m1_raw), dtype=bool)
    shape_check = np.ones(len(m1_raw), dtype=bool)
    Nmax_check = np.ones(len(m1_raw), dtype=bool)

    for i in range(len(m1_raw)):
        try:
            type_check[i] = (type(m1_raw[i]) == check)
            shape_check[i] = (m1_raw[i].shape == shape)
            shape_check[i] = True
            max_val = np.max(m1_raw[i]+m2_raw[i])
            if (max_val>450):#and(max_val<370):
                Nmax_check[i] = True
            else:
                Nmax_check[i] = True
        except:
            type_check[i] = False
            shape_check[i] = False
            Nmax_check[i] = False

    good = type_check & shape_check & Nmax_check
    print(good)

    # sorts the dataframe according to the y_axis and transforms the density of m1 and m2, clipped in the ROI, into a numpy array
    try:
        y = np.array(y_raw[good])          
        M1 = np.stack(m1_raw[good])[roi]
        M2 = np.stack(m2_raw[good])[roi]
    except Exception as e:
        print(e)
        print('FAIL: No good shots found')
        return


    # rescales density of m2 since the imaging is less efficient for that state
    IMAGING_FACTOR = 1.#1.3
    M2 *= IMAGING_FACTOR

    # calculates the total density and the magnetization
    D = M2+M1
    M = (M2-M1)/D
    M = np.clip(M, -1, +1)


    # creates array of position along x
    x_plot = np.arange(-0.5,M.shape[1],1)
    try:
        y_plot = np.concatenate((y, np.array([y[-1]+np.min(np.diff(y))])))
    except:
        y_plot = np.concatenate((y, np.array([y[-1]+1])))
        print('BEWARE: y has only one value, waterfall plot not to be trusted')


    # every value of M is clipped between -1 and 1 (ie if less than -1, it is set to -1, if more than 1, it is set to 1)

    fig, axs = plt.subplots(ncols=2,tight_layout=True, figsize=(10, 5),sharey=True)

    # fig1,ax1 = plt.subplots()
    # N_M1 = np.sum(M1,axis=1)
    # N_M2 = np.sum(M2,axis=1)
    # N_bec = sorted_df[(what, 'N_bec')][unique_indexes][good].values
    # N_tot = N_M1 + N_M2
    # ax1.plot(N_tot,N_bec,'.')

    ax, ax1 = axs
    
    ax.pcolormesh(x_plot,y_plot,M, vmin=-1, vmax=1, cmap='RdBu')
    

    #Check for red domains
    y_plot_centers = y_plot[:-1]+np.diff(y_plot)/2
    #bubble_counter = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        max_index = np.argmax(M1[i,:])
        Z_avg = np.mean(M[i,max_index-10:max_index+10])
        #if Z_avg < 0.1:
         #   bubble_counter[i] = 1
          #  ax.scatter(c, y_plot_centers[i], color = 'red')
    #last_bubble = y_plot[np.where(bubble_counter == 1)[0][-1]+1]

    
    ax.set_ylabel(y_axis)
    ax.set_xlabel('$\mu m$')

    if average:
        title += '\nAVERAGED\n'
    else:
        title += '\nUNIQUE\n'
    title = title + '   (' + f'seqs: {seqs}'+')'
    #if (scan=='ARP_Backward')or(scan=='ARP_Forward'):
    #    title += f'\nLast bubble at '+'{:.3f}'.format(last_bubble)
    if title_labels is not None:
        title += '\n'
        for label in title_labels:
            title += label + ': ' + str(np.unique(sorted_df[label][unique_indexes].values)) + '\n'
    ax.set_title(title)
    ax1.pcolormesh(x_plot,y_plot,D)
    plt.show()

   

df_orig = general_lib_lyse_mod.get_day_data(today = True)

try:
    #waterfall_plot(df,[36],'ARP_Forward','spin_waves',title_labels=['CigarV','Cigar_end'],constraints={'Cigar_end':[0.01]},average=False) #od_remove_thpart,spin_waves
    #waterfall_plot(df,[73],'ARP_Forward','show_ODs',title_labels=[],constraints={},average=False) #od_remove_thpart,spin_waves
    #waterfall_plot(df_orig,[29],'ARP_Backward','show_ODs',title_labels=[],constraints={},average=False) #od_remove_thpart
    waterfall_plot(df_orig,[109],'ARP_Backward','show_ODs',title_labels=[],constraints={},average=False) #od_remove_thpart,spin_waves


except Exception as e:
    raise(e)
    print('FAIL: waterfall plot failed')