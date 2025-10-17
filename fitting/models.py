#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 12-2018 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

import numpy as np

from lmfit import Parameters
from lmfit.model import Model
from lmfit.models import update_param_vals, COMMON_GUESS_DOC

from .functions import gaussian2d, thomasfermi2d, bose2d, bimodalbose2d, bimodal1d, guess_from_peak_2d, bimodalbose2d2centers, bimodal2d, g3 ,flat2d

from scipy.constants import hbar, Boltzmann as kB
from scipy.ndimage import gaussian_filter
# from MT_params import omega_x, omega_y


class Model2D(Model):
    def __init__(self, func, pixel_size=np.nan, *args, **kwargs):
        super(Model2D, self).__init__(func, independent_vars=['X', 'Y'], nan_policy='omit', *args, **kwargs)
        # self.set_param_hint('hbar', value=hbar, vary=False)
        # self.set_param_hint('kB', value=kB, vary=False)
        self.set_param_hint('pixel_size',value=pixel_size, vary=False)
        self.set_param_hint('cross_section', value=1.656425e-13, vary=False) # Sodium D line
        # self.set_param_hint('omega_x',value=pixel_size, vary=False)
        # self.set_param_hint('omega_x',value=pixel_size, vary=False)

    def fit(self, data, *args, **kwargs):
       if 'X' not in kwargs or 'Y' not in kwargs:
           Y, X = np.indices(data.shape)# *self.pixel_size
           kwargs.update({'X': X, 'Y': Y})
       return super(Model2D, self).fit(data, *args, **kwargs)


class Gaussian2DModel(Model2D):
    __doc__ = gaussian2d.__doc__ + COMMON_GUESS_DOC if gaussian2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Gaussian2DModel, self).__init__(func=gaussian2d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('sx', min=0)
        self.set_param_hint('sy', min=0)

        expressions = {'N': 'amp*2*pi * sigma_x*sigma_y/cross_section',
                       'sigma_x': 'sx*pixel_size',
                       'sigma_y': 'sy*pixel_size',
                       'integral': 'amp*2*pi*sx*sy'
                       }
        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp=amp, mx=mx, my=my, sx=sx, sy=sy)

class J_BimodalBose2DModel2Centers(Model2D):
    __doc__ = bimodalbose2d2centers.__doc__ + COMMON_GUESS_DOC if bimodalbose2d2centers.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(J_BimodalBose2DModel2Centers, self).__init__(func=bimodalbose2d2centers, *args, **kwargs)

        #1.2020569 is g3(1), see paper on the wall
        expressions = {self.prefix+'N_th': f'{self.prefix}amp_g * (2*pi) * {self.prefix}sx * {self.prefix}sy * 1.2020569',
                       self.prefix+'N_bec': f'(2*pi/5)*{self.prefix}rx*{self.prefix}ry*{self.prefix}amp_tf',
                       self.prefix+'N': f'{self.prefix}N_th + {self.prefix}N_bec',
                       }
        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)
    def guess(self, data,X,Y, **kwargs):
        data_smoothed = gaussian_filter(data, sigma=10)
        data_smoothed_var = np.var(data_smoothed)
        self.set_param_hint('var', value=data_smoothed_var, vary=False)
        data_smoothed_smooth = gaussian_filter(data_smoothed, sigma=50)
        offset = np.min(data_smoothed_smooth)
        data_smoothed -= offset
        amp_tot = np.max(data_smoothed)
        ind_my, ind_mx = np.unravel_index(np.argmax(data_smoothed,), data_smoothed.shape)
        mx = X[ind_my,ind_mx]
        my = Y[ind_my, ind_mx]
        area = np.trapz(data_smoothed, x=X[0], axis=1)  # integrate along Y
        area = np.trapz(area, x=Y[:,0])
        s = np.sqrt(area/(amp_tot * 2 * np.pi))
        x_c = X.mean()
        x_diff = X.max() - X.min()
        x_low_limit = x_c - 0.2*x_diff
        x_high_limit = x_c + 0.2*x_diff
        x_low_limit = X.min() + 0.1*x_diff
        x_high_limit = X.max() - 0.1*x_diff
        y_c = Y.mean()
        y_diff = Y.max() - Y.min()
        y_low_limit = y_c - 0.2*y_diff
        y_high_limit = y_c + 0.2*y_diff
        y_low_limit = Y.min() + 0.1*y_diff
        y_high_limit = Y.max() - 0.1*y_diff
        self.set_param_hint('offset', value=offset,min=offset -2e11,max = offset + 2e11)
        self.set_param_hint('amp_tf', value=1.5*amp_tot, min=0, max=10*amp_tot)
        self.set_param_hint('amp_g', value=amp_tot/5., min=0, max=10*amp_tot)
        self.set_param_hint('sx', value=s*1.,min = 0.01*x_diff,max = 5.*x_diff)
        self.set_param_hint('sy', value=s*1.,min = 0.01*y_diff,max = 5.*y_diff)
        self.set_param_hint('rx', value=s*1.,min = 0.01*x_diff,max = 5.*x_diff)
        self.set_param_hint('ry', value=s*1.,min = 0.01*y_diff,max = 5.*y_diff)
        self.set_param_hint('mbx', value=mx,min = x_low_limit,max = x_high_limit)
        self.set_param_hint('mby', value=my,min = y_low_limit,max = y_high_limit)
        self.set_param_hint('mtx', value=mx,min = x_low_limit,max = x_high_limit)
        self.set_param_hint('mty', value=my,min = y_low_limit,max = y_high_limit)
        return self.make_params()

    def eval_bimodal_components(self, params, X):
        args = [params[k].value for k in [f'{self.prefix}amp_g', f'{self.prefix}amp_tf', f'{self.prefix}c_g', f'{self.prefix}c_tf', f'{self.prefix}s_g', f'{self.prefix}r_tf', f'{self.prefix}offset']]
        therm, bec = bimodalbose2d2centers(X,  *args, return_both=True)
        return therm, bec

    def plot_best(self,params,X,ax,invert=False):
        Y_th, Y_bec = self.eval_bimodal_components(params,X)
        offset = params[f'{self.prefix}offset'].value
        Y_tot = Y_th + Y_bec - offset
        if invert:
            ax.plot(Y_tot,X,'r-',label='fit')
            ax.plot(Y_th,X,'y--',label='thermal')
            ax.plot(Y_bec,X,'m--',label='bec')
        else:
            ax.plot(X,Y_tot,'r-',label='fit')
            ax.plot(X,Y_th,'y--',label='thermal')
            ax.plot(X,Y_bec,'m--',label='bec')

class J_Flat2DModel(Model2D):
    __doc__ = gaussian2d.__doc__ + COMMON_GUESS_DOC if flat2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(J_Flat2DModel, self).__init__(func=flat2d, *args, **kwargs)

    def guess(self, data,X,Y, **kwargs):
        data_smoothed = gaussian_filter(data, sigma=10)
        self.set_param_hint('offset', value=np.mean(data_smoothed))
        return self.make_params()

    def plot_best(self,params,X,Y,ax,invert=False):
        Z = self.eval(params, X=X, Y=Y)
        ax.imshow(Z, origin='lower', extent=(X.min(), X.max(), Y.min(), Y.max()), aspect='auto')

class J_Gaussian2DModel(Model2D):
    __doc__ = gaussian2d.__doc__ + COMMON_GUESS_DOC if gaussian2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(J_Gaussian2DModel, self).__init__(func=gaussian2d, *args, **kwargs)

        expressions = {f'{self.prefix}N': f'{self.prefix}amp*2*pi * {self.prefix}sx*{self.prefix}sy'}
        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data,X,Y, **kwargs):
        data_smoothed = gaussian_filter(data, sigma=10)
        data_smoothed_var = np.var(data_smoothed)
        self.set_param_hint('var', value=data_smoothed_var, vary=False)
        offset = np.min(data_smoothed)
        data_smoothed -= offset
        amp = np.max(data_smoothed)
        ind_my, ind_mx = np.unravel_index(np.argmax(data_smoothed,), data_smoothed.shape)
        mx = X[ind_my,ind_mx]
        my = Y[ind_my, ind_mx]
        area = np.trapz(data_smoothed, x=X[0], axis=1)  # integrate along Y
        area = np.trapz(area, x=Y[:,0])
        s = np.sqrt(area/(amp * 2 * np.pi))
        x_c = X.mean()
        x_M = X.max()
        x_m = X.min()
        x_diff = x_M - x_m
        x_low_limit = x_c - 0.2*(x_M - x_m)
        x_high_limit = x_c + 0.2*(x_M - x_m)
        y_c = Y.mean()
        y_M = Y.max()
        y_m = Y.min()
        y_diff = y_M - y_m
        y_low_limit = y_c - 0.2*(y_M - y_m)
        y_high_limit = y_c + 0.2*(y_M - y_m)
        self.set_param_hint('offset', value=offset,min=-5e14,max = + 5e14)
        self.set_param_hint('amp', value=amp, min=0, max=10*amp)
        self.set_param_hint('sx', value=s,min = 0.05*x_diff,max = 0.8*x_diff)
        self.set_param_hint('sy', value=s,min = 0.05*y_diff,max = 0.8*y_diff)
        self.set_param_hint('mx', value=mx,min = x_low_limit,max = x_high_limit)
        self.set_param_hint('my', value=my,min = y_low_limit,max = y_high_limit)
        return self.make_params()

    def plot_best(self,params,X,Y,ax,invert=False):
        Z = self.eval(params, X=X, Y=Y)
        ax.imshow(Z, origin='lower', extent=(X.min(), X.max(), Y.min(), Y.max()), aspect='auto')

class ThomasFermi2DModel(Model2D):
    __doc__ = thomasfermi2d.__doc__ + COMMON_GUESS_DOC if thomasfermi2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(ThomasFermi2DModel, self).__init__(func=thomasfermi2d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)

        expressions = {'N': '1.25 * amp * rad_x*rad_y/cross_section',
                       'rad_x': 'rx*pixel_size',
                       'rad_y': 'ry*pixel_size',
                       }
        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp=amp, mx=mx, my=my, rx=sx, ry=sy)

class BimodalBose2DModel(Model2D):
    __doc__ = bimodalbose2d.__doc__ + COMMON_GUESS_DOC if bimodalbose2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BimodalBose2DModel, self).__init__(func=bimodalbose2d, *args, **kwargs)
        self.set_param_hint('amp_g', min=0)
        self.set_param_hint('amp_tf', min=0)
        self.set_param_hint('sx', min=0)
        self.set_param_hint('sy', min=0)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)


        #1.2020569 is g3(1), see paper on the wall
        expressions = {'Nth': 'amp_g*2*pi * sigma_x*sigma_y*1.2020569/cross_section',
                       'sigma_x': 'sx*pixel_size',
                       'sigma_y': 'sy*pixel_size',
                       'Nbec': '1.25 * amp_tf* rad_x*rad_y/cross_section',
                       'rad_x': 'rx*pixel_size',
                       'rad_y': 'ry*pixel_size',
                       'N': 'Nbec + Nth'
                       }

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'T_x'] = f'mass/({kB})*' +\
             f'(trapping_freq**2 * {self.prefix}sigma_x**2) / (1/(4*pi**2) + trapping_freq**2 * tof**2)'
            expressions[self.prefix+'T_y'] = f'mass/({kB})*' +\
              f'(trapping_freq**2 * {self.prefix}sigma_y**2) / (1/(4*pi**2) + trapping_freq**2 * tof**2)'


        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp_g=amp/3, amp_tf=amp, mx=mx, my=my, sx=sx, sy=sy, rx=sx, ry=sy)

    def eval_bimodal_components(self, params, X, Y):
        args = [params[k].value for k in ['amp_g', 'amp_tf', 'mx', 'my', 'sx', 'sy', 'rx', 'ry', 'offset']]
        therm, bec = bimodalbose2d(X, Y, *args, return_both=True)
        return therm, bec

class BimodalBose2DModel2Centers(Model2D):
    __doc__ = bimodalbose2d2centers.__doc__ + COMMON_GUESS_DOC if bimodalbose2d2centers.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BimodalBose2DModel2Centers, self).__init__(func=bimodalbose2d2centers, *args, **kwargs)
        self.set_param_hint('amp_g', min=0)
        self.set_param_hint('amp_tf', min=0)
        self.set_param_hint('sx', min=0)
        self.set_param_hint('sy', min=0)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)

        #1.2020569 is g3(1), see paper on the wall

        expressions = {'Nth': 'amp_g*2*pi * sigma_x*sigma_y*1.2020569/cross_section',
                       'sigma_x': 'sx*pixel_size',
                       'sigma_y': 'sy*pixel_size',
                       'Nbec': '(2*pi/5) * amp_tf* rad_x*rad_y/cross_section',
                       'rad_x': 'rx*pixel_size',
                       'rad_y': 'ry*pixel_size',
                       'N': 'Nbec + Nth'
                       }

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'T_x'] = f'mass/({kB})*' +\
             f'(trapping_freq**2 * {self.prefix}sigma_x**2) / (1/(4*pi**2) + trapping_freq**2 * tof**2)'
            expressions[self.prefix+'T_y'] = f'mass/({kB})*' +\
              f'(trapping_freq**2 * {self.prefix}sigma_y**2) / (1/(4*pi**2) + trapping_freq**2 * tof**2)'


        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp_g=amp/3, amp_tf=amp, mbx=mx, mtx = mx, mby=my, mty=my, sx=sx, sy=sy, rx=sx, ry=sy)

    def eval_bimodal_components(self, params, X, Y):
        args = [params[k].value for k in ['amp_g', 'amp_tf', 'mbx', 'mtx', 'mby', 'mty', 'sx', 'sy', 'rx', 'ry', 'offset']]
        therm, bec = bimodalbose2d(X, Y, *args, return_both=True)
        return therm, bec



class Bimodal2DModel(Model2D):
    __doc__ = bimodal2d.__doc__ + COMMON_GUESS_DOC if bimodal2d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Bimodal2DModel, self).__init__(func=bimodal2d, *args, **kwargs)
        self.set_param_hint('amp_g', min=0)
        self.set_param_hint('amp_tf', min=0)
        self.set_param_hint('sx', min=0)
        self.set_param_hint('sy', min=0)
        self.set_param_hint('rx', min=0)
        self.set_param_hint('ry', min=0)

        expressions = {'Nth': 'amp_g*2*pi * sigma_x*sigma_y/cross_section',
                       'sigma_x': 'sx*pixel_size',
                       'sigma_y': 'sy*pixel_size',
                       'Nbec': '1.25 * amp_tf* rad_x*rad_y/cross_section',
                       'rad_x': 'rx*pixel_size',
                       'rad_y': 'ry*pixel_size',
                       'N': 'Nbec + Nth'
                       }

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'T_x'] = f'mass/({kB})*' +\
             f'(trapping_freq**2 * {self.prefix}sigma_x**2) / (1/(4*pi**2) + trapping_freq**2 * tof**2)'
            expressions[self.prefix+'T_y'] = f'mass/({kB})*' +\
              f'(trapping_freq**2 * {self.prefix}sigma_y**2) / (1/(4*pi**2) + trapping_freq**2 * tof**2)'

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, mx, my, sx, sy = guess_from_peak_2d(data, **kwargs)
        return self.make_params(amp_g=amp/3, amp_tf=amp, mx=mx, my=my, sx=sx, sy=sy, rx=sx, ry=sy)

    def eval_bimodal_components(self, params, X, Y):
        args = [params[k].value for k in ['amp_g', 'amp_tf', 'mx', 'my', 'sx', 'sy', 'rx', 'ry', 'offset']]
        therm, bec = bimodal2d(X, Y, *args, return_both=True)
        return therm, bec
