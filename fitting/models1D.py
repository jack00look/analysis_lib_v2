#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np

from lmfit import Parameters
from lmfit.model import Model
from lmfit.models import update_param_vals, COMMON_GUESS_DOC

from fitting.functions import bimodal1d, gaussian1d, thomasfermi1d, guess_from_peak_1d, skewedgaussian1d, bimodalbose1d, g3, bose1d, bimodalbose1d2centers, J_bimodalbose1d2centers,J_LZ_p1
from scipy import constants as sc
from scipy.ndimage import gaussian_filter1d

K_Boltzmann = '1.380649*10**-23'

class Model1D(Model):
    def __init__(self, func, pixel_size=np.nan, cross_section=1.656425e-13, *args, **kwargs):
        super(Model1D, self).__init__(func, independent_vars=['X'], nan_policy='omit', *args, **kwargs)
        self.set_param_hint('pixel_size',value=pixel_size, vary=False)
        self.set_param_hint('cross_section', value=cross_section, vary=False) # Sodium D line

    def fit(self, data, *args, **kwargs):
        """
        1D model is fitted on the image integrated along the two direction
        """
        if 'X' not in kwargs:
            X = np.arange(data.size)
            kwargs.update({'X': X})

        return super(Model1D, self).fit(data, *args, **kwargs)

class Gaussian1DModel(Model1D):
    __doc__ = gaussian1d.__doc__ + COMMON_GUESS_DOC if gaussian1d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Gaussian1DModel, self).__init__(func=gaussian1d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('s', min=0)

        expressions = {self.prefix+'N': f'{self.prefix}amp * sqrt(2*pi) * {self.prefix}s * {self.prefix}pixel_size**2 /{self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}s*{self.prefix}pixel_size',
                       }

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        amp, m, s = guess_from_peak_1d(data, **kwargs)
        self.set_param_hint('amp', value=amp, max=data.max())
        self.set_param_hint('m', value=m)
        self.set_param_hint('s', value=s)

        return self.make_params()

class J_Gaussian1DModel(Model1D):
    __doc__ = gaussian1d.__doc__ + COMMON_GUESS_DOC if gaussian1d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(J_Gaussian1DModel, self).__init__(func=gaussian1d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('s', min=0)

        expressions = {self.prefix+'N': f'{self.prefix}amp * sqrt(2*pi) * {self.prefix}s'
                       }

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data,X, **kwargs):
        data_smoothed = gaussian_filter1d(data, 5)
        amp = np.max(data_smoothed)
        m = X[np.argmax(data_smoothed)]
        offset = np.min(data_smoothed)
        area = np.trapz(data_smoothed - offset, X)
        s = (area/amp/np.sqrt(2*np.pi))
        self.set_param_hint('amp', value=amp)
        self.set_param_hint('m', value=m)
        self.set_param_hint('s', value=s)
        self.set_param_hint('offset', value=offset)
        return self.make_params()

    
    def plot_best(self,params,X,ax,invert=False):
        Y = self.eval(params,X=X)
        if invert:
            ax.plot(Y,X,'r-',label='fit')
        else:
            ax.plot(X,Y,'r-',label='fit')

class J_LZp1Model(Model1D):
    __doc__ = J_LZ_p1.__doc__ + COMMON_GUESS_DOC if J_LZ_p1.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(J_LZp1Model, self).__init__(func=J_LZ_p1, *args, **kwargs)

        expressions = {}

        # for k, expr in expressions.items():
        #     self.set_param_hint(k, expr=expr)


    def guess(self, data,X, **kwargs):
        Bscan_res = -X[np.argmax(data)]
        

        data_smoothed = gaussian_filter1d(data, 5)
        amp = np.max(data_smoothed)
        m = X[np.argmax(data_smoothed)]
        offset = np.min(data_smoothed)
        area = np.trapz(data_smoothed - offset, X)
        s = (area/amp/np.sqrt(2*np.pi))
        self.set_param_hint('amp', value=amp)
        self.set_param_hint('m', value=m)
        self.set_param_hint('s', value=s)
        self.set_param_hint('offset', value=offset)
        return self.make_params()

    
    def plot_best(self,params,X,ax,invert=False):
        Y = self.eval(params,X=X)
        if invert:
            ax.plot(Y,X,'r-',label='fit')
        else:
            ax.plot(X,Y,'r-',label='fit')


class Bose1DModel(Model1D):
    __doc__ = bose1d.__doc__ + COMMON_GUESS_DOC if bose1d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Bose1DModel, self).__init__(func=bose1d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('sx', min=0)

        #1.2020569 is g3(1), see paper on the wall

        expressions = {self.prefix+'N': f'{self.prefix}amp * sqrt(2*pi) * 1.2020569*{self.prefix}sx * {self.prefix}pixel_size**2 /{self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}sx*{self.prefix}pixel_size',
                       }

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        amp, mx, sx = guess_from_peak_1d(data, **kwargs)
        self.set_param_hint('amp', value=amp, max=data.max())
        self.set_param_hint('mx', value=mx)
        self.set_param_hint('sx', value=sx)

        return self.make_params()


class SkewedGaussian1DModel(Model1D):
    __doc__ = gaussian1d.__doc__ + COMMON_GUESS_DOC if gaussian1d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(SkewedGaussian1DModel, self).__init__(func=skewedgaussian1d, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('s', min=0)
        self.set_param_hint('gamma')

        expressions = {self.prefix+'N': f'{self.prefix}amp * sqrt(2*pi) * {self.prefix}s * {self.prefix}pixel_size**2 /{self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}s*{self.prefix}pixel_size',
                       }

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        amp, m, s = guess_from_peak_1d(data, **kwargs)
        self.set_param_hint('amp', value=amp, max=data.max())
        self.set_param_hint('m', value=m)
        self.set_param_hint('s', value=s)
        self.set_param_hint('gamma', value=1)


        return self.make_params()


class ThomasFermi1DModel(Model1D):
    __doc__ = gaussian1d.__doc__ + COMMON_GUESS_DOC if gaussian1d.__doc__ else ""
    def __init__(self, pixel_size=np.nan, *args, **kwargs):
        super(ThomasFermi1DModel, self).__init__(func=thomasfermi1d, pixel_size=pixel_size, *args, **kwargs)
        # self.set_param_hint('pixel_size',value=pixel_size, vary=False)
        # self.set_param_hint('cross_section', value=1.656425e-13, vary=False) # Sodium D line
        self.set_param_hint('amp', min=0)
        self.set_param_hint('rx', min=0)

        expressions = {self.prefix+'N_bec': f'16/15*{self.prefix}rx*{self.prefix}amp*{self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'radius': f'{self.prefix}rx*{self.prefix}pixel_size',
                       }

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, mx, rx = guess_from_peak_1d(data, **kwargs)
        params = self.make_params()
        params[self.prefix+'amp'].set(value=amp)#, max=data.max(), min=0)
        params[self.prefix+'rx'].set(value=rx, min=0)
        params[self.prefix+'mx'].set(value=mx)
        return params

class Bimodal1DModel(Model1D):
    __doc__ = bimodal1d.__doc__ + COMMON_GUESS_DOC if bimodal1d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(Bimodal1DModel, self).__init__(func=bimodal1d, *args, **kwargs)
        self.set_param_hint('A', min=0) # gaussian amplitude
        self.set_param_hint('B', min=0) # parabola amplitude
        self.set_param_hint('sx', min=0)
        self.set_param_hint('rx', min=0)

        expressions = {self.prefix+'N_th': f'{self.prefix}A * sqrt(2*pi) * {self.prefix}sx * {self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'N_bec': f'16/15*{self.prefix}rx*{self.prefix}B*{self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}sx*{self.prefix}pixel_size',
                       self.prefix+'radius': f'{self.prefix}rx*{self.prefix}pixel_size',
                       }

        # add custom expressions

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'temperature'] = f'{self.prefix}mass/({K_Boltzmann})*' +\
             f'({self.prefix}trapping_freq**2 * {self.prefix}sigma**2) / (1/(4*pi**2) + {self.prefix}trapping_freq**2 * {self.prefix}tof**2)'

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        amp, mx, rx = guess_from_peak_1d(data, **kwargs)
        params = self.make_params()
        params[self.prefix+'A'].set(value=amp/5, max=data.max()/2, min=0)
        params[self.prefix+'B'].set(value=amp, max=1.5*data.max(), min=0)
        params[self.prefix+'rx'].set(value=rx, min=0)
        params[self.prefix+'sx'].set(value=rx*1.2, min=0)
        params[self.prefix+'mx'].set(value=mx)
        return params


    def eval_bimodal_components(self, params, x):
        args = [params[k].value for k in [f'{self.prefix}A', f'{self.prefix}B', f'{self.prefix}mx', f'{self.prefix}sx', f'{self.prefix}rx', f'{self.prefix}offset']]
        therm, bec = bimodal1d(x,  *args, return_both=True)
        return therm, bec


class BimodalBose1DModel(Model1D):
    __doc__ = bimodalbose1d.__doc__ + COMMON_GUESS_DOC if bimodalbose1d.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BimodalBose1DModel, self).__init__(func=bimodalbose1d, *args, **kwargs)
        self.set_param_hint('amp_g', min=0) # gaussian amplitude
        self.set_param_hint('amp_tf', min=0) # parabola amplitude
        self.set_param_hint('sx', min=0)
        self.set_param_hint('rx', min=0)

        #1.2020569 is g3(1), see paper on the wall
        expressions = {self.prefix+'N_th': f'{self.prefix}amp_g * sqrt(2*pi) * {self.prefix}sx * 1.2020569 * {self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'N_bec': f'16/15*{self.prefix}rx*{self.prefix}amp_tf*{self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}sx*{self.prefix}pixel_size',
                       self.prefix+'radius': f'{self.prefix}rx*{self.prefix}pixel_size',
                       }

        # add custom expressions

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'temperature'] = f'{self.prefix}mass/({K_Boltzmann})*' +\
             f'({self.prefix}trapping_freq**2 * {self.prefix}sigma**2) / (1/(4*pi**2) + {self.prefix}trapping_freq**2 * {self.prefix}tof**2)'

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        amp, mx, rx = guess_from_peak_1d(data, **kwargs)
        params = self.make_params()
        params[self.prefix+'amp_g'].set(value=amp/5, max=data.max()/2, min=0)
        params[self.prefix+'amp_tf'].set(value=amp, max=1.5*data.max(), min=0)
        params[self.prefix+'rx'].set(value=rx, min=0)
        params[self.prefix+'sx'].set(value=rx*1.2, min=0)
        params[self.prefix+'mx'].set(value=mx)
        return params


    def eval_bimodal_components(self, params, x):
        args = [params[k].value for k in [f'{self.prefix}amp_g', f'{self.prefix}amp_tf', f'{self.prefix}mx', f'{self.prefix}sx', f'{self.prefix}rx', f'{self.prefix}offset']]
        therm, bec = bimodal1d(x,  *args, return_both=True)
        return therm, bec



class BimodalBose1DModel2Centers(Model1D):
    __doc__ = bimodalbose1d2centers.__doc__ + COMMON_GUESS_DOC if bimodalbose1d2centers.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BimodalBose1DModel2Centers, self).__init__(func=bimodalbose1d2centers, *args, **kwargs)
        self.set_param_hint('amp_g', min=0) # gaussian amplitude
        self.set_param_hint('amp_tf', min=0) # parabola amplitude
        self.set_param_hint('sx', min=0)
        self.set_param_hint('rx', min=0)

        #1.2020569 is g3(1), see paper on the wall
        expressions = {self.prefix+'N_th': f'{self.prefix}amp_g * sqrt(2*pi) * {self.prefix}sx * 1.2020569 * {self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'N_bec': f'16/15*{self.prefix}rx*{self.prefix}amp_tf*{self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}sx*{self.prefix}pixel_size',
                       self.prefix+'radius': f'{self.prefix}rx*{self.prefix}pixel_size',
                       }

        # add custom expressions

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'temperature'] = f'{self.prefix}mass/({K_Boltzmann})*' +\
             f'({self.prefix}trapping_freq**2 * {self.prefix}sigma**2) / (1/(4*pi**2) + {self.prefix}trapping_freq**2 * {self.prefix}tof**2)'

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):

        amp, mx, rx = guess_from_peak_1d(data, **kwargs)
        params = self.make_params()
        params[self.prefix+'amp_g'].set(value=amp/5, max=data.max()/2, min=0)
        params[self.prefix+'amp_tf'].set(value=amp, max=1.5*data.max(), min=0)
        params[self.prefix+'rx'].set(value=rx, min=0)
        params[self.prefix+'sx'].set(value=rx*1.2, min=0)
        params[self.prefix+'mgx'].set(value=mx)
        params[self.prefix+'mtfx'].set(value=mx)

        return params


    def eval_bimodal_components(self, params, x):
        args = [params[k].value for k in [f'{self.prefix}amp_g', f'{self.prefix}amp_tf', f'{self.prefix}mgx', f'{self.prefix}mtfx', f'{self.prefix}sx', f'{self.prefix}rx', f'{self.prefix}offset']]
        therm, bec = bimodalbose1d2centers(x,  *args, return_both=True)
        return therm, bec

class BimodalBose1DModel2Centers(Model1D):
    __doc__ = bimodalbose1d2centers.__doc__ + COMMON_GUESS_DOC if bimodalbose1d2centers.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BimodalBose1DModel2Centers, self).__init__(func=bimodalbose1d2centers, *args, **kwargs)
        self.set_param_hint('amp_g', min=0) # gaussian amplitude
        self.set_param_hint('amp_tf', min=0) # parabola amplitude
        self.set_param_hint('sx', min=0)
        self.set_param_hint('rx', min=0)

        #1.2020569 is g3(1), see paper on the wall
        expressions = {self.prefix+'N_th': f'{self.prefix}amp_g * sqrt(2*pi) * {self.prefix}sx * 1.2020569 * {self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'N_bec': f'16/15*{self.prefix}rx*{self.prefix}amp_tf*{self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}sx*{self.prefix}pixel_size',
                       self.prefix+'radius': f'{self.prefix}rx*{self.prefix}pixel_size',
                       }

        # add custom expressions

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'temperature'] = f'{self.prefix}mass/({K_Boltzmann})*' +\
             f'({self.prefix}trapping_freq**2 * {self.prefix}sigma**2) / (1/(4*pi**2) + {self.prefix}trapping_freq**2 * {self.prefix}tof**2)'

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        amp, mx, rx = guess_from_peak_1d(data, **kwargs)
        params = self.make_params()
        params[self.prefix+'amp_g'].set(value=amp/5, max=data.max()/2, min=0)
        params[self.prefix+'amp_tf'].set(value=amp, max=1.5*data.max(), min=0)
        params[self.prefix+'rx'].set(value=rx, min=0)
        params[self.prefix+'sx'].set(value=rx*1.2, min=0)
        params[self.prefix+'mgx'].set(value=mx)
        params[self.prefix+'mtfx'].set(value=mx)

        return params


    def eval_bimodal_components(self, params, x):
        args = [params[k].value for k in [f'{self.prefix}amp_g', f'{self.prefix}amp_tf', f'{self.prefix}mgx', f'{self.prefix}mtfx', f'{self.prefix}sx', f'{self.prefix}rx', f'{self.prefix}offset']]
        therm, bec = bimodalbose1d2centers(x,  *args, return_both=True)
        return therm, bec

class BimodalBose1DModel2Centers(Model1D):
    __doc__ = bimodalbose1d2centers.__doc__ + COMMON_GUESS_DOC if bimodalbose1d2centers.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(BimodalBose1DModel2Centers, self).__init__(func=bimodalbose1d2centers, *args, **kwargs)
        self.set_param_hint('amp_g', min=0) # gaussian amplitude
        self.set_param_hint('amp_tf', min=0) # parabola amplitude
        self.set_param_hint('sx', min=0)
        self.set_param_hint('rx', min=0)

        #1.2020569 is g3(1), see paper on the wall
        expressions = {self.prefix+'N_th': f'{self.prefix}amp_g * sqrt(2*pi) * {self.prefix}sx * 1.2020569 * {self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'N_bec': f'16/15*{self.prefix}rx*{self.prefix}amp_tf*{self.prefix}pixel_size**2 / {self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}sx*{self.prefix}pixel_size',
                       self.prefix+'radius': f'{self.prefix}rx*{self.prefix}pixel_size',
                       }

        # add custom expressions

        #temperature
        if {'tof', 'trapping_freq', 'mass'} <= set(kwargs):
            self.set_param_hint('tof', value = kwargs['tof'], vary = False)
            self.set_param_hint('trapping_freq', value = kwargs['trapping_freq'], vary = False)
            self.set_param_hint('mass', value = kwargs['mass'], vary = False)

            expressions[self.prefix+'temperature'] = f'{self.prefix}mass/({K_Boltzmann})*' +\
             f'({self.prefix}trapping_freq**2 * {self.prefix}sigma**2) / (1/(4*pi**2) + {self.prefix}trapping_freq**2 * {self.prefix}tof**2)'

        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)


    def guess(self, data, **kwargs):
        
        amp, mx, rx = guess_from_peak_1d(data, **kwargs)
        params = self.make_params()
        params[self.prefix+'amp_g'].set(value=amp/5, max=data.max()/2, min=0)
        params[self.prefix+'amp_tf'].set(value=amp, max=1.5*data.max(), min=0)
        params[self.prefix+'rx'].set(value=rx, min=0)
        params[self.prefix+'sx'].set(value=rx*1.2, min=0)
        params[self.prefix+'mgx'].set(value=mx)
        params[self.prefix+'mtfx'].set(value=mx)

        return params


    def eval_bimodal_components(self, params, x):
        args = [params[k].value for k in [f'{self.prefix}amp_g', f'{self.prefix}amp_tf', f'{self.prefix}mgx', f'{self.prefix}mtfx', f'{self.prefix}sx', f'{self.prefix}rx', f'{self.prefix}offset']]
        therm, bec = bimodalbose1d2centers(x,  *args, return_both=True)
        return therm, bec
    
class J_BimodalBose1DModel2Centers(Model1D):
    __doc__ = J_bimodalbose1d2centers.__doc__ + COMMON_GUESS_DOC if J_bimodalbose1d2centers.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(J_BimodalBose1DModel2Centers, self).__init__(func=J_bimodalbose1d2centers, *args, **kwargs)

        #1.2020569 is g3(1), see paper on the wall
        expressions = {self.prefix+'N_th': f'{self.prefix}amp_g * sqrt(2*pi) * {self.prefix}s_g * 1.2020569',
                       self.prefix+'N_bec': f'16/15*{self.prefix}r_tf*{self.prefix}amp_tf',
                       self.prefix+'N_tot': f'{self.prefix}N_th + {self.prefix}N_bec',
                       }
        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)



    def guess(self, data, X):
        data_smoothed = gaussian_filter1d(data, 5)
        amp_tot_guess = np.max(data_smoothed)
        ind_max = np.argmax(data_smoothed)
        center_guess = X[ind_max]
        data_left = data_smoothed[X<center_guess]
        data_right = data_smoothed[X>center_guess]
        limit_left_ind = np.where(data_left<amp_tot_guess/2)[0][-1]

        w_left = ind_max-limit_left_ind
        limit_right_ind = np.where(data_right<amp_tot_guess/2)[0][0] + len(data_left)
        w_right = limit_right_ind-ind_max
        limit_left_ind = np.max([ind_max-w_left*5, 0])
        limit_right_ind = np.min([ind_max+w_right*5, len(data)-1])
        offset_guess_left = np.average(data_smoothed[limit_left_ind:limit_left_ind + w_left])
        offset_guess_right = np.average(data_smoothed[limit_right_ind-w_right:limit_right_ind])
        offset_guess = (offset_guess_left + offset_guess_right)/2
        area_guess = np.sum(data_smoothed[limit_left_ind:limit_right_ind]-offset_guess)*(X[1]-X[0])
        amp_tot_guess = amp_tot_guess-offset_guess
        width_guess = area_guess/amp_tot_guess/np.sqrt(2*np.pi)
        params = self.make_params()
        
        params[self.prefix+'amp_g'].set(value=amp_tot_guess/2, max=amp_tot_guess*2, min=0)
        params[self.prefix+'amp_tf'].set(value=amp_tot_guess/2, max=amp_tot_guess*2, min=0)
        params[self.prefix+'r_tf'].set(value=width_guess, min=0, max=width_guess*3)
        params[self.prefix+'s_g'].set(value=width_guess*2, min=0, max=width_guess*3)
        c_min = np.max([center_guess-width_guess*5, X[0]])
        c_max = np.min([center_guess+width_guess*5, X[-1]])
        params[self.prefix+'c_g'].set(value=center_guess,min = c_min, max = c_max)
        params[self.prefix+'c_tf'].set(value=center_guess,min = c_min, max = c_max)
        return params


    def eval_bimodal_components(self, params, X):
        args = [params[k].value for k in [f'{self.prefix}amp_g', f'{self.prefix}amp_tf', f'{self.prefix}c_g', f'{self.prefix}c_tf', f'{self.prefix}s_g', f'{self.prefix}r_tf', f'{self.prefix}offset']]
        therm, bec = bimodalbose1d2centers(X,  *args, return_both=True)
        return therm, bec

    def plot_best(self,params,X,ax,invert=False):
        Y_th, Y_bec = self.eval_bimodal_components(params,X)
        offset = params[f'{self.prefix}offset'].value
        print(offset)
        Y_tot = Y_th + Y_bec - offset
        if invert:
            ax.plot(Y_tot,X,'r-',label='fit')
            ax.plot(Y_th,X,'y--',label='thermal')
            ax.plot(Y_bec,X,'m--',label='bec')
        else:
            ax.plot(X,Y_tot,'r-',label='fit')
            ax.plot(X,Y_th,'y--',label='thermal')
            ax.plot(X,Y_bec,'m--',label='bec')

def LZ_func_p1(Bscan, Bset, a, c):
    """
    Function that computes the LZ probability for the p1 state.
    """
    p = np.exp(-a*((Bscan + c)**2 + Bset**2))
    return  p**2

class LZModel_p1(Model1D):
    __doc__ = LZ_func_p1.__doc__ + COMMON_GUESS_DOC if LZ_func_p1.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(LZModel_p1, self).__init__(func=LZ_func_p1, *args, **kwargs)
        self.set_param_hint('amp', min=0)
        self.set_param_hint('s', min=0)

        expressions = {self.prefix+'N': f'{self.prefix}amp * {self.prefix}pixel_size**2 /{self.prefix}cross_section',
                       self.prefix+'sigma': f'{self.prefix}s*{self.prefix}pixel_size',
                       }
        
        for k, expr in expressions.items():
            self.set_param_hint(k, expr=expr)

    def guess(self, data, **kwargs):
        amp, m, s = guess_from_peak_1d(data, **kwargs)
        self.set_param_hint('amp', value=amp, max=data.max())
        self.set_param_hint('m', value=m)
        self.set_param_hint('s', value=s)

        return self.make_params()