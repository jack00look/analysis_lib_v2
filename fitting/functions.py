#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 12-2018 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>

"""
A library of commonly used functions. This is shared between the modules

    * fitting
    * imageprocess
"""

import numpy as np
from scipy.special import spence, erf
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.stats import moment
from mpmath import polylog

try:
    import mpmath as mp #fast, low precision implementation
    from functools import partial

    _g32 = partial(mp.fp.polylog, 3/2)
    _g52 = partial(mp.fp.polylog, 5/2)
    _g22 = partial(mp.fp.polylog, 2)
    _g3 = partial(mp.fp.polylog, 3)


    g32 = np.vectorize(_g32, otypes=[np.float64,], doc="polylog(3/2, x). Vectorized by numpy.")
    g52 = np.vectorize(_g52, otypes=[np.float64,], doc="polylog(5/2, x). Vectorized by numpy.")
    g3 = np.vectorize(_g3, otypes=[np.float64,], doc="polylog(3, x). Vectorized by numpy.")

except ImportError:
    #FIXME: find an alternative for this
    print('mpmath module not found. The g32 and g52 functions will not be availlabe')
    def g32():
        raise NotImplementedError
    def g52():
        raise NotImplementedError
    def g22():
        raise NotImplementedError
    def g3():
        raise NotImplementedError


def printf_error(name, value, fmt=None):
    print(f"{name}: {value:.4uf} ({100*value.s/value.n:.2f} %)")

def g2(x):
    """ polylog(2, x) implemented via scipy.special.spence.
        This translation is due to scipy's definition of the dilogarithm
    """
    return spence(1.0 - x)

#def g52(x):
#    """
#    g5/2(x) created in a stupid efficient way
#    """
#    return np.array([float(np.real(polylog(5/2, i))) for i in x])

z2 = g2(1)

def _gaussian_1d_order(x, amp, x0, sigma, order=0,):
    """
    Computes a 1D Gaussian convolution kernel.
    This is actually a rewrite of _gaussian_kernel1d as implemented in scipy.filters == 1.2.0
    https://github.com/scipy/scipy/blob/v1.2.0/scipy/ndimage/filters.py#L136
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    x = x - x0
    phi_x = np.exp(p(x),)
    if order > 0:
        q = np.polynomial.Polynomial([1])
        p_deriv = p.deriv()
        for _ in range(order):
            # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
            # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
            q = q.deriv() + q * p_deriv
        phi_x *= q(x)
    return amp*phi_x

def guess_from_peak_2d(image, X=None, Y=None):
    """
        Guess and returns the initial 2d peak-model parameters
    """
    height, width = image.shape
    # image = gaussian_filter(image, sigma=(height+width)/30)
#    ym, xm = np.unravel_index(np.nanargmax(image), image.shape)

    ym, xm = center_of_mass(image.clip(0))
    xm = int(xm)
    ym = int(ym)
    amp = 3*image.mean()
    if X is None:
        mx = xm
        x = np.arange(width)-mx
    else:
        mx = X[ym, xm]
        x = X[ym,:]
    if Y is None:
        my = ym
        y = np.arange(height)-my
    else:
        my = Y[ym, xm]
        y = Y[:,xm]
    sx = np.nanstd((x-mx)*image[ym,:]/amp)
    sy = np.nanstd((y-my)*image[:,xm]/amp)
    return [amp, mx, my, sx, sy]

def guess_from_peak_1d(image, X=None):
    """
        Guess and returns the initial 1d peak-model parameters
    """
    image = image.clip(0)
    xm = center_of_mass(image.clip(0))[0]
    try:
        xm = int(xm)

    except ValueError:
        xm = int(image.size/2)
    amp = image.max()

    if X is None:
        mx = xm
        X = np.arange(image.size)
    else:
        mx = X[xm]
    sx = np.sqrt(((X-mx)**2*image/image.sum()).sum())
    return [amp, mx, sx]

###### gaussian

def gaussian1d(X, amp, m, s, offset=0,):
    return amp*np.exp(-(X-m)**2/(2*s**2)) + offset

def skewedgaussian1d(X, amp, m, s, gamma, offset = 0):
    return amp*np.exp(-(X-m)**2/(2*s**2))*(1+erf(gamma*(X-m)/(s*np.sqrt(2)))) + offset

def gaussian2d(X, Y, amp, mx, my, sx, sy, offset=0):
    """gaussian2d(X, Y, amplitude, mx, my, sx, sy)
    A 2d gaussian function.
    """
    return amp * np.exp(-(X-mx)**2/(2*sx**2) -(Y-my)**2/(2*sy**2)) + offset

def flat2d(X,Y,offset):
    return offset


def bose2d(X, Y, amp, mx, my, sx, sy, offset=0):
    """
    Bose function for the integrated density profile of a (not so hot) thermal cloud.
    """
    return amp * g2(gaussian2d(X, Y, 1., mx, my, sx, sy, 0)) + offset

def bose1d(X, amp, mx, sx, offset=0):
    """
    Bose function for the integrated 1d density profile of a (not so hot) thermal cloud.
    """
    return amp * g52(gaussian1d(X, 1, mx, sx, 0))+offset

def thomasfermi1d(X, amp, mx, rx, offset=0):
    """thomasfermi1d(X, amplitude, mx, rx)
        1D inverted parabola profile
    """
    b = np.maximum(0, 1-((X-mx)/rx)**2)**2
    return amp*b + offset

def thomasfermi2d(X, Y, amp, mx, my, rx, ry, offset=0):
    """thomasfermi2d(X, Y, amplitude, mx, my, sx, sy)
       A 2d Thomas-Fermi (integrated inverted parabola) function.
    """
    b = np.maximum(0, 1 -((X-mx)/rx)**2 -((Y-my)/ry)**2)**(3./2)
    return amp*b + offset

def rotate(X, Y, mx, my, alpha):
    alpha = (np.pi / 180.) * float(alpha)
    xr = + (X-mx) * np.cos(alpha) - (Y-my) * np.sin(alpha)
    yr = + (X-mx) * np.sin(alpha) + (Y-my) * np.cos(alpha)
    return xr + mx, yr + my

def thomasfermi2d_rotate(X, Y, amp, mx, my, rx, ry, offset, alpha):
    """thomasfermi2d(X, Y, amplitude, mx, my, sx, sy)
       A 2d Thomas-Fermi (integrated inverted parabola) function.
       alpha must be in radians
    """
    X, Y = rotate(X, Y, mx, my, alpha)
    b = 1 -((X-mx)/rx)**2 -((Y-my)/ry)**2
    b = np.maximum(0, b**3)
    return amp * np.sqrt(b) + offset

def bimodal1d(X, A, B, mx, sx, rx, offset=0, return_both=False):
    g = A*np.exp(-(X-mx)**2/(2*sx**2))
    b = B*np.maximum(0, 1 - ((X-mx)/rx)**2)**2
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset

def bimodal2d(X, Y, amp_g, amp_tf, mx, my, sx, sy, rx, ry, offset=0, return_both=False):
    g = gaussian2d(X, Y, amp_g, mx, my, sx, sy)
    b = thomasfermi2d(X, Y, amp_tf, mx, my, rx, ry,)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset

def bimodalbose1d(X, amp_g, amp_tf, mx, sx, rx, offset=0, return_both=False):
    g = bose1d(X, amp_g, mx, sx, 0)
    b = thomasfermi1d(X, amp_tf, mx, rx, 0)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset


def bimodalbose1d2centers(X, amp_g, amp_tf, mgx, mtfx, sx, rx, offset=0, return_both=False):
    g = bose1d(X, amp_g, mgx, sx, 0)
    b = thomasfermi1d(X, amp_tf, mtfx, rx, 0)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset

def J_bimodalbose1d2centers(X, amp_g, amp_tf, c_g, c_tf, s_g, r_tf, offset=0, return_both=False):
    g = bose1d(X, amp_g, c_g, s_g, 0)
    b = thomasfermi1d(X, amp_tf, c_tf, r_tf, 0)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset
    
def J_bimodalbose2d2centers(X, Y, amp_g, amp_tf, mbx, mtx, mby, mty, sx, sy, rx, ry, offset=0, return_both=False):
    g = bose2d(X, Y, amp_g, mtx, mty, sx, sy)
    b = thomasfermi2d(X, Y, amp_tf, mbx, mby, rx, ry)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset



def bimodalbose2d(X, Y, amp_g, amp_tf, mx, my, sx, sy, rx, ry, offset=0, return_both=False):
    g = bose2d(X, Y, amp_g, mx, my, sx, sy)
    b = thomasfermi2d(X, Y, amp_tf, mx, my, rx, ry)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset

def bimodalbose2d2centers(X, Y, amp_g, amp_tf, mbx, mtx, mby, mty, sx, sy, rx, ry, offset=0, return_both=False):
    g = bose2d(X, Y, amp_g, mtx, mty, sx, sy)
    b = thomasfermi2d(X, Y, amp_tf, mbx, mby, rx, ry)
    if return_both:
        return g + offset, b + offset
    else:
        return g + b + offset

def LZ_func_p1(Bscan, Bset, a, c):
    """
    Function that computes the LZ probability for the p1 state.
    """
    p = np.exp(-a*((Bscan + c)**2 + Bset**2))
    return  p**2

def J_LZ_p1(Bscan, Bscan_res, Bother_res, sigma_B):
    """
    Function that computes the LZ probability for the p1 state.
    """
    v = np.exp(-0.5*(((Bscan + Bscan_res)**2 + Bother_res**2) / sigma_B**2))
    return  v