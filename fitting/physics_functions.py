#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Create: 01-2019 - Carmelo Mordini <carmelo> <carmelo.mordini@unitn.it>
"""
Common functions
Disclaimer: all quantities must go in SI units
"""

import numpy as np
from scipy.constants import hbar, pi, atomic_mass, Boltzmann as kB, physical_constants
aB = physical_constants['Bohr radius'][0]

mass = 23*atomic_mass
a_scatt = 52*aB
g_int = 4*pi*hbar**2*a_scatt/mass
z2 = 1.644934066
z3 = 1.202056903

def tof_temperature_ax(t, sigma, omega):
    Tx = mass/kB * omega**2 * sigma**2 /(1 + (omega*t)**2) # K
    return Tx

def tof_temperature(t, sigma_x, sigma_y, omega_x, omega_y):
    """
    omega_y is meant to be the tight axis of the trap
    """
    tau = omega_y * t
    Tx = tof_temperature_ax(t, sigma_x, omega_x)
    Ty = tof_temperature_ax(t, sigma_y, omega_y)
    T0 = (2*tau**2 * Tx + (1+tau**2) * Ty)/(1 + 3*tau**2)
    return T0

# def calc_eta(df):
    # eta = 0.5*z3**(1./3) * (15*(df['N']*1e3)**(1./6) * a_scatt/a_ho)**(2./5)
    # return eta

def calc_Tc(N, omega_x, omega_y):
    Tc = hbar * (omega_x*omega_y**2 * N/z3)**(1./3)/kB
    return Tc