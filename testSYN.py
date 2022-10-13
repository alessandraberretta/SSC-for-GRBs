from cmath import isnan
import os
import sys
import numpy as np
from scipy import integrate
from scipy import optimize
import scipy.special as sc
import matplotlib.pyplot as plt
import mpmath
import sympy as sym
from sympy import besselk
from sympy import oo
from sympy import sqrt
from sympy.functions import exp
from tqdm import tqdm

mec2 = 8.187e-7  # ergs
mec2eV = 5.11e5  # eV
mpc2 = 1.5032e-3  # ergs
eV2Hz = 2.418e14
eV2erg = 1.602e12
kB = 1.3807e-16  # [erg/K]
# h = 6.6261e-27  # erg*sec
h = 4.136e-15  # eV*sec
me = 9.1094e-28  # g
mp = 1.6726e-24  # g
G = 6.6726e-8  # dyne cm^2/g^2
Msun = 1.989e33  # g
Lsun = 3.826e33  # erg/s
Rsun = 6.960e10  # cm
pc = 3.085678e18  # cm
e = 4.8032e-10  # statcoulonb
re = 2.8179e-13  # cm
sigmaT = 6.6525e-25  # cm^2
sigmaSB = 5.6705e-5  # erg/(cm^2 s K^4 )
Bcr = 4.414e13  # G
c = 2.99792e10  # cm/s
ckm = 299792  # km/s
H0 = 67.4  # km s^-1 Mpc^-1 (Planck Collaboration 2018)
omegaM = 0.315  # (Planck Collaboration 2018)

def regionSize(dt, delta, z):
    R = delta*c*dt/(1+z)
    return R

def T(x):
    # Adachi \& Kasai (2012) found that T(s) can be approximated as:
    b_1 = 2.64086441
    b_2 = 0.883044401
    b_3 = 0.0531249537
    c_1 = 1.39186078
    c_2 = 0.512094674
    c_3 = 0.0394382061
    T_s = np.sqrt(x)*((2+b_1*x**3 + b_2*x**6 + b_3*x**9) /
                      (1+c_1*x**3+c_2*x**6+c_3*x**9))
    return T_s

def luminosityDistance(z):
    s = ((1.-omegaM)/omegaM)**(1./3.)
    dL = ((ckm*(1+z))/(H0*np.sqrt(s*omegaM)))*(T(s)-T(s/(1+z)))
    return dL

def doppler(gamma, theta):  # theta in rad
    beta = np.sqrt(1. - 1./(gamma*gamma))
    d = 1./(gamma*(1-beta*np.cos(theta)))
    return d

def getLogFreqArray(min, max, N):
    fre = np.logspace(min, max, num=N, endpoint=True)
    return fre

def getLogGammaArray(min, max, N):
    ar = np.logspace(min, max, num=N, endpoint=True)
    return ar

def electronDistPL(n0, gammam, gammaM, p):
    ga = np.logspace(gammam, gammaM, 200)
    ne = n0*np.power(ga, -p)
    return ne

def nu_L(b):
    k = e/(2.*np.pi*me*c)
    nuL = k*b
    return nuL

def bessel(xs):
    o = 5./3.
    def bes(x): return sc.kv(o, x)
    r = integrate.quad(bes, xs, np.inf, limit=100)
    return r[0]*xs

def nu_s(gamma_e, B):
    nuc = gamma_e*gamma_e*nu_L(B)
    return nuc

def nu_c(gamma_e, B):
    nuc = (3./2)*gamma_e*gamma_e*nu_L(B)
    return nuc

def singleElecSynchroPower(nu, gamma, B):
    nuL = nu_L(B)
    n1 = 2.*np.pi*np.sqrt(3.)*e*e*nuL/c
    nus = nu_c(gamma, B)
    x0 = nu/nus
    y0 = bessel(x0)
    P = y0
    #print ("1--------------->",gamma,x0)
    return P

def syncEmissivityKernPL(gamma, nu, p, B):
    ne = np.power(gamma, -p)
    k1 = ne*singleElecSynchroPower(nu, gamma, B)
    return k1

def syncEmissivity(freq, gammam, gammaM, p, n0, B, R):

    nuL = nu_L(B)
    f1 = []
    y = []
    assorb = []
    n1 = 2.*np.pi*np.sqrt(3.)*e*e*nuL/c
    k0 = (p+2)/(8*np.pi*me)
    ar = getLogGammaArray(np.log10(gammam), np.log10(gammaM), 100)
    V_R = (4/3)*np.pi*np.power(R, 3)
    for f in freq:
        y1 = []
        yy = []
        for x in ar:
            asso = singleElecSynchroPower(f, x, B)*pow(x, -(p+1))
            js = syncEmissivityKernPL(x, f, p, B)
            y1.append(js)
            yy.append(asso)
            # print("------>",x,yy)
        r = integrate.simps(y1)
        al = integrate.simps(yy)
        if r > 1e-50:
            I = n1*r*n0 
            as1 = n1*al*k0*n0/pow(f, 2.)
            tau = as1*R
            if(tau > 0.1):
                assorb.append(f*I*R)
                I = I*R*(1.-np.exp(-tau))/tau
                y.append(f*I)
                # y1.append(alpha)
                f1.append(f)
            else:
                assorb.append(f*I*R)
                y.append(f*I*R)
                # y1.append(alpha)
                f1.append(f)
    # plt.plot(np.log10(f1), np.log10(y))
    # plt.plot(np.log10(f1), np.log10(assorb))
    # plt.plot(np.log10(f1), np.log10(y1))
    # plt.legend()
    # plt.ylim(35, 60)
    # plt.xlim(7., 25.)
    # plt.show()
    return np.log10(f1), np.log10(y)

def main():

    gammaL = 1500.  # bulk Lorentz
    dt = 1e3  # sec
    z = 1  # redshift
    thetaObs = 1./gammaL
    gamma_e = 100.  # lorentz factor of an electron
    B = 0.1  # gauss magnetic field
    gamma_min = 100.
    gamma_max = 1000000000.
    n0 = 1e-15
    p = 1.1

    # d = doppler(gammaL, thetaObs)
    d = 25 
    dl = luminosityDistance(z)
    nu_list_SYN = np.logspace(6, 27, num=200, endpoint=True)
    # R = regionSize(dt, d, z)
    R=2.3e16

    print(nu_list_SYN)

    freq_plot_syn, flux_plot_syn = syncEmissivity(nu_list_SYN, gamma_min, gamma_max, p, n0, B, R)

    plt.plot(freq_plot_syn, flux_plot_syn, c='red')
    plt.ylim(-15, -6)
    # plt.xlim(9., 28.)
    plt.show()

if __name__ == "__main__":
    main()