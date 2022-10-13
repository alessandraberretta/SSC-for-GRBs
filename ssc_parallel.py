from cmath import isnan
import os
import sys
import numpy as np
from scipy import integrate
from scipy import optimize
import scipy.special as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import multiprocessing as mp_lib

mec2 = 8.187e-7  # ergs
mec2eV = 5.11e5  # eV
mpc2 = 1.5032e-3  # ergs
eV2Hz = 2.418e14
eV2erg = 1.602e12
kB = 1.3807e-16  # [erg/K]
h = 6.6261e-27  # erg*sec
# h = 4.136e-15  # eV*sec
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
    dL = ((c*(1+z))/(H0*np.sqrt(s*omegaM)))*(T(s)-T(s/(1+z)))
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
    # x0 = nu/nus
    x0 = nu/(3/2*gamma*gamma*nuL)
    y0 = bessel(x0)
    P = y0
    return P

def syncEmissivityKernPL(gamma, nu, p, B):
    # ne = np.power(gamma, -p)
    gamma_min = 2
    gamma_c = 2e9
    if gamma_min < gamma < gamma_c:
      ne = np.power(gamma, -p)
      k1 = ne*singleElecSynchroPower(nu, gamma, B)
    elif gamma > gamma_c:
      ne = np.power(gamma, -p-1)
      k1 = ne*singleElecSynchroPower(nu, gamma, B)
    # else:
      # ne = np.power(gamma, -p-1)
      # k1= ne*singleElecSynchroPower(nu, gamma, B)
    # ne = np.power(gamma, -p)
    # k1 = ne*singleElecSynchroPower(nu, gamma, B)
    return k1

def syncEmissivity(freq, gammam, gammaM, p, n0, B, R, d, dl):
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
        #print (f,r)
        if r > 1e-90:
            I = n1*r*n0  # (r[0]/alpha)*(1.-exp(-tau))
            as1 = n1*al*k0*n0/pow(f, 2.)
            tau = as1*R
            if(tau > 1):
                # > 0.1
                assorb.append(f*I*R)
                I = I*R*(1.-np.exp(-tau))/tau
                y.append((f*I)/(dl*dl))
                # y1.append(alpha)
                f1.append(f)
            else:
                assorb.append(f*I*R)
                # y.append(f*I*R)
                y.append((f*I*R)/(dl*dl))
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

def F_syn(nu_list_SYN, B, p, R, n0, gamma_min, gamma_max, d, dl):

    gamma_list = getLogGammaArray(np.log10(gamma_min), np.log10(gamma_max), 100)
    nuL = nu_L(B)
    coeff_P_syn = (2.*np.pi*np.sqrt(3.)*e*e*nuL)/c
    k0 = (p+2)/(8*np.pi*me)
    flux_syn = []
    freq_plot = []
    assorb = []
    V_R = (4/3)*np.pi*np.power(R, 3)

    for elm_nu in tqdm(nu_list_SYN): 
        func1 = []
        func2 = []
        for elm_gamma in gamma_list:
            # x_1 = elm_nu/nu_c(elm_gamma, B)
            # alpha_PL = ((3*sigmaT)/(64*np.pi*me))*(np.power(nuL, (p-2)/2))*(np.power(elm_nu, -(p+4)/2))
            # tau = alpha_PL*R
            asso = singleElecSynchroPower(elm_nu, elm_gamma, B)*pow(elm_gamma, -(p+1))
            js = syncEmissivityKernPL(elm_gamma, elm_nu, p, B)
            # print('syn', js)
            func1.append(js)
            func2.append(asso)
        integral_simpson = integrate.simps(func1)
        al = integrate.simps(func2)
        if integral_simpson > 1e-50:
            I = coeff_P_syn*integral_simpson*n0 # (r[0]/alpha)*(1.-exp(-tau))
            # alpha_PL = ((3*sigmaT)/(64*np.pi*me))*(np.power(nuL, (p-2)/2))*(np.power(elm_nu, -(p+4)/2))
            alpha = coeff_P_syn*al*k0*n0/pow(elm_nu, 2.)
            tau = alpha*R
            if tau > 0.1:
                assorb.append(elm_nu*I*R)
                I = I*R*(1.-np.exp(-tau))/tau
                flux_syn.append(elm_nu*I*np.power(d,4)/dl*dl)
                freq_plot.append(elm_nu)
            else:
                flux_syn.append(elm_nu*I*R*np.power(d,4)/dl*dl)
                # flux_syn.append((elm_nu*I*R*np.power(d,4))/(4*np.pi*np.power(dl,2)))
                assorb.append(elm_nu*I*R)
                freq_plot.append(elm_nu)
            # flux_syn.append(elm_nu*I*R)
            # freq_plot.append(elm_nu)
    # freq_plot_syn = np.log10(freq_plot)
    # flux_plot_syn = np.log10(flux_syn)
    # plt.plot(np.log10(freq_plot), np.log10(flux_syn), c='blue')
    # plt.ylim(-10, 10)
    # plt.xlim(7., 25.)
    # plt.show()
    return np.log10(freq_plot), np.log10(flux_syn)

def first_int(nui , nu, gamma, p, B, R):
    
    q = (nu)/(4*np.power(gamma, 2)*nui*(1-(h*nu)/(gamma*mec2)))
    GAMMA = (4*gamma*h*nui)/(mec2)
    f_tot = -999
    
    if 1/(4*gamma*gamma) <= q <= 1:
       
        tau = 2*singleElecSynchroPower(nui, gamma, B)*pow(gamma, -(p+1))*R
        f0 = (9/4*c)*(syncEmissivityKernPL(gamma, nui, p, B)/singleElecSynchroPower(nui, gamma, B)*pow(gamma, -(p+1)))*(1-(2/tau*tau)*(1-np.exp(-tau)*(tau+1)))
        f1 = 2*(q*np.log(q))
        f2 = 1
        f3 = q
        f4 = (-2*np.power(q,2))
        f5 = (np.power(GAMMA,2)*np.power(q,2))*(1-q)/(2+2*GAMMA*q)
        f_tot = 2*np.pi*np.power(re,2)*c*(nu/(np.power(nui,2)*np.power(gamma,2)))*f0*(f1 + f2 + f3 + f4 + f5)
        
    return f_tot, nui


def P_ssc_allnumeric(B, gamma_min, gamma_max, nu_list_SSC, R, p, n0, dl, d):

    pool = mp_lib.Pool(mp_lib.cpu_count())

    nu_s_min = 1.2*1e6*np.power(gamma_min, 2)*B
    nu_s_max = 1.2*1e6*np.power(gamma_max, 2)*B

    gamma_list = np.linspace(gamma_min, gamma_max, 100)
    # nui_list = getLogFreqArray(5., 15., 400)
    nui_list = np.logspace(1, 10, 100)

    flux_ssc = []
    freq_plot = []
    
    for id, elm_nu in enumerate(nu_list_SSC):
        print('nu -------------->', elm_nu)
        list2 = []
        gamma_surv = []
        for elm_gamma in gamma_list:
            
            result = [pool.apply_async(first_int, args=(tmp_nui, elm_nu, elm_gamma, p, B, R)) for tmp_nui in nui_list]
            
            list1 = [elm.get()[0] for elm in result if elm.get()[0]!=-999]
            nui_surv = [elm.get()[1] for elm in result if elm.get()[0]!=-999]

            if len(list1) != 0:
                integral1 = integrate.simps(list1, x=nui_surv)
                integral1_float = float(integral1)
                if integral1_float > 1e-100:
                    gamma_c = 1e4
                    if gamma_min < elm_gamma < gamma_c:
                      all_electrons = n0*np.power(elm_gamma, -p)*integral1_float
                      gamma_surv.append(elm_gamma)
                      list2.append(all_electrons)
                    if elm_gamma > gamma_c:
                      all_electrons = n0*np.power(elm_gamma, -p-1)*integral1_float
                      gamma_surv.append(elm_gamma)
                      list2.append(all_electrons)
        if len(list2) != 0:
            integral2 = integrate.simps(list2, x=gamma_surv)
            integral2_float = float(integral2)
            if integral2_float > 1e-100:
                print('SECOND INTEGRATION ---->', integral1_float)
                flux_ssc.append(elm_nu*integral2_float*(4/3)*pow(R,3)/(4*np.pi*dl*dl))
                freq_plot.append(elm_nu)                 
    
    pool.close()

    return np.log10(freq_plot), np.log10(flux_ssc)
     

def plot_syn_ssc(x_b, y_b, x_t, y_t, x1, y1, x2, y2, freq, flux, errflux, freq_syn, flux_syn, freq_ssc, flux_ssc):
    # x = 24
    # y = -6
    # uplims = np.zeros(x.shape)
    # uplims[[1, 5, 9]] = True
    # plt.errorbar(x, y, xerr=1, yerr=1, marker='o', ms=8, ecolor='green', c='green', uplims=True)
    plt.plot(freq_syn, flux_syn, c='purple', label='SYN', linewidth=1)
    plt.plot(freq_ssc, flux_ssc, c='magenta', label='SSC', linewidth=1)
    plt.errorbar(freq, flux, yerr=errflux, linestyle='', c='green', fmt='o', ms=3, elinewidth=2, capsize=4, label='SED Swift of GRB090510')
    # plt.errorbar(freq_U, flux_U, yerr=errfluxU, linestyle='', c='orange', fmt='o', ms=3, elinewidth=2, capsize=4, label='SED Swift-UVOT of GRB090510')
    # plt.loglog(freq_lat,E**2*(F+Ferr), c='cornflowerblue', linewidth=1)
    # plt.loglog(freq_lat,E**2*abs(F-Ferr), c='cornflowerblue', linewidth=1)
    # plt.plot(freq_lat,E*E*F, c='cornflowerblue')
    plt.plot(x_b, y_b, '-', c='cornflowerblue', linewidth=1, label='Butterfly Fermi-LAT of GRB090510')
    plt.plot(x_t,y_t,'-', c='cornflowerblue', linewidth=1)
    plt.plot(x1, y1, c='cornflowerblue', linewidth=1)
    plt.plot(x2, y2, c='cornflowerblue', linewidth=1)
    # xerr=errfreq, 
    # plt.plot(freq, flux, c='green', label='Fermi-LAT sensitivity')
    # plt.plot(freq_ssc, flux_ssc, c='blue', label='SSC')
    # plt.ylabel(r'$\log_{10}(\nu F_{\nu}) [erg/cm^2/s]$', fontsize=15)
    plt.ylabel(r'$\nu F_{\nu} [erg/cm^2/s]$', fontsize=15)
    plt.xlabel(r'$frequency [Hz]$', fontsize=15)
    plt.legend(fontsize=8, loc='upper left')
    plt.xlim(1e14, 1e29)
    plt.ylim(1e-16, 1e-6)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('SSC_sed_GRB090510.png', dpi=500)
    plt.title('SSC model on multi-wavelength data of GRB 090510')
    plt.show()

def main():

    gamma_e = 100.
    # B = 1
    B = 1
    # gamma_min_syn = 1e5
    gamma_min_syn = 3e5
    # gamma_max_syn = 5e9
    gamma_max_syn = 1e10
    # 1e8
    gamma_min_ssc = 10
    # gamma_max_ssc = np.inf
    gamma_max_ssc = 5e10
    # 1e11
    # gammaL = 100
    gammaL = 85
    # dt_syn = 1e6
    dt_syn = 1e6
    dt_ssc = 2e1
    z = 0.9
    # z = 1
    thetaObs = 1./gammaL
    n0 = 9e3
    # p = 2.2
    # p = 2.1
    p = 2.00

    # t=10 #s

    # gamma_c = (6*np.pi*me*c)/(sigmaT*gamma_e*t*B*B)
    # print(gamma_c)

    d = doppler(gammaL, thetaObs)
    dl = luminosityDistance(z)
    nu_list_SYN = np.logspace(8, 27, num=300, endpoint=True)
    # fre = getLogFreqArray(5., 18., 400)
    # nu_list_SSC = getLogFreqArray(10., 30., 100)
    nu_list_SSC = np.logspace(17, 29, num=80, endpoint=True)
    # nu_list_SSC = np.logspace(12., 22., 30, endpoint=True)
    R_syn = regionSize(dt_syn, d, z)
    R_ssc = regionSize(dt_ssc, d, z)
    # R = 2.3e16
    '''
    DF = pd.read_csv('LAT_sensitivity_lb00.csv')
    energy = DF['Energy[MeV]'].values
    flux = DF['[erg cm^{-2} s^{-1}]'].values
    freq_log = np.log10(energy/4.13e-21)
    flux_log = np.log10(flux)
    '''

    df = pd.read_csv('/Users/alessandraberretta/JetFit/xrt_090510_sorted.csv')
    freq_GRB = df['freq'].values
    flux_GRB = df['nufnu'].values
    # freq_err_GRB = df['freq_err'].values
    flux_err_GRB = df['nufnu_err'].values
    '''
    freq_UVOT = 8.57037845E+14
    flux_UVOT = 1.34422882E-14
    flux_err_UVOT = 2.16000751E-15

    f = lambda E,N0,E0,gamma: N0*(E/E0)**(-1*gamma)
    N0_err = 0.1
    N0 = 1 
    E0 = 1
    gamma = 2+2.5
    gamma_err = 0.06

    f = lambda E,N0,E0,gamma: N0*(E/E0)**(-1*gamma)
    cov_gg = gamma_err*gamma_err
    ferr = lambda E,F,N0,N0err,E0,cov_gg: F*np.sqrt(N0err**2/N0**2+((np.log(E/E0))**2)*cov_gg)

    E = np.logspace(8,10,100, endpoint=True)
    F = f(E,N0,E0,gamma)*1e11
    Ferr = ferr(E,F,N0,N0_err,E0,cov_gg)

    nu_lat = np.logspace(22,24,100)

    point1 = [np.min(nu_lat), np.max(E**2*(F+Ferr))]
    point2 = [np.min(nu_lat), np.max(E**2*abs(F-Ferr))]
    x_values=[point1[0], point2[0]]
    y_values=[point1[1], point2[1]]

    point3 = [np.max(nu_lat), np.min(E**2*(F+Ferr))]
    point4 = [np.max(nu_lat), np.min(E**2*abs(F-Ferr))]
    x_values_=[point3[0], point4[0]]
    y_values_=[point3[1], point4[1]]
    '''

    df = pd.read_csv('/Users/alessandraberretta/JetFit/bowtie_df_GRB120915A.csv')

    x_b = df['x_bot'].values
    x_t = df['x_top'].values
    y_b = df['y_bot'].values/1e1
    y_t = df['y_top'].values/1e1

    point1 = [np.min(x_t), np.max(y_t)]
    point2 = [np.min(x_b), np.max(y_b)]
    x_values=[point1[0], point2[0]]
    y_values=[point1[1], point2[1]]

    point3 = [np.max(x_t), np.min(y_t)]
    point4 = [np.max(x_b), np.min(y_b)]
    x_values_=[point3[0], point4[0]]
    y_values_=[point3[1], point4[1]]


    # freq_plot_syn, flux_plot_syn = F_syn(nu_list_SYN, B, p, R, n0, gamma_min, gamma_max_syn, d, dl)
    freq_plot_syn, flux_plot_syn = syncEmissivity(nu_list_SYN, gamma_min_syn, gamma_max_syn, p, n0, B, R_syn, d, dl)
    #print(freq_plot_syn)
    
    freq_plot_ssc, flux_plot_ssc = P_ssc_allnumeric(B, gamma_min_ssc, gamma_max_ssc, nu_list_SSC, R_ssc, p, n0, dl, d)
    #print('nu_list', freq_plot_ssc, 'flux', flux_plot_ssc)
    plot_syn_ssc(x_b, y_b, x_t, y_t, x_values, y_values, x_values_, y_values_, freq_GRB, flux_GRB, flux_err_GRB, np.power(10,freq_plot_syn), np.power(10,flux_plot_syn), np.power(10,freq_plot_ssc), np.power(10,flux_plot_ssc))


if __name__ == "__main__":
    main()

