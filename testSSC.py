from cmath import isnan
import os
import sys
# from numba import jit
import numpy as np
from scipy import integrate
from scipy import optimize
import scipy.special as sc
import matplotlib.pyplot as plt
# import mpmath
# import sympy as sym
# from sympy import besselk
# from sympy import oo
# from sympy import sqrt
# from sympy.functions import exp
from tqdm import tqdm

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

def get_nu_i_min(gamma_e, B, gamma_min, gamma, fre):
    nus = nu_s(gamma_e, B)
    nus_min = 1.2*1e6*np.power(gamma_min, 2)*B
    f_nu_i_min = []
    for elm in fre:
        f_nu_i_min.append(
            nus_min*(nus)/(4*np.power(gamma, 2)*(1-(h*elm)/(gamma*mec2))))
    max_f_nu_i_min = max(f_nu_i_min)
    return max_f_nu_i_min

def get_nu_i_max(gamma_e, B, gamma_max, gamma, fre):
    nus = nu_s(gamma_e, B)
    nus_max = 1.2*1e6*np.power(gamma_max, 2)*B
    f_nu_i_max = []
    for elm in fre:
        f_nu_i_max.append(nus_max*(nus)/(1-(h*elm)/(gamma*mec2)))
    max_f_nu_i_max = max(f_nu_i_max)
    return max_f_nu_i_max

def luminosity_ssc(R, j_ssc):
    V_R = (4/3)*np.pi*np.power(R, 3)
    L_ssc = (4*np.pi*V_R*j_ssc)
    return L_ssc

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
        #print (f,r)
        if r > 1e-50:
            I = n1*r*n0  # (r[0]/alpha)*(1.-exp(-tau))
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

def GAMMA_fc(gamma):
    f1 = lambda nui: (4*gamma*h*nui)/(mec2)
    return f1

def q_fc(nu, gamma):
    f0 = lambda nui: (nu)/(4*np.power(gamma, 2)*nui*(1-(h*nu)/(gamma*mec2)))
    return f0

def A1(nu, gamma): 
    a1 = 2*np.pi*np.power(re,2)*c*(nu/np.power(gamma,2))
    return a1

def P_ssc(R, B, n0, nu, p, gamma_min, gamma_max, gamma):

    V_R = (4/3)*np.pi*np.power(R, 3)
    nuL = nu_L(B)
    A = 8*np.pi*np.power(re,2)*c*h
    k0 = (p+2)/(8*np.pi*me)
    coeff_P_syn = (2.*np.pi*np.sqrt(3.)*e*e*nuL)/c
    nu_s_min = 1.2*1e6*np.power(gamma_min, 2)*B
    nu_s_max = 1.2*1e6*np.power(gamma_max, 2)*B
    nus = nu_s(gamma, B)
    nui_min = max(nu_s_min, (nus)/(4*np.power(gamma, 2)*(1-(h*nu)/(gamma*mec2))))
    nui_max = max(nu_s_max, (nus)/(1-(h*nu)/(gamma*mec2)))
    nusmin = (nu*mec2)/(4*gamma*(gamma*mec2-h*nu))

    q = lambda nui: (nu)/(4*np.power(gamma, 2)*nui*(1-(h*nu)/(gamma*mec2)))
    GAMMA = lambda nui: (4*gamma*h*nui)/(mec2)

    f0 = lambda nui: (3/4)*R*syncEmissivityKernPL(gamma, nui, p, B)-(9/16)*np.power(R,2)*syncEmissivityKernPL(gamma, nui, p, B)*singleElecSynchroPower(nui, gamma, B)*pow(gamma, -(p+1))*np.power(1/nui,2)
    f1 = lambda nui: f0(nui)*2*(q(nui)*np.log(q(nui)))
    f2 = lambda nui: f0(nui)
    f3 = lambda nui: f0(nui)*q(nui)
    f4 = lambda nui: f0(nui)*(-2*np.power(q(nui),2))
    f5 = lambda nui: f0(nui)*(np.power(GAMMA(nui),2)*np.power(q(nui),2))*(1-q(nui))/(2+2*GAMMA(nui)*q(nui))

    f_tot = lambda nui: (nu/(np.power(nui,2)*np.power(gamma,2)))*(f0(nui))*(f1(nui) + f2(nui) + f3(nui) + f4(nui) + f5(nui))
    
    r = integrate.quad(f_tot, 1e3, 1e15)

    return r[0]

def P_ssc_allnumeric(B, gamma_min, gamma_max, nu_list_SSC, R, p, n0, dl, d):

    nu_s_min = 1.2*1e6*np.power(gamma_min, 2)*B
    nu_s_max = 1.2*1e6*np.power(gamma_max, 2)*B

    gamma_list = np.linspace(gamma_min, gamma_max, 200)
    # nui_list = getLogFreqArray(5., 15., 400)
    nui_list = np.logspace(6, 16, 200)

    flux_ssc = []
    freq_plot = []
    
    for id, elm_nu in enumerate(nu_list_SSC):
        print('nu -------------->', elm_nu)
        list2 = []
        gamma_surv = []
        for elm_gamma in gamma_list:
            # print('gamma --->', elm_gamma)
            nus = nu_s(elm_gamma, B)
            nui_min = max(nu_s_min, (nus)/(4*np.power(elm_gamma, 2)*(1-(h*elm_nu)/(elm_gamma*mec2))))
            nui_max = max(nu_s_max, (nus)/(1-(h*elm_nu)/(elm_gamma*mec2)))
            nusmin = (elm_nu*mec2)/(4*elm_gamma*(elm_gamma*mec2-h*elm_nu))
            # nui_list = np.linspace(nui_min, nui_max, endpoint=True)
            list1 = []
            nui_surv = []
            for elm_nui in nui_list:
                q = (elm_nu)/(4*np.power(elm_gamma, 2)*elm_nui*(1-(h*elm_nu)/(elm_gamma*mec2)))
                GAMMA = (4*elm_gamma*h*elm_nui)/(mec2)
                if 1/(4*elm_gamma*elm_gamma) <= q <= 1:
                    # alpha_delta = (3*sigmaT*(np.power(B,2)/8*np.pi))/(64*np.pi*me)*np.power(e*B/(2*np.pi*me*c),(p-2)/2)*np.power(elm_nui, (-p-4)/2)
                    # alpha_delta = syncEmissivityKernPL(elm_gamma, elm_nui, p, B)*((p+2)*n0/(8*np.pi*me*elm_nui*elm_nui))*asso
                    # f0 = (3/4)*R*syncEmissivityKernPL(elm_gamma, elm_nui, p, B)-(9/16)*np.power(R,2)*syncEmissivityKernPL(elm_gamma, elm_nui, p, B)*singleElecSynchroPower(elm_nui, elm_gamma, B)*pow(elm_gamma, -(p+1))*(p+2)*n0/(8*np.pi*me*np.power(elm_nui,2))
                    # f0 = (3/16*np.pi)*R*syncEmissivityKernPL(elm_gamma, elm_nui, p, B)
                    tau = 2*singleElecSynchroPower(elm_nui, elm_gamma, B)*pow(elm_gamma, -(p+1))*R
                    f0 = (9/4*c)*(syncEmissivityKernPL(elm_gamma, elm_nui, p, B)/singleElecSynchroPower(elm_nui, elm_gamma, B)*pow(elm_gamma, -(p+1)))*(1-(2/tau*tau)*(1-np.exp(-tau)*(tau+1)))
                    # - (9/16)*np.power(R,2)*alpha_delta
                    # 
                    # f0 = L_syn
                    # freq, flux_syn = F_syn(nu_list_SYN, B, p, R, n0, gamma_min, gamma_max, dl)
                    f1 = 2*(q*np.log(q))
                    f2 = 1
                    f3 = q
                    f4 = (-2*np.power(q,2))
                    f5 = (np.power(GAMMA,2)*np.power(q,2))*(1-q)/(2+2*GAMMA*q)
                    f_tot = 2*np.pi*np.power(re,2)*c*(elm_nu/(np.power(elm_nui,2)*np.power(elm_gamma,2)))*f0*(f1 + f2 + f3 + f4 + f5)
                    # func2.append(asso)
                    # print('nui_surv', elm_nui)
                    list1.append(f_tot)
                    # print('total_integrand', f_tot)
                    nui_surv.append(elm_nui)
                # else:
                    # list1.append(-999)
            if len(list1) != 0:
                # list1_arr = np.array(list1)
                # print(list1_arr)
                # mask = list1 != -999
                integral1 = integrate.simps(list1, x=nui_surv)
                integral1_float = float(abs(integral1))
                # print('FIRST INTEGRATION < 1e-50 ---->', integral1_float)
                if integral1_float > 1e-50:
                    # print('FIRST INTEGRATION ---->', integral1_float)
                    all_electrons = n0*np.power(elm_gamma, -p)*integral1_float
                    gamma_surv.append(elm_gamma)
                    list2.append(all_electrons)
                # else: 
                    # list2.append(-999)
        if len(list2) != 0:
            # list2_arr = np.array(list2)
            # mask2 = list2 != -999
            integral2 = integrate.simps(list2, x=gamma_surv)
            integral2_float = float(integral2)
            if integral2_float > 1e-50:
                print('SECOND INTEGRATION ---->', integral1_float)
                flux_ssc.append(elm_nu*integral2_float*np.power(d,4)/(4*np.pi*dl*dl))
                freq_plot.append(elm_nu)                 
    return np.log10(freq_plot), np.log10(flux_ssc)

def emissivity_SSC(R, B, n0, nu, p, gamma_min, gamma_max, gamma):
    ne = np.power(gamma, -p)
    #q_nui = lambda nui: (nu)/(4*np.power(gamma, 2)*nui*(1-(h*nu)/(gamma*mec2)))
    r1 =ne*P_ssc(R, B, n0, nu, p, gamma_min, gamma_max, gamma)
    # (2*np.pi*np.power(re,2)*c*(nu/np.power(gamma,2)))
    # print('j_ssc:', r1)
    return r1

def sympyintegral(R, B, n0, p, gamma_min, gamma_max, nu_list_SSC, dl):
    
    flux_ssc = []
    freq_plot = []
    gamma_list = np.linspace(gamma_min, gamma_max, 100)
    V_R = (4/3)*np.pi*np.power(R, 3)

    for elm_nu in tqdm(nu_list_SSC):
        j_ssc_list = [] 
        for elm_gamma in tqdm(gamma_list):
            j_ssc = emissivity_SSC(R, B, n0, elm_nu, p, gamma_min, gamma_max, elm_gamma)
            # print('j_ssc', j_ssc)
            j_ssc_list.append(j_ssc)
            # for j in j_ssc_list: 
                # if np.isnan(j) == True: 
                    # j_ssc_list.remove(j)
        integral_simpson1 = integrate.simps(j_ssc_list)
        # print('integral_simps_ssc_1:', integral_simpson1)
        if np.isnan(integral_simpson1) == True:
            print('integral_simps_ssc_1:', integral_simpson1)
            print('gamma', elm_gamma)
            print('nu:', elm_nu)
            print('j_ssc_list', j_ssc_list)
            # sys.exit()
        if integral_simpson1 > 1e-50:
            # print('integral_simps_ssc_2:', integral_simpson1)
            I1 = integral_simpson1*n0
            # L_ssc = (4*np.pi*V_R*I1)
            # flux_ssc.append((elm_nu*L_ssc*np.power(d,4))/(4*np.pi*np.power(dl,2)))
            flux_ssc.append(elm_nu*I1/np.power(dl,2))
            freq_plot.append(elm_nu)
    # freq_plot_ssc = np.log10(freq_plot)
    # flux_plot_ssc = np.log10(flux_ssc)       
    # plt.plot(np.log10(freq_plot), np.log10(flux_ssc), c='red')
    # plt.xlim(12., 24.)
    # plt.show()
    return np.log10(freq_plot), np.log10(flux_ssc)

def plot_syn_ssc(freq_syn, flux_syn, freq_ssc, flux_ssc):

    plt.plot(freq_syn, flux_syn, c='red')
    # print(flux_syn)
    plt.plot(freq_ssc, flux_ssc, c='blue')
    print(flux_ssc)
    # plt.ylim(-20, -6)
    # plt.xlim(7., 25.)
    plt.show()

def main():

    # gamma_e = 100.
    B = 0.1
    gamma_min = 100.
    gamma_max_syn = 1000000000.
    gamma_max_ssc = 1000000000.
    gammaL = 1500.
    dt = 1e3
    z = 1.0
    thetaObs = 1./gammaL
    n0 = 1e-10
    p = 1.1
    # nu = 1e17

    # d = doppler(gammaL, thetaObs)
    d = 25 
    dl = luminosityDistance(z)
    nu_list_SYN = np.logspace(6, 27, num=200, endpoint=True)
    # fre = getLogFreqArray(5., 18., 400)
    # nu_list_SSC = getLogFreqArray(10., 30., 100)
    nu_list_SSC = np.logspace(20, 29, num=50, endpoint=True)
    # nu_list_SSC = np.logspace(12., 22., 20, endpoint=True)
    # R = regionSize(dt, d, z)
    R = 2.3e16

    # ghisellini_spectrum(gamma_min, gamma_max, nu_list_SSC, p, R, n0, B)

    # freq_plot_syn, flux_plot_syn = F_syn(nu_list_SYN, B, p, R, n0, gamma_min, gamma_max_syn, d, dl)
    freq_plot_syn, flux_plot_syn = syncEmissivity(nu_list_SYN, gamma_min, gamma_max_syn, p, n0, B, R)
    # freq_plot_ssc, flux_plot_ssc = sympyintegral(R, B, n0, p, gamma_min, gamma_max_ssc, nu_list_SSC, dl)
    freq_plot_ssc, flux_plot_ssc = P_ssc_allnumeric(B, gamma_min, gamma_max_ssc, nu_list_SSC, R, p, n0, dl, d)
    # F_syn(nu_list_SYN, B, p, R, n0, gamma_min, gamma_max, d, dl)
    # sympyintegral(B, R, d, dl, gamma_min, gamma_max, n0, p, nu_list_SSC)
    plot_syn_ssc(freq_plot_syn, flux_plot_syn, freq_plot_ssc, flux_plot_ssc)
    # freq_plot_ssc, flux_plot_ssc


if __name__ == "__main__":
    main()

