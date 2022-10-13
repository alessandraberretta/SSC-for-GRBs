import numpy as np
from scipy import integrate
from scipy import optimize
import scipy.special as sc

class FluxSSC_Class:
    
    def __init__(self):
        '''
        Initialize FluxGeneratorClass. 
        '''
        self.mec2 = 8.187e-7  # ergs
        self.mec2eV = 5.11e5  # eV
        self.mpc2 = 1.5032e-3  # ergs
        self.eV2Hz = 2.418e14
        self.eV2erg = 1.602e12
        self.kB = 1.3807e-16  # erg/K
        self.h = 6.6261e-27  # erg*sec
        self.me = 9.1094e-28  # g
        self.mp = 1.6726e-24  # g
        self.G = 6.6726e-8  # dyne cm^2/g^2
        self.Msun = 1.989e33  # g
        self.Lsun = 3.826e33  # erg/s
        self.Rsun = 6.960e10  # cm
        self.pc = 3.085678e18  # cm
        self.e = 4.8032e-10  # statcoulonb
        self.re = 2.8179e-13  # cm
        self.sigmaT = 6.6525e-25  # cm^2
        self.sigmaSB = 5.6705e-5  # erg/(cm^2 s K^4 )
        self.Bcr = 4.414e13  # G
        self.c = 2.99792e10  # cm/s
        self.ckm = 299792  # km/s
        self.H0 = 67.4  # km s^-1 Mpc^-1 (Planck Collaboration 2018)
        self.omegaM = 0.315  # (Planck Collaboration 2018)

    def regionSize(self, dt, delta, z):
        R = delta*(self.c)*dt/(1+z)
        return R

    def T(self, x):
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

    def luminosityDistance(self, z):
        s = ((1.-(self.omegaM))/(self.omegaM))**(1./3.)
        dL = (((self.c)*(1+z))/((self.H0)*np.sqrt(s*(self.omegaM))))*((self.T(s))-(self.T(s/(1+z))))
        return dL

    def doppler(self, gamma, theta):  # theta in rad
        beta = np.sqrt(1. - 1./(gamma*gamma))
        d = 1./(gamma*(1-beta*np.cos(theta)))
        return d

    def getLogFreqArray(self, min, max, N):
        fre = np.logspace(min, max, num=N, endpoint=True)
        return fre
    
    def getLogGammaArray(self, min, max, N):
        ar = np.logspace(min, max, num=N, endpoint=True)
        return ar

    def electronDistPL(self, n0, gammam, gammaM, p):
        ga = np.logspace(gammam, gammaM, 200)
        ne = n0*np.power(ga, -p)
        return ne

    def nu_L(self, b):
        k = (self.e)/(2.*np.pi*(self.me)*(self.c))
        nuL = k*b
        return nuL

    def bessel(self, xs):
        o = 5./3.
        def bes(x): return sc.kv(o, x)
        r = integrate.quad(bes, xs, np.inf, limit=100)
        return r[0]*xs

    def nu_s(self, gamma_e, B):
        nuc = gamma_e*gamma_e*(self.nu_L(B))
        return nuc

    def nu_c(self, gamma_e, B):
        nuc = (3./2)*gamma_e*gamma_e*(self.nu_L(B))
        return nuc
    
    def singleElecSynchroPower(self, nu, gamma, B):
        n1 = 2.*np.pi*np.sqrt(3.)*(self.e)*(self.e)*(self.nu_L(B))/(self.c)
        # nus = self.nu_c(gamma, B)
        # x0 = nu/nus
        x0 = (nu/(3/2*gamma*gamma*(self.nu_L(B))))
        y0 = self.bessel(x0)
        bes = y0
        return bes

    def syncEmissivityKernPL(self, gamma, nu, p, B):
        # ne = np.power(gamma, -p)
        gamma_min = 2
        gamma_c = 1e8
        if gamma_min < gamma < gamma_c:
            ne = np.power(gamma, -p)
            k1 = ne*(self.singleElecSynchroPower(nu, gamma, B))
        elif gamma > gamma_c:
            ne = np.power(gamma, -p-1)
            k1 = ne*(self.singleElecSynchroPower(nu, gamma, B))
        # else:
        # ne = np.power(gamma, -p-1)
        # k1= ne*singleElecSynchroPower(nu, gamma, B)
        # ne = np.power(gamma, -p)
        # k1 = ne*singleElecSynchroPower(nu, gamma, B)
        return k1

    def syncEmissivity(self, nu_list_SYN):
        
        gammam = 100
        gammaM = 5e8 
        # p = P['p']
        p = 2.2
        n0 = 1e5 
        B = 1
        dt_syn = 1e6
        z = 1 
        d = self.doppler(100, 1/100)
        dl = self.luminosityDistance(1)
        R = self.regionSize(dt_syn, d, z)
        # d = self.doppler(P['gammaL'], P['theta_obs'])
        # R = self.regionSize(dt_syn, d, z)
        # dl = self.luminosityDistance(z)

        nuL = self.nu_L(B)
        f1 = []
        y = []
        assorb = []
        n1 = 2.*np.pi*np.sqrt(3.)*(self.e)*(self.e)*(nuL)/(self.c)
        k0 = (p+2)/(8*np.pi*(self.me))
        ar = self.getLogGammaArray(np.log10(gammam), np.log10(gammaM), 100)
        V_R = (4/3)*np.pi*np.power(R, 3)
        for f in nu_list_SYN:
            y1 = []
            yy = []
            for x in ar:
                asso = (self.singleElecSynchroPower(f, x, B))*pow(x, -(p+1))
                js = self.syncEmissivityKernPL(x, f, p, B)
                y1.append(js)
                yy.append(asso)
            r = integrate.simps(y1)
            al = integrate.simps(yy)
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
        return np.log10(f1), np.log10(y)

    def P_ssc_allnumeric(self, nu_list_SSC, P):

        B = 1 #Gauss
        gamma_min = 10
        gamma_max = 5e10
        dt_ssc = 1e5
        z = 1 
        d = self.doppler(P['gammaL'], P['theta_obs'])
        R = self.regionSize(dt_ssc, d, z)
        p = P['p'] 
        n0 = 1e5 
        dl = self.luminosityDistance(z)

        gamma_list = np.linspace(gamma_min, gamma_max, 100)
        nui_list = np.logspace(1, 10, 100)

        flux_ssc = []
        freq_plot = []

        for elm_nu in nu_list_SSC:
            list2 = []
            gamma_surv = []
            for elm_gamma in gamma_list:
                list1 = []
                nui_surv = []
                for elm_nui in nui_list:
                    q = (elm_nu)/(4*np.power(elm_gamma, 2)*elm_nui*(1-((self.h)*elm_nu)/(elm_gamma*(self.mec2))))
                    GAMMA = (4*elm_gamma*(self.h)*elm_nui)/(self.mec2)
                    if 1/(4*elm_gamma*elm_gamma) <= q <= 1:
                        tau = 2*(self.singleElecSynchroPower(elm_nui, elm_gamma, B))*pow(elm_gamma, -(p+1))*R
                        f0 = (9/4*(self.c))*((self.syncEmissivityKernPL(elm_gamma, elm_nui, p, B))/(self.singleElecSynchroPower(elm_nui, elm_gamma, B))*pow(elm_gamma, -(p+1)))*(1-(2/tau*tau)*(1-np.exp(-tau)*(tau+1)))
                        f1 = 2*(q*np.log(q))
                        f2 = 1
                        f3 = q
                        f4 = (-2*np.power(q,2))
                        f5 = (np.power(GAMMA,2)*np.power(q,2))*(1-q)/(2+2*GAMMA*q)
                        f_tot = 2*np.pi*np.power((self.re),2)*(self.c)*(elm_nu/(np.power(elm_nui,2)*np.power(elm_gamma,2)))*f0*(f1 + f2 + f3 + f4 + f5)
                        list1.append(f_tot)
                        nui_surv.append(elm_nui)
                if len(list1) != 0:
                    integral1 = integrate.simps(list1, x=nui_surv)
                    integral1_float = float(integral1)
                    if integral1_float > 1e-80:
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
                if integral2_float > 1e-85:
                    print('SECOND INTEGRATION ---->', integral1_float)
                    # flux_ssc.append(elm_nu*integral2_float*np.power(d,4)*pow(R,2)/(4*np.pi*dl*dl))
                    flux_ssc.append(elm_nu*integral2_float*(4/3)*pow(R,3)/(4*np.pi*dl*dl))
                    # flux_ssc.append(elm_nu*integral2_float*R*sigmaT)
                    freq_plot.append(elm_nu)  
        return np.log10(freq_plot), np.log10(flux_ssc)
        # np.log10(freq_plot),

    def P_ssc_allnumeric2(self, nu_list_SSC, P):

        B = 1 #Gauss
        gamma_min = 10
        gamma_max = 5e10
        dt_ssc = 1e5
        z = 1 
        d = self.doppler(P['gammaL'], P['theta_obs'])
        R = self.regionSize(dt_ssc, d, z)
        p = P['p'] 
        n0 = 1e5 
        dl = self.luminosityDistance(z)

        gamma_list = np.linspace(gamma_min, gamma_max, 100)
        nui_list = np.logspace(1, 10, 100)

        flux_ssc = []
        freq_plot = []

        for elm_nu in nu_list_SSC:
            list2 = []
            gamma_surv = []
            for elm_gamma in gamma_list:
                list1 = []
                nui_surv = []
                for elm_nui in nui_list:
                    q = (elm_nu)/(4*np.power(elm_gamma, 2)*elm_nui*(1-((self.h)*elm_nu)/(elm_gamma*(self.mec2))))
                    GAMMA = (4*elm_gamma*(self.h)*elm_nui)/(self.mec2)
                    if 1/(4*elm_gamma*elm_gamma) <= q <= 1:
                        tau = 2*(self.singleElecSynchroPower(elm_nui, elm_gamma, B))*pow(elm_gamma, -(p+1))*R
                        f0 = (9/4*(self.c))*((self.syncEmissivityKernPL(elm_gamma, elm_nui, p, B))/(self.singleElecSynchroPower(elm_nui, elm_gamma, B))*pow(elm_gamma, -(p+1)))*(1-(2/tau*tau)*(1-np.exp(-tau)*(tau+1)))
                        f1 = 2*(q*np.log(q))
                        f2 = 1
                        f3 = q
                        f4 = (-2*np.power(q,2))
                        f5 = (np.power(GAMMA,2)*np.power(q,2))*(1-q)/(2+2*GAMMA*q)
                        f_tot = 2*np.pi*np.power((self.re),2)*(self.c)*(elm_nu/(np.power(elm_nui,2)*np.power(elm_gamma,2)))*f0*(f1 + f2 + f3 + f4 + f5)
                        list1.append(f_tot)
                        nui_surv.append(elm_nui)
                if len(list1) != 0:
                    integral1 = integrate.simps(list1, x=nui_surv)
                    integral1_float = float(integral1)
                    if integral1_float > 1e-80:
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
                if integral2_float > 1e-85:
                    print('SECOND INTEGRATION ---->', integral1_float)
                    # flux_ssc.append(elm_nu*integral2_float*np.power(d,4)*pow(R,2)/(4*np.pi*dl*dl))
                    flux_ssc.append(elm_nu*integral2_float*(4/3)*pow(R,3)/(4*np.pi*dl*dl))
                    # flux_ssc.append(elm_nu*integral2_float*R*sigmaT)
                    freq_plot.append(elm_nu)  
        return np.log10(flux_ssc)               

