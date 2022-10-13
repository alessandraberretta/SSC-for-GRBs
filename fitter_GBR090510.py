import numpy as np
import emcee as em
import corner
import matplotlib.pyplot as plt
import pandas as pd
import sys
from flux_fit_script import FitterSSC_Class
from flux_fit_script import FluxSSC_Class


Info = {
    'Fit': np.array(['B', 'p', 'theta_obs', 'gammaL']),         # Fitting parameters (Parameter names see P dictionary below)
    'ThetaObsPrior': 'Uniform',                         # Prior for observation angle: Sine or Uniform
}

# bounds for parameters in linear scale
FitBound = {
    'B': np.array([1,10]),
    'p': np.array([1.80,2.21]),
    'theta_obs': np.array([0.011,0.1]),
    'gammaL': np.array([90.,500.])
}

P = {
    'B': 5,
    'p': 1.99,
    'theta_obs': 0.015,
    'gammaL': 200.,
}

# GRB = './sedXRT_090510_sorted.csv'
GRB = './xrt_090510_sorted.csv'
# MCMC fit parameters
SamplerType = "ParallelTempered"
NTemps = 1
NWalkers = 10
Threads = 1

BurnLength = 50
RunLength = 50

Explore = True

# initialize fitter class
MCMC_fit = FitterSSC_Class(Info, FitBound, P, Explore=Explore)
FLUX_gen = FluxSSC_Class()

# LoadData
DF = pd.read_csv(GRB, sep=',')
# Freqs, FreqErrs, Fluxes, FluxErrs = DF['freq'].values, DF['freq_err'].values, DF['nufnu'].values, DF['nufnu_err'].values
Freqs, Fluxes, FluxErrs = DF['freq'].values, DF['nufnu'].values, DF['nufnu_err'].values
MCMC_fit.LoadData(Freqs, Fluxes, FluxErrs)

# initialize sampler
MCMC_fit.GetSampler(SamplerType, NTemps, NWalkers, Threads)

# burning in
BurnInResult = MCMC_fit.BurnIn(BurnLength = BurnLength)
# fitting and store chain results to Result
Result = MCMC_fit.RunSampler(RunLength = RunLength, Output = None)

# Find the best fitting parameters
TheChain = Result['Chain']
LnProbability = Result['LnProbability']
FitDim = len(Info['Fit'])

BestWalker = np.unravel_index(np.nanargmax(LnProbability), LnProbability.shape)
BestParameter = TheChain[BestWalker]
BestLnProbability = LnProbability[BestWalker]
BestLinearParameter = BestParameter

BestP = P.copy()
for i, key in enumerate(Info['Fit']):
    BestP[key] = BestLinearParameter[i]

df = pd.read_csv('bowtie_df.csv')

x_b = df['x_bot'].values
x_t = df['x_top'].values
y_b = df['y_bot'].values/8e1
y_t = df['y_top'].values/8e1

point1 = [np.min(x_t), np.max(y_t)]
point2 = [np.min(x_b), np.max(y_b)]
x_values=[point1[0], point2[0]]
y_values=[point1[1], point2[1]]

point3 = [np.max(x_t), np.min(y_t)]
point4 = [np.max(x_b), np.min(y_b)]
x_values_=[point3[0], point4[0]]
y_values_=[point3[1], point4[1]]

# NewFreqs_syn = np.logspace(8, 24, num=100, endpoint=True)
NewFreqs_syn = np.logspace(11, 27, num=300, endpoint=True)
NewFreqs_syn_chi2 = np.logspace(11, 27, num=14, endpoint=True)
flux_syn = np.asarray(FLUX_gen.syncEmissivity(NewFreqs_syn)[1])
flux_syn_chi2 = np.asarray(FLUX_gen.syncEmissivity(NewFreqs_syn_chi2)[1])
freq_syn_plot = np.asarray(FLUX_gen.syncEmissivity(NewFreqs_syn)[0])

# print('ciao', flux_syn, freq_syn_plot)

NewFreqs_ssc = np.logspace(18, 29, num=10, endpoint=True)
flux_ssc = np.asarray(FLUX_gen.P_ssc_allnumeric(NewFreqs_ssc, BestP)[1])
freq_ssc_plot = np.asarray(FLUX_gen.P_ssc_allnumeric(NewFreqs_ssc, BestP)[0])

print(Fluxes)
print(flux_syn_chi2)
print(FluxErrs)
chi_square = np.sum(((Fluxes - np.power(10,flux_syn_chi2))/FluxErrs)**2)
print(chi_square)

'''
ChiSquare = np.sum(((Fluxes - flux_ssc)/FluxErrs)**2)
DoF = len(Fluxes) - 8 - 1
ChiSquareRed = ChiSquare/DoF
print('degree of freedom', DoF)
print('chi^2 red', ChiSquareRed)
'''

# print('ciao2', flux_ssc, freq_ssc_plot)

plt.plot(np.power(10,freq_syn_plot), np.power(10,flux_syn), c='purple', label='SYN', linewidth=1)
plt.plot(np.power(10,freq_ssc_plot), np.power(10,flux_ssc), c='magenta', label='SSC', linewidth=1)
plt.errorbar(Freqs, Fluxes, yerr=FluxErrs, linestyle='', c='green', fmt='o', ms=3, elinewidth=2, capsize=4, label='SED Swift of GRB090510')
# plt.plot(np.power(10,freq_syn_plot), np.power(10,flux_syn), '--', color='red', linewidth=1.5)
# plt.plot(np.power(10,freq_ssc_plot), np.power(10,flux_ssc), '--', color='red', linewidth=1.5)
# plt.errorbar(Freqs, Fluxes, FluxErrs, FreqErrs, color='blue', linewidth=1.5)
plt.plot(x_b, y_b, '-', c='cornflowerblue', linewidth=1, label='Butterfly Fermi-LAT of GRB090510')
plt.plot(x_t, y_t,'-', c='cornflowerblue', linewidth=1)
plt.plot(x_values, y_values, c='cornflowerblue', linewidth=1)
plt.plot(x_values_, y_values_, c='cornflowerblue', linewidth=1)
plt.ylabel(r'$\nu F_{\nu} [erg/cm^2/s]$', fontsize=15)
plt.xlabel(r'$frequency [Hz]$', fontsize=15)
plt.legend(fontsize=8, loc='upper left')
plt.xlim(1e14, 1e29)
plt.ylim(1e-16, 1e-6)
plt.xscale('log')
plt.yscale('log')
plt.title('SSC model on multi-wavelength data of GRB 090510')
plt.savefig('SSC_sed_GRB090510_prove_fitMCMC_3.png', dpi=500)
plt.title('SSC model on multi-wavelength data of GRB 090510')

Latex = {
    'B': r'$B$',
    'p': r'$p$',
    'theta_obs': r'$\theta_{obs}$',
    'gammaL': r'$\Gamma$',
}

Label = []
for x in Info['Fit']:
    Label.append(Latex[x])


# plot contour with ChainConsumer
Chain = Result['Chain'].reshape((-1, FitDim))
fig = corner.corner(Chain, labels=Label, label_size=20, bins=40, plot_datapoints=False, 
                    quantiles=[0.16,0.5,0.84], show_titles = True, color='darkblue', 
                    label_kwargs={'fontsize': 18},
                    title_kwargs={"fontsize": 18})
fig.savefig("contour_prova_fitMCMC_3.png")