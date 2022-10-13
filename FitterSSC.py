import numpy as np
import pandas as pd
import emcee as em
import sys
# from FluxSSC_Class import FluxSSC_Class
from FluxSSC import P_ssc_allnumeric2


class FitterSSC_Class:
    '''
    Perform MCMC analysis to fit boosted fireball model to observational data.
    '''
    # Private: Fitting Parameter
    _FitDim = 0
    _Info = None
    _FitBound = None
    _P = None

    ### Private: Sampler
    _Sampler = None
    _SamplerType = None
    _Position0 = None
    _Position1 = None

    ### Public: Interpolator and Sampler
    FluxGenerator = None
    # TableInfo = None

    # Public: Observation Data
    Times = None
    TimeBnds = None
    Fluxes = None
    FluxErrs = None
    Freqs = None

    def __init__(self, Info, FitBound, P, Explore=False):
        '''
        Initialize class.
        Args:
            FitParameter (Array): values for fitting parameters
            FitBound (2D Array): lower & upper bounds for fitting parameters: Shape = (len(FitParameter, 2))
            P (dictionary): contains all paramters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB,theta_obs}
            Explore (bool): For Explore run, walkers will randomly distribute in the whole parameter space. 
            This helps to find the maximum posterior. 
            Then, set Explore = False, the walkers will distribute near dictionary P.

        '''
        # Prevent modification of outside dictionary
        import copy
        Info = copy.deepcopy(Info)
        FitBound = copy.deepcopy(FitBound)
        P = copy.deepcopy(P)

        # self.FluxGenerator = FluxSSC_Class()
        self._SetFitParameter(Info, FitBound, P, Explore=Explore)

    # Private Function
    def _SetFitParameter(self, Info, FitBound, P, Explore=False):
        '''
        Args:
            Info (dict): information for MCMC. For example, set fitting parameters.
            FitBound (dict): Bounds for parameters.
            P (dictionary): contains all paramters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB,theta_obs}
        '''

        self._FitDim = len(Info['Fit'])
        self._Info = Info

        # only consider bounds for fitting parameters. Set proper scales.
        temp = []
        for key in Info['Fit']:
            temp.append(FitBound[key])
        self._FitBound = np.array(temp)
        self._FitBoundDict = FitBound

        # Set initial regions for walkers
        if Explore == True:
            self._InitialBound = self._FitBound
        else:
            temp = []
            for key in Info['Fit']:
                if key in Info['Log']:
                    if Info['LogType'] == 'Log10':
                        temp.append(
                            [np.log10(P[key])*0.98, np.log10(P[key])*1.02])
                    else:
                        temp.append([np.log(P[key])*0.98, np.log(P[key])*1.02])
                else:
                    temp.append([P[key]*0.98, P[key]*1.02])
            self._InitialBound = np.array(temp)

        self._P = P.copy()

    # Public Functions
    def LoadData(self, Freqs, Fluxes, FluxErrs):
        '''
        Args:
            Times (Array): observational time in second
            TimeBnbs (Array): observational time bounds; Current MCMC will not use this information
            Fluxes (Array): fluxes in mJy
            FluxErrs (Array): flux errors
            Freqs (Array): frequencies

        Load data to ScaleFit class.
        '''
        self.Freqs = Freqs
        self.Fluxes = Fluxes
        self.FluxErrs = FluxErrs
        

    def GetSampler(self, SamplerType, NTemps, NWalkers, Threads):
        '''
        Set up sampler and position.
        Args:
            SamplerType (str): 'Ensemble' or 'PT'
            NTemps (int): only valid for Parallel-Tempering. set the # of temperature
            NWalkers (int): # of walkers
            Threads (int): # of threads
        '''

        self._SamplerType = SamplerType
        if SamplerType == "Ensemble":
            self._Sampler = em.EnsembleSampler(NWalkers, self._FitDim, LogPosterior, threads=Threads,
                                               args=[self._FitBound, self._Info, self._P, self.Freqs, self.Fluxes, self.FluxErrs])
            # self.FluxGenerator, 
            self._Position0 = self._InitialBound[:, 0] + (
                self._InitialBound[:, 1]-self._InitialBound[:, 0])*np.random.rand(NWalkers, self._FitDim)
        else:
            self._Sampler = em.PTSampler(NTemps, NWalkers, self._FitDim, LogLike, LogPrior, threads=Threads,
                                         loglargs=[self._Info, self._P, self.Freqs, self.Fluxes, self.FluxErrs],
                                         logpargs=[self._FitBound, self._Info]
                                         )
            # self.FluxGenerator,
            self._Position0 = self._InitialBound[:, 0] + (
                self._InitialBound[:, 1]-self._InitialBound[:, 0])*np.random.rand(NTemps, NWalkers, self._FitDim)

    def BurnIn(self, BurnLength=10, Output=None):
        '''
        Args:
            BurnLength (int): burning length
            Output (str/None): directory to save burning results
        Return:
            dict: results: chain, log probability and acceptance fraction.
        '''

        from pickle import dump, HIGHEST_PROTOCOL
        from time import time

        # Run sampler
        Start = time()
        i = 0
        for StepResult in self._Sampler.sample(self._Position0, iterations=BurnLength, storechain=True):
            i += 1
            DeltaTime = time() - Start
            Label = "%02d m %02d s" % (DeltaTime/60, DeltaTime % 60)
            sys.stdout.write('\r Burning ... %.1f%% Time=%s' %
                             ((100.0*i)/(BurnLength), Label))
            sys.stdout.flush()
        sys.stdout.write('\n')
        # Save postion to self._Position1, which is the starting position for later run.
        self._Position1 = StepResult[0]
        # Save Burn in results
        BurnInResult = {}
        if self._SamplerType is 'Ensemble':
            BurnInResult['Chain'] = self._Sampler.chain
            BurnInResult['LnProbability'] = self._Sampler.lnprobability
            BurnInResult['AcceptanceFraction'] = self._Sampler.acceptance_fraction
        else:
            BurnInResult['Chain'] = self._Sampler.chain[0]
            BurnInResult['LnProbability'] = self._Sampler.lnprobability[0]
            BurnInResult['AcceptanceFraction'] = self._Sampler.acceptance_fraction[0]

        if Output is not None:
            with open(Output, 'wb') as handle:
                dump(BurnInResult, handle, protocol=HIGHEST_PROTOCOL)

        return BurnInResult

    def RunSampler(self, RunLength=500, Output=None):
        '''
        Args:
            RunLength (int): running length
            Output (str/None): directory to save burning results
        Return:
            dict: results: chain, log probability and acceptance fraction.
        '''

        from pickle import dump, HIGHEST_PROTOCOL
        from time import time

        # Run sampler
        self._Sampler.reset()
        Start = time()
        i = 0
        for StepResult in self._Sampler.sample(self._Position1, iterations=RunLength, storechain=True):
            i += 1
            DeltaTime = time() - Start
            Label = "%02d m %02d s" % (DeltaTime/60, DeltaTime % 60)
            sys.stdout.write('\r Running ... %.1f%% Time=%s' %
                             ((100.0*i)/(RunLength), Label))
            sys.stdout.flush()
        sys.stdout.write('\n')
        # Save results
        Result = {}
        if self._SamplerType is 'Ensemble':
            Result['Chain'] = self._Sampler.chain
            Result['LnProbability'] = self._Sampler.lnprobability
            Result['AcceptanceFraction'] = self._Sampler.acceptance_fraction
        else:
            Result['Chain'] = self._Sampler.chain[0]
            Result['LnProbability'] = self._Sampler.lnprobability[0]
            Result['AcceptanceFraction'] = self._Sampler.acceptance_fraction[0]

        if Output is not None:
            OutputData = {
                "Result": Result,
                "Info": self._Info,
                "FitBound": self._FitBoundDict,
                "P": self._P,
                "TableInfo": self.FluxGenerator.TableInfo
            }
            with open(Output, 'wb') as handle:
                dump(OutputData, handle, protocol=HIGHEST_PROTOCOL)
        return Result

'''
We need to explicitly define the prior, likelyhood and posterior.
To run emcee in parallel, the definitions for prior and likelyhood are tricky. Please check the emcee document: http://dfm.io/emcee/current/user/advanced/#multiprocessing
'''

def LogPrior(FitParameter, FitBound, Info):
    '''
    This function is for PTSampler
    Args:
        FitParameter (Array): values for fitting parameters
        FitBound (2D Array): lower & upper bounds for fitting parameters: Shape = (len(FitParameter, 2))
        Info (dict): prior information
    Return:
        float: if within bounds, log(prior); else, -inf;
    '''

    if((FitParameter[:] > FitBound[:, 0]) * (FitParameter[:] < FitBound[:, 1])).all():
        if 'theta_obs' in Info['Fit']:
            if Info['ThetaObsPrior'] == 'Sine':
                i = np.argwhere(Info['Fit'] == 'theta_obs')[0][0]
                return np.log(np.sin(FitParameter[i]))
            elif Info['ThetaObsPrior'] == 'Uniform':
                return 0.0
            else:
                raise ValueError("Cannot recognize Info['ThetaObsPrior'].")
        else:
            return 0.0
    else:
        return -np.inf


def LogLike(FitParameter, Info, P, Freqs, Fluxes, FluxErrs):
    # FluxSSC_Class, 
    '''
    This function is for PTSampler
    Args:
        FitParameter (Array): values for fitting parameters
        Info (dict): prior information
        P (dictionary): contains all paramters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB,theta_obs}
        FluxSSC_Class (object): flux generator
    Return:
        float: caculate Chi^2 and return -0.5*Chi^2 (details see Ryan+ 2014)

    '''

    for i, key in enumerate(Info['Fit']):
        P[key] = FitParameter[i]
    # FluxesModel = FluxSSC_Class.P_ssc_allnumeric(Freqs, P)[1]
    FluxesModel = P_ssc_allnumeric2(Freqs, P)[1]
    if np.isnan(FluxesModel[0]):
        return -np.inf
    else:
        print(P)
        print(len(Fluxes), len(FluxesModel))
        ChiSquare = np.sum(((Fluxes - FluxesModel)/FluxErrs)**2)
        Like = -0.5*ChiSquare
        return Like


def LogPosterior(FitParameter, FitBound, Info, P, Freqs, Fluxes, FluxErrs):
    # FluxSSC_Class, 
    '''
    This function is for EnsembleSampler.
    Args:
        FitParameter (Array): values for fitting parameters
        FitBound (2D Array): lower & upper bounds for fitting parameters: Shape = (len(FitParameter, 2))
        Info (dict): prior information
        P (dictionary): contains all paramters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB,theta_obs}
        FluxGenerator (object): flux generator
        Times (Array): observational time in second
        Freqs (Array): frequencies. The length should be the same as Times
        Fluxes (Array): flux in mJy
        FluxErrs (Array): flux error in mJy
    Return:
        float: log(prior) + log(likelyhood)
    '''

    # Log Prior
    LogPriorFunction = LogPrior(FitParameter, FitBound, Info)

    # Log Likelyhood
    LogLikeFunction = LogLike(FitParameter, Info, P, Freqs, Fluxes, FluxErrs)
    # FluxSSC_Class,

    # Log Posterior
    if np.isfinite(LogPriorFunction) and np.isfinite(LogLikeFunction):
        LogPosterior = LogPriorFunction + LogLikeFunction
        return LogPosterior
    else:
        return -np.inf
