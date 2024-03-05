# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:36:43 2024

@author: TharpBM
"""
import time
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.insert(0, '..\\')
from library import general as GenFun
from library.catenary import flexible as Flex

def conductor_particulars(mw, cw, ha):
    """
    Create a pandas DataFrame of the conductor details for use in running
    detailed span calculations. Weights and tensions should be input
    utilizing the major unit, ie lbf/ft for weight and lbf for tension.
    
    At this time, hanger weight is added to MW height as a per linear span
    unit basis. It should be input as an average hanger weight per linear span.
    
    No unit conversion are made in this calculation set.
    
    Example input statement
    LC1 = _system.conductor_particulars(
        (1.544, 5000), (1.063, 3000), 0.2)

    Parameters
    ----------
    mw : list
        MW Weight and Tension, formated in a list. ex. (1.544, 5000)
    cw : list
        CW Weight and Tension, formated in a list. ex. (1.063, 3000)
    ha : list
        Average Hanger Span Weight, ex. 0.2

    Returns
    -------
    pandas DataFrame
    """
    _df = pd.DataFrame({
        'Type': ['Messenger Wire', 'Contact Wire', 'Hangers'],
        'Weight': [mw[0], cw[0], ha],
        'Tension': [mw[1], cw[1], 0]
         })
    return _df

def wire_run(path, first=None, last=None):
    """
    Create a pandas DataFrame of the wire run details for use in running
    detailed span calculations.
    
    Example contents of a properly formatted csv file
    
    |  PoleID,STA,Rail EL,MW Height,CW Height,PreSag,Deviation Angle
    |  1,162297,0,24,20,-1.1625,0
    |  2,162452,0,23.5,19.25,-1.2,0
    |  3,162612,0,23.5,18.5,-0.9975,0
    |  4,162745,0,22.5,18.5,-1.125,0
    |  5,162895,0,22.5,18.5,-1.2975,0
    |  6,163068,0,22.5,18.5,-1.425,0

    Please take note of the example units!
    No unit conversion are made in this calculation set.
    PoleID (string), STA (feet), Rail EL (feet), MW Height (feet),
    CW Height (feet), PreSag (inches), Deviation Angle (degrees)
    
    Parameters
    ----------
    path : string
        Full file path to a .csv file containing the wire run details.
    first : int, optional
        Index of row to start, leave out to go from start.
    last : int, optional
        Index of row to end, leave out to go to end.
        
    * Index is zero based. The first data row is 0. 
    * The column header row is not counted
    
    Returns
    -------
    _wr : TYPE
        DESCRIPTION.

    """
    _df = pd.read_csv(path)
    _df = _df.truncate(before=first, after=last, copy=False)
    _df.reset_index(drop=True, inplace=True)
    _wr = GenFun.WireRun_df(_df,1)
    return _wr

def combine_plots(names, dfargs, col=None, row=None, xscale = 1, yscale = 1):
    """
    Combine plots of wire sag for comparison

    Parameters
    ----------
    names : string
        'Type' description for plots associated with the dataFrame.
    dfargs : pandas DataFrame
        Output dataFram.
    col : TYPE, optional
        DESCRIPTION. The default is None.
    row : TYPE, optional
        DESCRIPTION. The default is None.
    xscale : TYPE, optional
        DESCRIPTION. The default is 1.
    yscale : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    plot : Seaborn 
        Seaborn plot of combined wire charts.

    """
    """
    combine plots of wire sag for comparison

    Parameters
    ----------
    names : string
        'Type' description for plots associated with the dataFrame
    dfargs : pandas dataFrame
        Output dataFrame

    Returns
    -------
    Seaborn plot of combined wire charts

    """
    for _n, _df in zip(names, dfargs):
        _df.type = _n
        _df.Elevation *= yscale
        _df.Stationing *= xscale
    _cdf = pd.concat(dfargs)
    plot = sns.relplot(data=_cdf, x='Stationing', y='Elevation', kind='line',
                       hue='type', size='cable', sizes=(1,1),
                       col=col, row=row) #, margin_titles=True)
    return plot

def combine_plots_load(names, dfargs, col=None, row=None):
    """
    combine plots of loading for comparison

    Parameters
    ----------
    names : string
        Type description for plots associated with the dataFrame
    dfargs : pandas dataFrame
        Output dataFrame

    Returns
    -------
    Seaborn plot of combined wire charts

    """
    for _n, _df in zip(names, dfargs):
        _df.type = _n
    _cdf = pd.concat(dfargs)
    plot = sns.relplot(data=_cdf, x='Stationing', y='Load', kind='line',
                       hue='type',
                       col=col, row=row) #, margin_titles=True)
    return plot

class CatenaryFlexible():
    """ A simple container for the catenary system containing all design data
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, conductorparticulars, dataframe):
        self._cp = conductorparticulars
        self._wr = dataframe
        #[MaxHASpacing, minCW_P, MinHALength, HA_Accuracy]
        self._hd = np.array([28, 0, 3/12, 1/50/12])
        #[xStep, xRound, xMultiplier, yMultiplier, SteadyArmLength]
        self._bd = np.array([1, 1, 1, 1, 3])
        #[P_DiscreteLoad_CW, STA_DiscreteLoad_CW,
        # P_DiscreteLoad_MW, STA_DiscreteLoad_MW]
        self._sl = (
           (0, 0, 0), (0, 0, 0),
            (0, 0, 0), (0, 0, 0)
            )
        self._solved = False
        self._catenarysag = None
        self._sag = None

    def _solve(self):
        """ Solve catenary system and cache results"""
        self._catenarysag = Flex.CatenarySag_Flexible(
            self._wr, self._cp['Weight'], self._cp['Tension'],
            self._sl, self._hd, self._bd)
        # Nominal dictionary keys
        # Stationing, HA_STA, HA_EL
        # LoadedSag, SupportLoad_CW, P_SpanWeight
        # LoadedSag_MW, SupportLoad_MW, P_SpanWeight_MW
        _df_cw = pd.DataFrame({
            'Stationing': self._catenarysag.get('Stationing'),
            'Elevation': self._catenarysag.get('LoadedSag'),
            'type': 'CW',
            'cable': 'CW'
            })
        _df_mw = pd.DataFrame({
            'Stationing': self._catenarysag.get('Stationing'),
            'Elevation': self._catenarysag.get('LoadedSag_MW'),
            'type': 'MW',
            'cable': 'MW'
            })
        self._sag = pd.concat([_df_mw, _df_cw], ignore_index=False)
        self._solved = True

    def resetloads(self, loadlist):
        """ update the load list for the alternate condition """
        self._solved = False
        self._sl = (
            loadlist[2], loadlist[1],
            loadlist[3], loadlist[1]
            )

    def getcatenarysag(self):
        """ return the cache solution dictionary """
        return self._catenarysag

    def getcp(self):
        """ return condutor particulars """
        return self._cp

    def getwr(self):
        """ return the wire run """
        return self._wr

    def gethd(self):
        """ return the hanger design """
        return self._hd

    def getbd(self):
        """ return the base design """
        return self._bd

    def savetocsv(self, path):
        """ Output wire data to CSV """
        if not self._solved:
            self._solve()
        self._sag.to_csv(path)

    def plot(self):
        """ return a seaborn plot of the wire """
        """ default y exaggeration  is 5:1"""
        if not self._solved:
            self._solve()
        __sta = self._catenarysag.get('Stationing')
        _xtotal = (__sta[-1] - __sta[0])
        __elMax = max(
            max(self._catenarysag.get('LoadedSag_MW')),
            max(self._catenarysag.get('LoadedSag'))
            ) 
        __elMin = min(
            min(self._catenarysag.get('LoadedSag_MW')),
            min(self._catenarysag.get('LoadedSag'))
            )
        _ytotal = 5*(__elMax-__elMin)
        print('x', _xtotal, 'y', _ytotal)
        pl = sns.relplot(
            data=self._sag, x='Stationing', y='Elevation',
            kind='line', hue='type', height=10, aspect=_xtotal/_ytotal/10)
        return pl 

    def plot_spt(self):
        """ return a seaborn plot of the wire """
        if not self._solved:
            self._solve()
        _sptL_cw = pd.DataFrame({
            'Stationing': self._catenarysag.get('HA_STA'),
            'Load': self._catenarysag.get('SupportLoad_CW'),
            'type': 'HA',
            'cable': 'HA'
            })
        _sptL_mw = pd.DataFrame({
            'Stationing': self._wr.STA,
            'Load': self._catenarysag.get('SupportLoad_MW'),
            'type': 'MW',
            'cable': 'MW'
            })
        _sptL_df = pd.concat([_sptL_mw, _sptL_cw], ignore_index=False)
        pl = sns.relplot(
            data=_sptL_df, x='Stationing', y='Load',
            kind='scatter', hue='type')
        #pl.refline(y=self._hd[1])
        return pl

    def plot_halength(self):
        """ return a seaborn plot of the wire """
        if not self._solved:
            self._solve()
        __sta = self._catenarysag.get('Stationing')
        __hasta = self._catenarysag.get('HA_STA')
        __cwel = self._catenarysag.get('LoadedSag')
        __mwel = self._catenarysag.get('LoadedSag_MW')
        _haL = __mwel[np.in1d(__sta, __hasta)] - __cwel[np.in1d(__sta, __hasta)]
        _haL_df = pd.DataFrame({
            'Stationing': __hasta,
            'Length': _haL*12,
            'type': 'HA Length',
            'cable': 'HA Length'
            })
        pl = sns.relplot(
            data=_haL_df, x='Stationing', y='Length',
            kind='scatter')
        pl.set_axis_labels(y_var='HA Length (in)')
        pl.refline(y=self._hd[2]*12)
        return pl

    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        if not self._solved:
            self._solve()
        return self._sag

class AltCondition():
    """ container for calculating changes made to an already calculated
    CatenaryFlexible. Changes could be due to weight, tension, or point loads
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, altconductor, basedesign):
        self._ref = basedesign
        self._cp = None
        self._ld = np.array([0,0])
        self._solved = False
        self._catenarysag = None
        self._sag = None
        self._loops = 0
        self._sr = None
        self._srdf = None
        self._cwdiffdf = None
        self.updateconductor(altconductor)

    def resetloadlist(self, loadlist):
        """ update the load list for the alternate condition """
        self._solved = False
        self._ld = loadlist

    def updateconductor(self, altconductor):
        """ update conductor particulars
        weight into iterative loop should be difference only
        tension should be full
        """
        self._solved = False
        _alt = altconductor.copy()
        _weightdiff = _alt['Weight'] - self._ref.getcp()['Weight']
        _alt.Weight = _weightdiff
        self._cp = _alt

    def _solve(self):
        """ Solve alternate condition and cache results"""
        self._catenarysag, self._loops, self._sr = Flex.CatenarySag_FlexibleHA_Iterative(
            self._ref.getwr(), self._cp['Weight'], self._cp['Tension'],
            self._ref.gethd(), self._ref.getbd(), self._ld, self._ref.getcatenarysag())
        # Uplift dictionary keys
        # Stationing, HA_STA, HA_EL
        # LoadedSag, SupportLoad_CW, P_SpanWeight
        # LoadedSag_MW, SupportLoad_MW, P_SpanWeight_MW
        # Wire Sag DataFrame
        _df_cw = pd.DataFrame({
            'Stationing': self._catenarysag.get('Stationing'),
            'Elevation': self._catenarysag.get('LoadedSag'),
            'type': 'CW',
            'cable': 'CW'
            })
        _df_mw = pd.DataFrame({
            'Stationing': self._catenarysag.get('Stationing'),
            'Elevation': self._catenarysag.get('LoadedSag_MW'),
            'type': 'MW',
            'cable': 'MW'
            })
        self._sag = pd.concat([_df_mw, _df_cw], ignore_index=False)
        # CW Support Resistance DataFrame
        self._srdf = pd.DataFrame({
            'Stationing': self._ref.getwr()['STA'],
            'Load': self._sr,
            'type': 'CW Uplift Resistance',
            'cable': 'CW Uplift Resistance'
            })
        # CW EL Difference DataFrame
        self._cwdiffdf = pd.DataFrame({
            'Stationing': self._catenarysag.get('Stationing'),
            'Elevation': self._catenarysag.get('LoadedSag') - self._ref.getcatenarysag().get('LoadedSag'),
            'type': 'CW',
            'cable': 'CW'
            })
        self._solved = True

    def getloops(self):
        """ return the loops """
        if not self._solved:
            self._solve()
        return self._loops

    def savetocsv_alt(self, path):
        """ Output wire data to CSV """
        if not self._solved:
            self._solve()
        self._sag.to_csv(path)

    def plot_alt(self):
        """ return a seaborn plot of the wire """
        if not self._solved:
            self._solve()
        return sns.relplot(
            data=self._sag, x='Stationing', y='Elevation',
            kind='line', hue='type')
    
    def plot_sr(self):
        """ Output reaction resistance of steady arms due to radial load """
        if not self._solved:
            self._solve()
        return sns.relplot(
            data=self._srdf, x='Stationing', y='Load',
            kind='line')

    def plot_cwdiff(self):
        """ Output CW EL difference """
        if not self._solved:
            self._solve()
        return sns.relplot(
            data=self._cwdiffdf, x='Stationing', y='Elevation',
            kind='line')
    
    def compareplot(self):
        """ return a seaborn plot of the wire overlayed with the
        original design data
        """
        if not self._solved:
            self._solve()
        _plot = combine_plots(
            ('Original', 'Alternate'),
            (self._ref.dataframe(), self.dataframe())
            )
        return _plot

    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        if not self._solved:
            self._solve()
        return self._sag
    
    def dataframe_cwdiff(self):
        """ return the cw elevation difference from original, + is higher """
        if not self._solved:
            self._solve()
        return self._cwdiffdf
    
    def dataframe_sr(self):
        """ return a pandas dataFrame of the CW Support Resistance """
        if not self._solved:
            self._solve()
        return self._srdf

class Elasticity():
    """ container for calculating span elasticity from an already calculated
    CatenaryFlexible """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, altconductor, basedesign, upliftforce, stepsize, startspt=0, endspt=None):
        self._ref = basedesign
        self._cp = altconductor
        self._diff = None
        self._cycle(upliftforce, stepsize, startspt, endspt)

    def _cycle(self, upliftforce, stepsize, startspt, endspt):
        """ Solve elasticitiy graph and cache results"""
        _st = time.time()
        _elastic = AltCondition(self._cp, self._ref)
        if endspt is None:
            endspt = len(self._ref.getwr()['STA'])-1
        _startsta = self._ref.getwr()['STA'].iloc[startspt]
        _endsta = self._ref.getwr()['STA'].iloc[endspt]
        if startspt == 0:
            _startsta += stepsize
        if endspt == len(self._ref.getwr()['STA'])-1:
            _endsta -= stepsize
        _staval = np.arange(_startsta, _endsta, stepsize, dtype=float)
        _staval = GenFun.RoundVal(_staval, 1)
        print('Checking elasticity from STA', _startsta, 'to STA', _endsta)
        _avgcycletime = 0.75
        print(' | ', int(len(_staval)), 'total cycles estimating to take', 
              round(_avgcycletime * int(len(_staval)),4), 'seconds')
        _diffmin = np.copy(_staval*0)
        _diffmax = np.copy(_staval*0)
        _cycleloops = np.copy(_staval*0)
        _cycletime = np.copy(_staval*0)
        i = 0
        for _sta in _staval:
            _cst = time.time()
            #print(i, ' | calculating uplift at STA =', _sta)
            _elastic.resetloadlist((-upliftforce, _sta))
            _eldiff = _elastic.dataframe_cwdiff().Elevation
            _diffmin[i] = np.nanmin(_eldiff)
            _diffmax[i] = np.nanmax(_eldiff)
            _cycleloops[i] = _elastic.getloops()
            _cet = time.time()
            _cycletime[i] = _cet - _cst
            i += 1
        _df_min = pd.DataFrame({
            'Stationing': _staval,
            'Rise (in)': _diffmin*12,
            'type': 'DiffMIN',
            'cable': 'DiffMIN'
            })
        _df_max = pd.DataFrame({
            'Stationing': _staval,
            'Rise (in)': _diffmax*12,
            'type': 'DiffMAX',
            'cable': 'DiffMAX'
            })
        self._diff = pd.concat([_df_min, _df_max], ignore_index=False)
        _et = time.time()
        _totaltime = _et - _st
        _meantime = np.mean(_cycletime)
        print(i, ' total loops processed in', round(_totaltime,4), 'seconds',
              'with average cycle time of', round(_meantime*1000,4), 'milliseconds')

    def savetocsv(self, path):
        """ Output wire data to CSV """
        self._diff.to_csv(path)

    def plot(self):
        """ return a seaborn plot of the wire """
        return sns.relplot(
            data=self._diff, x='Stationing', y='Rise (in)',
            kind='line', hue='type')

    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        return self._diff
