# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:36:43 2024

@author: TharpBM
"""
import time
import pandas as pd
import numpy as np
import general as GenFun
import flexible as Flex

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

def sample_wr_df():
    """
    Create a pandas DataFrame of a sample input csv file necessary
    to properly import wire run data.
    """
    _df = pd.DataFrame({
        'PoleID': ['Pole1', 'Pole2', 'Pole3'],
        'STA': [162297, 162452, 162612],
        'Rail EL': [0, 0, 0],
        'MW Height': [24, 23.5, 22.5],
        'CW Height': [20, 19.25, 18.5],
        'PreSag': [0.19375, 0.2, 0.16625],
        'Deviation Angle': [0, 4, -5]
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
        #self._sl = (
        #    df_loadlist['CW Loading'], df_loadlist['Stationing'],
        #    df_loadlist['MW Loading'], df_loadlist['Stationing']
        #    )
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

    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        if not self._solved:
            self._solve()
        return self._sag

    def dataframe_ha(self):
        """ return a pandas dataFrame of the hangers """
        if not self._solved:
            self._solve()
        _cwel = self._catenarysag.get('LoadedSag')[
            np.in1d(self._catenarysag.get('Stationing'),
                    self._catenarysag.get('HA_STA'))]
        _mwel = self._catenarysag.get('LoadedSag_MW')[
            np.in1d(self._catenarysag.get('Stationing'),
                    self._catenarysag.get('HA_STA'))]
        _hal = _mwel - _cwel
        _df_ha = pd.DataFrame({
            'Stationing': self._catenarysag.get('HA_STA'),
            'CW Elevation': _cwel,
            'MW Elevation': _mwel,
            'HA Length': _hal,
            'HA Load': self._catenarysag.get('SupportLoad_CW')
            })
        return _df_ha

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

    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        return self._diff
