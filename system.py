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

def sample_df_cp():
    _df = pd.DataFrame({
        'Cable': ['MW', 'CW', 'HA'],
        'Weight': [1.544, 1.063, 0.2],
        'Tension': [5000, 3300, 0]
    })
    return _df

def sample_df_acp():
    _df = pd.DataFrame({
        'Load Condition': ['LC1', 'LC2'],
        'MW Weight': [1.544, 1.063],
        'MW Tension': [5129, 4187],
        'CW Weight': [1.063, 1.063],
        'CW Tension': [3385, 2482],
        'HA Weight': [0.2, 0.2]
    })
    return _df

def sample_df_hd():
    #[MaxHASpacing, minCW_P, MinHALength, HA_Accuracy]
    #_hd = np.array([28, 0, 3/12, 1/50/12])
    # test _hd[0] == 28
    _df = pd.DataFrame({
        'Variable Description': ['Max HA Spacing',
                                'Min HA Load',
                                'Min HA Length',
                                'HA Accuracy'],
        'Value': [28, 0, 3/12, 1/50/12]
    })
    # test _df.Values[0] == 28
    return _df

def sample_df_bd():
    #[xStep, xRound, xMultiplier, yMultiplier, SteadyArmLength]
    #_bd = np.array([1, 1, 1, 1, 3])
    _df = pd.DataFrame({
        'Variable Description': ['xStep',
                                'xRound',
                                'xMultiplier',
                                'yMultiplier',
                                'Steady Arm Length'],
        'Value': [1, 1, 1, 1, 3]
    })
    return _df

def sample_df_sl():
    #[P_DiscreteLoad_CW, STA_DiscreteLoad_CW,
    # P_DiscreteLoad_MW, STA_DiscreteLoad_MW]
    #_sl = (
    #    (0, 0, 0), (0, 0, 0),
    #    (0, 0, 0), (0, 0, 0)
    #    )
    _df = pd.DataFrame({
        'Description': ['a', 'b', 'c', 'd', 'e'],
        'Stationing': [162472, 162427, 164730, 164780, 166364],
        'CW Loading': [15, 5, -15, 15, 30],
        'MW Loading': [15, 5, 5, 0, 85]
    })
    return _df

def sample_df_dd():
    """
    Create a properly formatted string containing design data.
    This can be written to csv as a sample to properly import design data.
    """
    _df = GenFun.df_pad(sample_df_cp(),'Conductor Particulars')
    _df = GenFun.df_add(_df, GenFun.df_pad(sample_df_acp(),'Alternate Conductor Particulars'))
    _df = GenFun.df_add(_df, GenFun.df_pad(sample_df_hd(),'Hanger Design Data'))
    _df = GenFun.df_add(_df, GenFun.df_pad(sample_df_bd(),'Calculation Constants'))
    _df = GenFun.df_add(_df, GenFun.df_pad(sample_df_sl(),'Span Loading'))
    _df.reset_index(drop=True, inplace=True)
    return _df

def add_base_acd(_cd, _acd):
    _df = _acd.copy()
    _df.loc[-1] = [
        'BASE',
        _cd.loc[0,'Weight'], _cd.loc[0,'Tension'],
        _cd.loc[1,'Weight'], _cd.loc[1,'Tension'],
        _cd.loc[2,'Weight']
    ]
    _df = _df.sort_index().reset_index(drop=True)
    return _df

def split_acd(_acd, row):
    _df = pd.DataFrame({
        'Cable': ['MW', 'CW', 'HA'],
        'Weight': [_acd.loc[row,'MW Weight'], _acd.loc[row,'CW Weight'], _acd.loc[row,'HA Weight']],
        'Tension': [_acd.loc[row,'MW Tension'], _acd.loc[row,'CW Tension'], 0]
        })
    return _df
    
def split_df(_df, first=None, last=None):
    _dfc = _df.copy()
    _dfc = _dfc.truncate(before=first, after=last, copy=False)
    #_dfc = _dfc.iloc[start:end,:]
    _dfc = _dfc.dropna(axis=1, how='all')
    _dfc.columns = _dfc.iloc[0]
    _dfc = _dfc[1:]
    _dfc.reset_index(drop=True, inplace=True)
    _dfc.iloc[:,1:] = _dfc.iloc[:,1:].astype(float)
    return _dfc

def create_df_dd(_df):
    _dd = _df.copy()
    icp = int(np.where(_dd == 'Conductor Particulars')[0])
    iacp = int(np.where(_dd == 'Alternate Conductor Particulars')[0])
    ihd = int(np.where(_dd == 'Hanger Design Data')[0])
    ibd = int(np.where(_dd == 'Calculation Constants')[0])
    isl = int(np.where(_dd == 'Span Loading')[0])
    _df_cd = split_df(_dd, icp+1, iacp-2)
    _df_acd = split_df(_dd, iacp+1, ihd-2)
    _df_acd = add_base_acd(_df_cd, _df_acd)
    _df_hd = split_df(_dd, ihd+1, ibd-2)
    _df_bd = split_df(_dd, ibd+1, isl-2)
    _df_sl = split_df(_dd, isl+1)
    return _df_cd, _df_acd, _df_hd, _df_bd, _df_sl  

def sample_df_wr():
    """
    Create a properly formatted pandas DataFrame containing wire run data.
    This can be written to csv as a sample to properly import wire run data.
    """
    _df = pd.DataFrame({
        'PoleID': ['Pole1', 'Pole2', 'Pole3', 'Pole4'],
        'STA': [162297, 162452, 162612, 162775],
        'Rail EL': [0, 0, 0, 0],
        'MW Height': [24, 23.5, 22.5, 24],
        'CW Height': [20, 19.25, 18.5, 18.5],
        'PreSag': [0.19375, 0.2, 0.16625, 0],
        'Deviation Angle': [0, 4, -5, 0]
    })
    return _df

def wire_run(path, first=None, last=None):
    """
    Create a pandas DataFrame of the wire run details for use in running
    detailed span calculations.
    """
    _df = pd.read_csv(path)
    _df = _df.truncate(before=first, after=last, copy=False)
    _df.reset_index(drop=True, inplace=True)
    _df.iloc[:,1:] = _df.iloc[:,1:].astype(float)
    return _df

def df_ft(_df):
    df = _df.copy()
    df['Stationing'] = (df['Stationing'].astype(str) + "'")
    df['Elevation'] = (df['Elevation'].astype(str) + "'")
    return df
    
def sag_scr(_df) -> str:
    val = '_pline\n'
    val += _df.to_csv(columns=['Stationing', 'Elevation'], index=False, header=False, encoding='UTF-8')
    val += '\n'
    return val

def ha_scr(_df) -> str:
    df = _df.copy()
    df = df.assign(hastr=[
        '_line ' + 
        str(x) + "'," + str(y) + "' " +
        str(x) + "'," + str(z) + "' "
        for x, y, z in
        zip(df['Stationing'], df['CW Elevation'], df['MW Elevation'])])
    val = df.to_csv(columns=['hastr'], index=False, header=False, encoding='UTF-8')
    val = val.replace('"','')
    return val

def _pad_scr(txt) -> str:
    val = '(setq oldsnap (getvar "osmode"))\n'
    val += '(setvar "osmode" 0)\n'
    val += txt
    val += '(setvar "osmode" oldsnap)\n'
    return val
    
def SagtoCAD(ref, yscale=1) -> str:
    _df = ref.dataframe()
    _df['Elevation'] = _df['Elevation'] * yscale
    df = df_ft(_df)
    df_mw = df[df.cable == 'MW']
    txt = sag_scr(df_mw)
    df_cw = df[df.cable == 'CW']
    txt += sag_scr(df_cw)
    _df = ref.dataframe_ha()
    _df['CW Elevation'] = _df['CW Elevation'] * yscale
    _df['MW Elevation'] = _df['MW Elevation'] * yscale
    txt += ha_scr(_df)
    val = _pad_scr(txt)
    return val
    
class CatenaryFlexible():
    """ A simple container for the catenary system containing all design data
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, pandasdesigndetails, dataframe):
        self._cp, self._acp, self._hd, self._bd, self._sl = create_df_dd(pandasdesigndetails)
        #self._cp = conductorparticulars
        #self._hd = sample_df_hd()
        #self._bd = sample_df_bd()
        self._wr = GenFun.WireRun_df(dataframe, self._bd.iloc[1,1])
        #[P_DiscreteLoad_CW, STA_DiscreteLoad_CW,
        # P_DiscreteLoad_MW, STA_DiscreteLoad_MW]
        self._solved = False
        self._catenarysag = None
        self._sag = None
        self._sag_w_ha = None

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
        _mw_ha_sta, _mw_ha_el = GenFun.LoadedSagMW_wHanger(
            self._catenarysag.get('Stationing'), 
            self._catenarysag.get('LoadedSag'), 
            self._catenarysag.get('LoadedSag_MW'), 
            self._catenarysag.get('HA_STA')
        )
        _df_mw_w_ha = pd.DataFrame({
            'Stationing': _mw_ha_sta,
            'Elevation': _mw_ha_el,
            'type': 'MW',
            'cable': 'MW'
            })
        self._sag = pd.concat([_df_mw, _df_cw], ignore_index=False)
        self._sag_w_ha = pd.concat([_df_mw_w_ha, _df_cw], ignore_index=False)
        self._solved = True

    def resetloads(self, loadlist):
        """ update the load list for the alternate condition """
        self._solved = False
        #self._sl = (
        #    df_loadlist['CW Loading'], df_loadlist['Stationing'],
        #    df_loadlist['MW Loading'], df_loadlist['Stationing']
        #    )
        #df_loading = (
        #    loadlist['Description'].values,
        #    loadlist['Stationing'].values,
        #    loadlist['CW Loading'].values,
        #    loadlist['MW Loading'].values
        #    )
        #self._sl = (
        #    df_loading[2], df_loading[1],
        ##    df_loading[3], df_loading[1]
        #   )
        self._sl = loadlist

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
        basedesign._solve()
        self._ref = basedesign
        self._cp = None
        self._ld = np.array([0,0])
        self._solved = False
        self._catenarysag = None
        self._sag = None
        self._loops = 0
        self._sr = None
        self._srdf = None
        self._df_cw = None
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
        self._df_cw = _df_cw
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

    def dataframe_cw(self):
        """ return a pandas dataFrame of the wire """
        if not self._solved:
            self._solve()
        return self._df_cw

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

class AltCondition_Series():
    def __init__(self, altconductor, basedesign):
        basedesign._solve()
        self._data = altconductor
        self._bd = basedesign
        self._df = None
        self._df_cw = None
        self._df_cwdiff = None
        self._df_sr = None
        self._solved = False
    
    def _solve(self):
        _df = pd.DataFrame()
        _df_cw = pd.DataFrame()
        _df_cwdiff = pd.DataFrame()
        _df_sr = pd.DataFrame()
        for index, row in self._data.iterrows():
            if np.isfinite(row['MW Weight']):
                _acd = split_acd(self._data,index)
                ALT = AltCondition(_acd,self._bd)
                _df_tmp = ALT.dataframe()
                _df_cw_tmp = ALT.dataframe_cw()
                _df_cwdiff_tmp = ALT.dataframe_cwdiff()
                _df_sr_tmp = ALT.dataframe_sr()
                #_df_tmp.type += '_' + row['Load Condition']
                _df_tmp.type = row['Load Condition']
                _df_cwdiff_tmp.type = row['Load Condition']
                _df_sr_tmp.type = row['Load Condition']
                _df = pd.concat([_df, _df_tmp], ignore_index=False)
                _df_cw = pd.concat([_df_cw, _df_cw_tmp], ignore_index=False)
                _df_cwdiff = pd.concat([_df_cwdiff, _df_cwdiff_tmp], ignore_index=False)
                _df_sr = pd.concat([_df_sr, _df_sr_tmp], ignore_index=False)
        self._df = _df
        self._df_cw = _df_cw
        self._df_cwdiff = _df_cwdiff
        self._df_sr = _df_sr
        self._solved = True

    def dataframe(self):
        if not self._solved:
            self._solve()
        return self._df

    def dataframe_cwd(self):
        if not self._solved:
            self._solve()
        return self._df_cw
    
    def dataframe_cwdiff(self):
        if not self._solved:
            self._solve()
        return self._df_cwdiff

    def dataframe_sr(self):
        if not self._solved:
            self._solve()
        return self._df_sr

class Elasticity_series():
    def __init__(self, altconductor, basedesign, upliftforce, stepsize, startspt=0, endspt=None):
        self._ref = basedesign
        self._acp = altconductor
        self._df = None
        self._cycle(upliftforce, stepsize, startspt, endspt)
    
    def _cycle(self, upliftforce, stepsize, startspt, endspt):
        _df = pd.DataFrame()
        for index, row in self._acp.iterrows():
            if np.isfinite(row['MW Weight']):
                _acd = split_acd(self._acp,index)
                EL = Elasticity(_acd, self._ref, upliftforce, stepsize, startspt, endspt)
                _df_tmp = EL.dataframe()
                _df_tmp.type = row['Load Condition']
                _df = pd.concat([_df, _df_tmp], ignore_index=False)
        self._df = _df

    def dataframe(self):
        return self._df
    
class Elasticity():
    """ container for calculating span elasticity from an already calculated
    CatenaryFlexible """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, altconductor, basedesign, upliftforce, stepsize, startspt=0, endspt=None):
        self._ref = basedesign
        self._cp = altconductor
        self._df = None
        self._cycle(upliftforce, stepsize, startspt, endspt)

    def _cycle(self, upliftforce, stepsize, startspt, endspt):
        """ Solve elasticitiy graph and cache results"""
        _st = time.time()
        _elastic = AltCondition(self._cp, self._ref)
        _originalsag = _elastic.dataframe_cw()
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
        #print('Checking elasticity from STA', _startsta, 'to STA', _endsta)
        _avgcycletime = 0.75
        #print(' | ', int(len(_staval)), 'total cycles estimating to take', 
        #      round(_avgcycletime * int(len(_staval)),4), 'seconds')
        _diffmin = np.copy(_staval*0)
        _diffmax = np.copy(_staval*0)
        _cycleloops = np.copy(_staval*0)
        _cycletime = np.copy(_staval*0)
        i = 0
        for _sta in _staval:
            _cst = time.time()
            #print(i, ' | calculating uplift at STA =', _sta)
            _elastic.resetloadlist((-upliftforce, _sta))
            _upliftsag = _elastic.dataframe_cw()
            #_eldiff = _elastic.dataframe_cwdiff().Elevation
            _eldiff = GenFun.CWELDifference(_originalsag, _upliftsag)
            _diffmax[i] = np.nanmax(_eldiff)
            _cycleloops[i] = _elastic.getloops()
            _cet = time.time()
            _cycletime[i] = _cet - _cst
            i += 1
        _df_max = pd.DataFrame({
            'Stationing': _staval,
            'Rise (in)': _diffmax*12,
            'type': 'DiffMAX',
            'cable': 'DiffMAX'
            })
        #self._df = pd.concat([_df_min, _df_max], ignore_index=False)
        self._df = _df_max
        _et = time.time()
        _totaltime = _et - _st
        _meantime = np.mean(_cycletime)
        #print(i, ' total loops processed in', round(_totaltime,4), 'seconds',
        #      'with average cycle time of', round(_meantime*1000,4), 'milliseconds')

    def savetocsv(self, path):
        """ Output wire data to CSV """
        self._df.to_csv(path)

    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        return self._df
