# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:36:43 2024

@author: TharpBM
"""
import numpy as np
import pandas as pd

# GENERAL

def FtIn(val):
    if val < 0:
        sign = '(-) '
    else:
        sign =''
    _val = abs(val)
    _roundIN = 16 #Round inches to 1/[]th of an inch
    _ft = int(_val)
    _in = (_val - _ft) * 12
    _in = round(_in * _roundIN, 0) / _roundIN
    if _ft > 0 and _in > 0:
        _strFtIn = sign + str(_ft) + "'-" + str(_in) + '"'
    elif _ft > 0:
        _strFtIn = sign + str(_ft) + "'"
    elif _in > 0:
        _strFtIn = sign + str(_in) + '"'
    elif _val == 0 or _in == 0:
        _strFtIn = '0"'
    else:
        _strFtIn = 'Error'
    return _ft, _in, _strFtIn

def WindPressure(windspeed, shapecoefficient=1):
    return windspeed ** 2 * 0.00256 * shapecoefficient

def IcedWeight(radius, iced_radius):
    _c_iced_area = np.pi * (((radius + iced_radius)/12)**2 - (radius/12)**2)
    _c_ice_density = 57 #lbf per cubic feet
    _c_iced_weight = _c_iced_area * _c_ice_density
    return _c_iced_weight

def RoundVal(arr,num):
    if num == 0:
        return round(arr,0)
    return (arr/num).round(0)*num

def shift(arr, num, fill_value=np.nan):
    val = np.empty_like(arr)
    if num > 0:
        val[:num] = fill_value
        val[num:] = arr[:-num]
    elif num < 0:
        val[num:] = fill_value
        val[:num] = arr[-num:]
    else:
        val[:] = arr
    return val

def df_pad(_df_original, name):
    _df = _df_original.copy().astype(str)
    _df.insert(0,0,0)
    _df.loc[-1] = (_df.columns.values)
    _df.loc[-2] = (0)
    _df.loc[-2,0] = str(name)
    _df.columns = range(_df.shape[1])
    _df = _df.sort_index().reset_index(drop=True)
    _df[0] = _df[0].replace(0,np.NaN)
    _df.iloc[0] = _df.iloc[0].replace(0,np.NaN)
    return _df

def df_add(_df1, _df2):
    _df_empty = pd.DataFrame(data=None, columns=[0], index=[0])
    _df = pd.concat([_df1, _df_empty, _df2])
    return _df

# WIRE RUN GENERAL
def WireRun_df(df, STARound):
    #expected DataFrame Headers [PoleID, STA, RailEL, MWHT, CWHT, PreSag, DeviationAngle
    val = df.copy()
    val.sort_values('STA',axis = 0, ascending=True)
    val['Deviation Angle'] = abs(val['Deviation Angle'])
    val['STA'] = (val['STA']/STARound).round(0)*STARound
    SpanLength=val['STA'].shift(-1)-val['STA']
    val['SpanLength']=SpanLength
    return val

def ScaleWireRun(WR,xScale,yScale):
    #expected DataFrame Headers [PoleID, STA, RailEL, MWHT, CWHT, PreSag, DeviationAngle, SpanLength
    val = WR.copy()
    val.iloc[:,1:] = val.iloc[:,1:]*[xScale,yScale,yScale,yScale,yScale,1,xScale]
    #val = WR*[1,xScale,yScale,yScale,yScale,yScale,1,xScale]
    return val

# SIMPLE SAG SPAN EQUATIONS
def sag(l,H,G,h,a):
    if l == 0:
        val = 0
    else:
        val = (((a * (l - a)) / 2) * (G / H))
        val += ((h * a) * (1 / l))
    return val

# SPAN GEOMETRY
def x_SPAN_LIST(WR, STA_Step, STARound):
    STAStart = RoundVal(WR['STA'].iloc[0],STARound)
    STAEnd = RoundVal(WR['STA'].iloc[-1],STARound)
    TotalElements=int((STAEnd-STAStart)/STA_Step+1)
    val = np.linspace(STAStart,STAEnd,num=TotalElements, endpoint=True)
    #val = np.arange(STAStart, STAEnd, RoundVal(STA_Step,STARound))
    val = np.unique(RoundVal(val,STARound))
    return val

def L_HA_QTY(WR, MaxLength):
    return (WR['SpanLength']/MaxLength).round(0)

def L_HA_STA(WR, HA_QTY, STARound):
    HASpacing=WR['SpanLength']/HA_QTY
    val = []
    val.append(WR['STA'].iloc[0])
    for i in range(0,len(HASpacing)):
        if HA_QTY[i]>0:
            for j in range(1,int(HA_QTY[i])+1):
                a_HA = (j-0.5)*HASpacing[i]
                val.append(WR['STA'].iloc[i]+a_HA)
    val.append(WR['STA'].iloc[-1])
    val = np.array(val)
    val = np.unique(RoundVal(val,STARound))
    return val

def LoadTribSpan(StationList):
    i = StationList[1:]
    j = StationList[:-1]
    return np.append((i-j),0)

def L_HA_EL(WR, HangerStationing, CableWeight, CableTension, STARound):
    j=0
    TribSpan = LoadTribSpan(HangerStationing)
    #HAEL = np.empty_like(HangerStationing)
    HAEL = HangerStationing*0
    for i in range(0,(len(HangerStationing))):
        if i == 1 and i < (len(HangerStationing)):
            HASag = sag(2*TribSpan[i],CableTension,CableWeight,0,TribSpan[i])
        elif i == (len(HangerStationing)-1) and i>0:
            HASag = sag(2*TribSpan[i],CableTension,CableWeight,0,TribSpan[i])
        else:
            HASag = sag(TribSpan[i],CableTension,CableWeight,0,0.5*TribSpan[i])
        if HangerStationing[i] > WR['STA'].iloc[j+1]:
            j += 1
        PriorSupportSTA = WR['STA'].iloc[j]
        PriorEL = WR['Rail EL'].iloc[j]+WR['CW Height'].iloc[j]
        NextSupportSTA = WR['STA'].iloc[j+1]
        NextEL = WR['Rail EL'].iloc[j+1]+WR['CW Height'].iloc[j+1]
        preSag = ((NextSupportSTA-PriorSupportSTA)-(HangerStationing[i]-PriorSupportSTA))*(HangerStationing[i]-PriorSupportSTA)
        preSag *= 8*WR['PreSag'].iloc[j]
        preSag /= (2*(NextSupportSTA-PriorSupportSTA)*(NextSupportSTA-PriorSupportSTA))
        HAEL[i] = (HangerStationing[i]-PriorSupportSTA)*((NextEL-PriorEL)/(NextSupportSTA-PriorSupportSTA))+PriorEL+HASag-preSag
    return HAEL

def DistanceFromHA(STAList, HangerStationing, STARound):
    #a = np.empty_like(STAList)
    a = STAList*0
    j = 0
    for i in range(0,len(STAList)-1):
        if RoundVal(STAList[i],STARound) == RoundVal(HangerStationing[j],STARound):
            j +=1
        a[i] = STAList[i] - HangerStationing[j-1]
    return a

def DistanceFromHA_WeightSpan(STAList, HangerStationing, STARound):
    #a = np.empty_like(STAList)
    a = STAList*0
    j=0
    for i in range(0,len(STAList)-1):
        if RoundVal(STAList[i],STARound) == RoundVal(HangerStationing[j],STARound):
            j +=1
        a[i] = 0.5 * (STAList[i+1] - STAList[i])
        a[i] += (STAList[i] - HangerStationing[j-1])
    return a

def HangerLength(Stationing, CWSag, MWSag, HangerStationing):
    HALength = MWSag[np.in1d(Stationing,HangerStationing)] - CWSag[np.in1d(Stationing,HangerStationing)]
    return HALength

def HangerLengthDifference(Stationing, OriginalCWSag, OriginalMWSag, NewCWSag, NewMWSag, HangerStationing):
    oHALength = OriginalMWSag[np.in1d(Stationing,HangerStationing)] - OriginalCWSag[np.in1d(Stationing,HangerStationing)]
    nHALength = NewMWSag[np.in1d(Stationing,HangerStationing)] - NewCWSag[np.in1d(Stationing,HangerStationing)]
    return nHALength - oHALength

def CWSupportELDifference(Stationing, OriginalCWSag, NewCWSag, SupportStationing):
    rSTA = Stationing[np.in1d(Stationing, SupportStationing)]
    oSPTEL = OriginalCWSag[np.in1d(Stationing, SupportStationing)]
    nSPTEL = NewCWSag[np.in1d(Stationing, SupportStationing)]
    #print('lenSupportSTA', len(SupportStationing), 'lenSTA', len(Stationing), 'lenOrig', len(oSPTEL), 'lenNew', len(nSPTEL))
    return (nSPTEL - oSPTEL, rSTA)

def CWELDifference(dforiginal, dfuplift):
    oSTA = dforiginal['Stationing']
    oCWEL = dforiginal['Elevation']
    nSTA = dfuplift['Stationing']
    nCWEL = dfuplift['Elevation']
    oSPTEL = oCWEL[np.in1d(oSTA, nSTA)]
    nSPTEL = nCWEL[np.in1d(nSTA, oSTA)]
    #print('lenSupportSTA', len(SupportStationing), 'lenSTA', len(Stationing), 'lenOrig', len(oSPTEL), 'lenNew', len(nSPTEL))
    return (nSPTEL - oSPTEL)

# SPAN LOADING
def LoadSpanWeight(StationList, UnitCableWeight):
    return UnitCableWeight*LoadTribSpan(StationList)

def LoadSpanTension(StationList, UnitCableTension):
    return StationList*0+UnitCableTension

def RadialLoad(Angle, Tension):
    return 2*Tension*np.sin(0.5*np.radians(Angle))

def SteadyArm_VerticalResettingForce(StaticRadialLoad, RadialLoadSTA, ELDiff, ELDiffSTA, SteadyArmLength):
    #print('lenRadial', len(StaticRadialLoad), 'lenELDiff', len(ELDiff), 'lenSTA')
    rL = StaticRadialLoad #[np.in1d(RadialLoadSTA, ELDiffSTA)]
    eL  = ELDiff #[np.in1d(RadialLoadSTA, ELDiffSTA)]
    return (rL*eL)/SteadyArmLength

def AddDiscreteLoadstoSpan(ExistingLoading, ExistingSTA, NewLoad, NewLoadSTA, STARound):
    P = np.copy(ExistingLoading)
    STAList = np.copy(NewLoadSTA)
    STAList = RoundVal(STAList,STARound)
    LOADList = np.copy(NewLoad)
    try:
        len(STAList)
    except TypeError:
        STAList = [STAList]
        LOADList = [LOADList]
    
    for i in range(0,len(STAList)):
        P += ((ExistingSTA == STAList[i])*LOADList[i])
    return P

def SupportLoadRight(STAList, LoadList, TensionList, HangerStationing, HangerElevation, STARound):
    aList = DistanceFromHA_WeightSpan(STAList, HangerStationing, STARound)
    #Load = np.empty_like(HangerStationing)
    Load = HangerStationing*0
    for i in range(0,len(HangerStationing)-1):
        Step = (STAList>=HangerStationing[i]) * (STAList < HangerStationing[i+1])
        Load[i] = 0
        if sum(Step) > 0:
            P = Step*LoadList
            a = Step*aList
            H = TensionList[np.nonzero(TensionList*Step)].mean()
            h = HangerElevation[i]-HangerElevation[i+1]
            l = HangerStationing[i+1]-HangerStationing[i]
            Load[i] = (np.nansum(P*a)-H*h)/l
    Load = np.roll(Load,1)
    Load[0] = 0
    return Load

def SupportLoadLeft(STAList, LoadList, HangerStationing, HangerSupportLoadRight, STARound):
    #SupportLoad = np.empty_like(HangerStationing)
    Load = HangerStationing*0
    for i in range(0,len(HangerStationing)-1):
        Step = (STAList>=HangerStationing[i]) * (STAList < HangerStationing[i+1])
        P = Step*LoadList
        Load[i] = np.nansum(P) - HangerSupportLoadRight[i+1]
    return Load

def SupportLoad(STAList, LoadList, TensionList, HangerStationing, HangerElevation, STARound):
    Right = SupportLoadRight(STAList, LoadList, TensionList, HangerStationing, HangerElevation, STARound)
    Left = SupportLoadLeft(STAList, LoadList, HangerStationing, Right, STARound)
    return Left+Right

def DiscreteMoment(STAList, LoadList, HangerStationing, STARound):
    aList = DistanceFromHA(STAList, HangerStationing, STARound)
    #DM = np.empty_like(STAList)
    DM = STAList*0
    for i in range(1,len(STAList)):
        vala = 0.5*(STAList[i] - STAList[i-1])
        valb = STAList[i-1] - STAList
        aPrior = vala+valb
        Step = (aPrior <= aList[i]) * (aPrior >= 0)
        DM[i] = sum(Step*LoadList*aPrior)
    return DM
