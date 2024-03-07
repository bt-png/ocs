import streamlit as st
import pandas as pd

def df_pad(_df_original, name):
    _df = _df_original.copy()
    _df.insert(0,0,0)
    _df.loc[-1] = (_df.columns.values)
    _df.loc[-2] = (0)
    _df.loc[-2,0] = name
    _df.columns = range(_df.shape[1])
    _df = _df.sort_index().reset_index(drop=True)
    return _df

def design_data_cd(path):
    """Conductor Particulars"""
    _df = pd.DataFrame({
        'Cable': ['MW', 'CW', 'HA'],
        'Weight': [0, 0, 0],
        'Tension': [0, 0, 0]
    })
    header = False
    for line in path:
        val = line.decode('utf-8')
        head = 'Conductor Particulars'
        if header:
            if val[:1] != ',':
                header = False
                break
            _lineval = val.split(',')
            for i in range(0,len(_df['Cable'])):
                if _lineval[1] == _df['Cable'].iloc[i]:
                    _df['Weight'].iloc[i] = float(_lineval[2])
                    _df['Tension'].iloc[i] = float(_lineval[3])
        if val[:len(head)] == head:
            header = True
    return _df

def design_data_acd(path):
    """Alternate Conductor Particulars"""
    _df = pd.DataFrame({
        'Cable': ['MW', 'CW', 'HA'],
        'Weight': [0, 0, 0],
        'Tension': [0, 0, 0]
    })
    header = False
    for line in path:
        val = line.decode('utf-8')
        head = 'Alternate Conductor Particulars'
        if header:
            if val[:1] != ',':
                header = False
                break
            _lineval = val.split(',')
            for i in range(0,len(_df['Cable'])):
                if _lineval[1] == _df['Cable'].iloc[i]:
                    _df['Weight'].iloc[i] = float(_lineval[2])
                    _df['Tension'].iloc[i] = float(_lineval[3])
        if val[:len(head)] == head:
            header = True
    return _df

def design_data_sd(path):
    """System Design Variables"""
    _df = pd.DataFrame({
        'Variable Description': ['Max HA Spacing',
                                'Min CW Load',
                                'Min HA Length',
                                'Steady Arm Length'],
        'Value': [0, 0, 0, 0]
    })
    header = False
    for line in path:
        val = line.decode('utf-8')
        head = 'System Design Variables'
        if header:
            if val[:1] != ',':
                header = False
                break
            _lineval = val.split(',')
            for i in range(0,len(_df['Variable Description'])):
                if _lineval[1] == _df['Variable Description'].iloc[i]:
                    _df['Value'].iloc[i] = float(_lineval[2])
        if val[:len(head)] == head:
            header = True
    return _df

def design_data_cc(path):
    """Calculation Constants"""
    _df = pd.DataFrame({
        'Variable Description': ['HA Accuracy',
                                'xStep',
                                'xRound',
                                'xMultiplier',
                                'yMultiplier'],
        'Value': [0, 0, 0, 0, 0]
    })
    header = False
    for line in path:
        val = line.decode('utf-8')
        head = 'Calculation Constants'
        if header:
            if val[:1] != ',':
                header = False
                break
            _lineval = val.split(',')
            for i in range(0,len(_df['Variable Description'])):
                if _lineval[1] == _df['Variable Description'].iloc[i]:
                    _df['Value'].iloc[i] = float(_lineval[2])
        if val[:len(head)] == head:
            header = True
    return _df
