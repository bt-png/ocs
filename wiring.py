# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:04:04 2024

@author: TharpBM
"""

import streamlit as st
import pandas as pd
import conductor
import numpy as np
import scipy.spatial.distance as distance
from scipy.interpolate import griddata
import general as GenFun
import system as OCS

use_plotter = 'plotly' #'matplotlib' #'pyvista'
if use_plotter == 'plotly':
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
elif use_plotter == 'matplotlib':
    import matplotlib.pyplot as plt
elif use_plotter == 'pyvista':
    import pyvista as pv

def system_types():
    return ('Surface', 'Single Conductor', 'Catenary System')

def form_SF(_key):
    #SURFACE
    with st.form(key=_key):
        col1, col2 = st.columns([1,1])
        col3, col4 = st.columns([1,1])
        col1.write('Edge 1')
        _sta1 = col1.number_input(label='Edge 1 Stationing (feet)', value=000)
        _ht1 = col1.number_input(label='Edge 1 Elevation (feet)', value=24.0)
        _off1 = col1.number_input(label='Edge 1 Offset (feet)', value=0.75)
        col2.write('Edge 2')
        _sta2 = col2.number_input(label='Edge 2 Stationing (feet)', value=000)
        _ht2 = col2.number_input(label='Edge 2 Elevation (feet)', value=24.0)
        _off2 = col2.number_input(label='Edge 2 Offset (feet)', value=-0.75)
        col3.write('Edge 3')
        _sta3 = col1.number_input(label='Edge 3 Stationing (feet)', value=150)
        _ht3 = col1.number_input(label='Edge 3 Elevation (feet)', value=24.0)
        _off3 = col1.number_input(label='Edge 3 Offset (feet)', value=0.75)
        col4.write('Edge 4')
        _sta4 = col2.number_input(label='Edge 4 Stationing (feet)', value=150)
        _ht4 = col2.number_input(label='Edge 4 Elevation (feet)', value=24.0)
        _off4 = col2.number_input(label='Edge 4 Offset (feet)', value=-0.75)
        pressed =  st.form_submit_button(label='Update')
        if pressed:
            new_dict = ({
                'Type': 'SF',
                'STA': [_sta1, _sta2, _sta3, _sta4],
                'Offset': [_off1, _off2, _off3, _off4],
                'Elevation': [_ht1, _ht2, _ht3, _ht4]
                })
            #x = []
            #x = np.outer(np.linspace(_startsta, _endsta,50), np.ones(50))
            #y = np.outer(np.linspace(_startsta, _endsta,50), np.ones(50))
            st.session_state.wiring_values[_key] = new_dict

def form_SW(_key,):
    #SINGLE WIRE
    with st.form(key=_key):
        _diameter = st.number_input(label='Cable Diameter (inches)', value=0.62, min_value=0.1)
        _weight = st.number_input(label='Cable Weight (pounds per foot)', value=1.063, min_value=0.1)
        _tension = st.number_input(label='Nominal Tension (pounds)', value=2000, min_value=0)
        col1, col2 = st.columns([1,1])
        col3, col4 = st.columns([1,1])
        _startsta = col1.number_input(label='Start Stationing (feet)', value=000)
        _endsta = col2.number_input(label='End Stationing (feet)', value=150)
        _startht = col1.number_input(label='Start Height (feet)', value=18.0)
        _endht = col2.number_input(label='End Height (feet)', value=18.0)
        _startoff = col1.number_input(label='Start Offset (feet)', value=0.75)
        _endoff = col2.number_input(label='End Offset (feet)', value=0.75)
        pressed =  st.form_submit_button(label='Update')
        if pressed:
            new_dict = ({
                'Type': 'SW',
                'Diameter': _diameter, 
                'Weight': _weight, 
                'Tension': _tension,
                'Start STA': _startsta, 
                'End STA': _endsta,
                'Start Height': _startht, 
                'End Height': _endht,
                'Start Offset': _startoff, 
                'End Offset': _endoff,
                })
            SW = conductor.SingleWire(new_dict)
            _df = SW.dataframe()
            new_dict['DataFrame'] = _df
            st.session_state.wiring_values[_key] = new_dict

def form_SC(_key,):
    #SIMPLE CATENARY
    with st.form(key=_key):
        st.write('Contact Wire')
        c_diameter = st.number_input(label='CW Diameter (inches)', value=0.62, min_value=0.1)
        c_weight = st.number_input(label='CW Weight (pounds per foot)', value=1.063, min_value=0.1)
        c_tension = st.number_input(label='CW Nominal Tension (pounds)', value=2000, min_value=0)
        st.write('Messenger Wire')
        m_diameter = st.number_input(label='MW Diameter (inches)', value=0.811, min_value=0.1)
        m_weight = st.number_input(label='MW Weight (pounds per foot)', value=1.544, min_value=0.1)
        m_tension = st.number_input(label='MW Nominal Tension (pounds)', value=6000, min_value=0)
        st.write('Hanger Wire')
        h_diameter = st.number_input(label='HA Diameter (inches)', value=0.125, min_value=0.1)
        h_weight = st.number_input(label='HA Weight (pounds per foot)', value=0.026, min_value=0.0)
        h_spacing = st.number_input(label='Max Spacing (feet)', value=28, min_value=1)
        col1, col2 = st.columns([1,1])
        col3, col4 = st.columns([1,1])
        st.write('Single Span')
        _startsta = col1.number_input(label='Start Stationing (feet)', value=000)
        _endsta = col2.number_input(label='End Stationing (feet)', value=150)
        c_startht = col1.number_input(label='CW Start Height (feet)', value=18.0)
        c_endht = col2.number_input(label='CW End Height (feet)', value=18.0)
        m_startht = col1.number_input(label='MW Start Height (feet)', value=20.0)
        m_endht = col2.number_input(label='MW End Height (feet)', value=20.0)
        _startoff = col1.number_input(label='Start Offset (feet)', value=0.75)
        _endoff = col2.number_input(label='End Offset (feet)', value=0.75)
        pressed =  st.form_submit_button(label='Update')
        if pressed:
            df_cp = OCS.create_df_cp(m_weight,m_tension, c_weight, c_tension, h_weight)
            df_acp = OCS.create_df_acp(None, None, None, None)
            df_hd = OCS.create_df_hd(h_spacing, 0, 3/12, 1/50/12)
            df_bd = OCS.sample_df_bd()
            df_sl = OCS.create_df_sl(None, None, None, None)
            df_dd = OCS.combine_df_dd(df_cp, df_acp, df_hd, df_bd, df_sl)
            df_wr = OCS.create_df_wr(['start', 'end'], [_startsta, _endsta], [0, 0], [m_startht, m_endht], [c_startht, c_endht], [0, 0], [0, 0])
            new_dict = ({
                'Type': 'SC',
                'DF_CP': df_cp, 
                'DF_DD': df_dd,
                'DF_WR': df_wr,
                'Start Offset': _startoff, 
                'End Offset': _endoff,
                'MW Diameter': m_diameter,
                'CW Diameter': c_diameter,
                'HA Diameter': h_diameter,
                })
            SC = OCS.CatenaryFlexible(df_dd, df_wr)
            _df = SC.dataframe()
            OCS.Add_Stagger_Offset(SC, _startoff, _endoff)
            _df = SC.dataframe()
            _df_ha = SC.dataframe_ha()
            new_dict['DataFrame'] = _df
            new_dict['DataFrameHA'] = _df_ha
            st.session_state.wiring_values[_key] = new_dict


def dist_SW(_df1, _df2):
    return distance.cdist(_df1, _df2)

def ReportDistance():
    if (st.session_state.wiring_values['val1']['Type'] == 'SW'):
        df1 = st.session_state.wiring_values['val1']['DataFrame'].copy()
    elif (st.session_state.wiring_values['val1']['Type'] == 'SC'):
        df1 = st.session_state.wiring_values['val1']['DataFrame'].copy()
        df1.drop(labels=['type', 'cable'], axis=1, inplace=True)
        df1 = df1.reset_index(drop=True)
        _df = st.session_state.wiring_values['val1']['DataFrameHA'].copy()   
        for index, row in _df.iterrows():
            df_ha = pd.DataFrame({
                'Stationing': np.full((10,),row['Stationing']),
                'Offset': np.full((10,),row['Offset']),
                'Elevation': np.linspace(start=row['MW Elevation'], stop=row['CW Elevation'], num=10)
                })
            df1 = pd.concat([df1, df_ha], ignore_index=False)
    if (st.session_state.wiring_values['val2']['Type'] == 'SW'):
        df2 = st.session_state.wiring_values['val2']['DataFrame'].copy()
    elif (st.session_state.wiring_values['val2']['Type'] == 'SC'):
        df2 = st.session_state.wiring_values['val2']['DataFrame'].copy()
        df2.drop(labels=['type', 'cable'], axis=1, inplace=True)
        df2 = df2.reset_index(drop=True)
        _df = st.session_state.wiring_values['val2']['DataFrameHA'].copy()   
        for index, row in _df.iterrows():
            df_ha = pd.DataFrame({
                'Stationing': np.full((10,),row['Stationing']),
                'Offset': np.full((10,),row['Offset']),
                'Elevation': np.linspace(start=row['MW Elevation'], stop=row['CW Elevation'], num=10)
                })
            df2 = pd.concat([df2, df_ha], ignore_index=False)
    try:
        dist = distance.cdist(df1, df2).min()
    except:
        dist=None
    if dist != None:
        st.write(f"Minimum clearance is: {GenFun.FtIn_Simple(dist)}")

def draw_SF(_plotter, _key):
    _color = 'green' if _key == 'val1' else 'red'
    if use_plotter == 'plotly':
        x=st.session_state.wiring_values[_key]['STA']
        y=st.session_state.wiring_values[_key]['Offset']
        z=st.session_state.wiring_values[_key]['Elevation']
        X, Y = np.meshgrid(x,y)
        Z = griddata((x,y),z,(X,Y), method='cubic')
        _plotter.add_trace(
            go.Surface(z=Z, x=x, y=y),
            row=1, col=1)
        #_plotter.add_trace(
        #   go.Scatter(x=st.session_state.wiring_values[_key]['Offset'], y=st.session_state.wiring_values[_key]['Stationing'], line=dict(width=1, color=_color), mode='lines', name=_key),
        #    row=1, col=2)
        #_plotter.add_trace(
        #    go.Scatter(x=st.session_state.wiring_values[_key]['Stationing'], y=st.session_state.wiring_values[_key]['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
        #    row=2, col=1)
        #_plotter.add_trace(
        #    go.Scatter(x=st.session_state.wiring_values[_key]['Offset'], y=st.session_state.wiring_values[_key]['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
        #    row=2, col=2)
    elif use_plotter == 'matplotlib':
        None
    elif use_plotter == 'pyvista':
        None
        
def draw_SW(_plotter, _key):
    _color = 'green' if _key == 'val1' else 'red'
    df = st.session_state.wiring_values[_key]['DataFrame'].copy()
    if use_plotter == 'plotly':
        _plotter.add_trace(
            go.Scatter3d(x=df['Stationing'], y=df['Offset'], z=df['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=1, col=1)
        _plotter.add_trace(
            go.Scatter(x=df['Offset'], y=df['Stationing'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=1, col=2)
        _plotter.add_trace(
            go.Scatter(x=df['Stationing'], y=df['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=2, col=1)
        _plotter.add_trace(
            go.Scatter(x=df['Offset'], y=df['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=2, col=2)
    elif use_plotter == 'matplotlib':
        None
    elif use_plotter == 'pyvista':
        spline = pv.Spline(df).tube(radius=0.5*st.session_state.wiring_values[_key]['Diameter']/12)
        _plotter.add_mesh(spline, color=_color)

def draw_SC_HA(_plotter, _key):
    _color = 'green' if _key == 'val1' else 'red'
    #df_mw = st.session_state.wiring_values[_key]['DataFrame'].copy()
    #df_mw = df_mw[df_mw['cable'] == 'MW'].copy()
    #_offset = GenFun.SpanOffset(df_mw['Stationing'].to_numpy(), st.session_state.wiring_values[_key]['Start Offset'], st.session_state.wiring_values[_key]['End Offset'])
    #df_mw['Offset'] = _offset
    _df = st.session_state.wiring_values[_key]['DataFrameHA'].copy()
    #df_temp = pd.merge(df, df_mw, how='inner', on='Stationing')
    #df = df.reset_index(drop=True)
    #df['Offset'] = df_temp['Offset']    
    for index, row in _df.iterrows():
        _df_ha = pd.DataFrame({
            'Stationing': [row['Stationing'], row['Stationing']],
            'Offset': [row['Offset'], row['Offset']],
            'Elevation': [row['MW Elevation'], row['CW Elevation']]
            })
        if use_plotter == 'plotly':
            _plotter.add_trace(go.Scatter3d(x=_df_ha['Stationing'], y=_df_ha['Offset'], z=_df_ha['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
                row=1, col=1)
            _plotter.add_trace(
                go.Scatter(x=_df_ha['Offset'], y=_df_ha['Stationing'], line=dict(width=1, color=_color), mode='lines', name=_key),
                row=1, col=2)
            _plotter.add_trace(
                go.Scatter(x=_df_ha['Stationing'], y=_df_ha['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
                row=2, col=1)
            _plotter.add_trace(
                go.Scatter(x=_df_ha['Offset'], y=_df_ha['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
                row=2, col=2)
        elif use_plotter == 'matplotlib':
            None
        elif use_plotter == 'pyvista':
            ha = pv.Spline(_df_ha).tube(radius=0.5*st.session_state.wiring_values[_key]['HA Diameter']/12)
            _plotter.add_mesh(ha, color=_color)
    
def draw_SC(_plotter, _key):
    _color = 'green' if _key == 'val1' else 'red'
    df = st.session_state.wiring_values[_key]['DataFrame'].copy()
    
    df_mw = df[df['cable'] == 'MW'].copy()
    df_mw.drop(labels=['type', 'cable'], axis=1, inplace=True)
    #_offset = GenFun.SpanOffset(df_mw['Stationing'].to_numpy(), st.session_state.wiring_values[_key]['Start Offset'], st.session_state.wiring_values[_key]['End Offset'])
    #df_mw.insert(loc=1, column='Offset', value=_offset)
    
    df_cw = df[df['cable'] == 'CW'].copy()
    df_cw.drop(labels=['type', 'cable'], axis=1, inplace=True)
    #_offset = GenFun.SpanOffset(df_cw['Stationing'].to_numpy(), st.session_state.wiring_values[_key]['Start Offset'], st.session_state.wiring_values[_key]['End Offset'])
    #df_cw.insert(loc=1, column='Offset', value=_offset)
    
    if use_plotter == 'plotly':
        _plotter.add_trace(
            go.Scatter3d(x=df_mw['Stationing'], y=df_mw['Offset'], z=df_mw['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=1, col=1)
        _plotter.add_trace(
            go.Scatter(x=df_mw['Offset'], y=df_mw['Stationing'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=1, col=2)
        _plotter.add_trace(
            go.Scatter(x=df_mw['Stationing'], y=df_mw['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=2, col=1)
        _plotter.add_trace(
            go.Scatter(x=df_mw['Offset'], y=df_mw['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=2, col=2)
        _plotter.add_trace(
            go.Scatter3d(x=df_cw['Stationing'], y=df_cw['Offset'], z=df_cw['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=1, col=1)
        _plotter.add_trace(
            go.Scatter(x=df_cw['Offset'], y=df_cw['Stationing'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=1, col=2)
        _plotter.add_trace(
            go.Scatter(x=df_cw['Stationing'], y=df_cw['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=2, col=1)
        _plotter.add_trace(
            go.Scatter(x=df_cw['Offset'], y=df_cw['Elevation'], line=dict(width=1, color=_color), mode='lines', name=_key),
            row=2, col=2)
    elif use_plotter == 'matplotlib':
        _plotter.plot(df_mw['Stationing'],df_mw['Offset'],df_mw['Elevation'], label='MW')
    elif use_plotter == 'pyvista':
        spline_mw = pv.Spline(df_mw).tube(radius=0.5*st.session_state.wiring_values[_key]['MW Diameter']/12)
        _plotter.add_mesh(spline_mw, color=_color)
        spline_cw = pv.Spline(df_cw).tube(radius=0.5*st.session_state.wiring_values[_key]['CW Diameter']/12)
        _plotter.add_mesh(spline_cw, color=_color)
    
    draw_SC_HA(_plotter, _key)


def draw_element(_plotter, _key):
    if st.session_state.wiring_values[_key]['Type'] == 'SF':
        draw_SF(_plotter, _key)
    elif st.session_state.wiring_values[_key]['Type'] == 'SW':
        draw_SW(_plotter, _key)
    elif st.session_state.wiring_values[_key]['Type'] == 'SC':
        draw_SC(_plotter, _key)

def draw():
    if False:
        plotter = go.Figure()
        draw_element(plotter, 'val1')
        draw_element(plotter, 'val2')
        plotter.update_layout(
            scene=dict(
                aspectratio = dict(x=1, y=1/20, z=1/20),
                aspectmode = 'manual'
                ),
            )
        st.plotly_chart(plotter)
    elif use_plotter == 'plotly':
        plotter = make_subplots(rows=2, cols=2,
                                column_widths=[0.7, 0.3],
                                specs=[[{'type': 'scatter3d'}, {'type': 'xy'}],
                                      [{'type': 'xy'}, {'type': 'xy'}]])
        draw_element(plotter, 'val1')
        draw_element(plotter, 'val2')
        plotter.update_layout(
            showlegend=False,
            scene=dict(
                aspectratio = dict(x=1, y=1/10, z=1/10),
                aspectmode = 'manual'
                ),
            )
        st.plotly_chart(plotter)
    elif use_plotter == 'matplotlib':
        plotter = plt.figure().add_subplot(projection='3d')
        draw_element(plotter, 'val1')
        draw_element(plotter, 'val2')
        st.pyplot(plotter, clear_figure=True)
    elif use_plotter == 'pyvista':
        plotter = pv.Plotter()
        draw_element(plotter, 'val1')
        draw_element(plotter, 'val2')
        plotter.show()
    
def run():
    if 'wiring_values' not in st.session_state:
        st.session_state.wiring_values = ({'val1': ({'values': 1}), 'val2': ({'values': 2})})
    else:
        st.session_state.wiring_values = st.session_state.wiring_values
      
    col1, col2 = st.columns([1,1])
    type1 = col1.selectbox(
        label='system 1', 
        options=system_types(), 
        placeholder='Type of element...',
        index=None, 
        label_visibility='hidden'
        )
    
    type2 = col2.selectbox(
        label='system 2', 
        options=system_types(), 
        placeholder='Type of element...',
        index=None, 
        label_visibility='hidden'
        )
    with col1:
        if type1 == 'Surface':
            form_SF('val1')
        elif type1 == 'Single Conductor':
            form_SW('val1')
        elif type1 == 'Catenary System':
            form_SC('val1')
    with col2:
        if type2 == 'Surface':
            form_SF('val2')
        elif type2 == 'Single Conductor':
            form_SW('val2')
        elif type2 == 'Catenary System':
            form_SC('val2')
    if ((len(st.session_state.wiring_values['val1']) > 1) & (len(st.session_state.wiring_values['val2']) > 1)):
        if st.button('Draw'):
            draw()
            ReportDistance()
