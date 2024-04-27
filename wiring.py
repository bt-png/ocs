import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import conductor

def system_types():
    return ('Structure', 'Single Conductor', 'Catenary System')

#@st.experimental_fragment
def form_SC(_key,):
    with st.form(key=_key):
        _diameter = st.number_input(label='Cable Diameter (inches)', value=0.62, min_value=0.1)
        _weight = st.number_input(label='Cable Weight (pounds per foot)', value=1.063, min_value=0.1)
        _tension = st.number_input(label='Nominal Tension (pounds)', value=2000, min_value=0)
        col1, col2 = st.columns([1,1])
        _startsta = col1.number_input(label='Start Stationing (feet)', value=15000)
        _endsta = col2.number_input(label='End Stationing (feet)', value=15150)
        _startht = col1.number_input(label='Start Height (feet)', value=18.0)
        _endht = col2.number_input(label='End Height (feet)', value=18.0)
        _startoff = col1.number_input(label='Start Offset (feet)', value=0.75)
        _endoff = col2.number_input(label='End Offset (feet)', value=0.75)
        pressed =  st.form_submit_button(label='Update')
        if pressed:
            new_dict = ({
                'Type': 'SC',
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
            _df['type'] = _key
            new_dict['DataFrame'] = _df
            st.session_state.wiring_values[_key] = new_dict
            
def draw():
    df1 = st.session_state.wiring_values['val1']['DataFrame']
    df2 = st.session_state.wiring_values['val2']['DataFrame']
    #df = pd.concat([df1, df2], ignore_index=True)
    #st.write(df)
    #fig = go.Figure()
    #fig.add_trace(data=go.Scatter3d(x=df['Stationing'], y=df['Offset'], z=df['Elevation'], mode='lines'))
    #fig.add_trace(name='val2', x=df2['Stationing'], y=df2['Offset'], z=df2['Elevation'], mode='lines', selector=dict(type='scatter3d'))
    #data=go.Scatter3d(x=df['Stationing'], y=df['Offset'], z=df['Elevation'], mode='lines')
    #fig.update_layout(scene_aspectmode='cube') #cube
    #fig.update_layout(scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1, z=2)) # elevation doubled
    #st.plotly_chart(fig, use_container_width=False, theme='streamlit')
    trace1 = go.Scatter3d(
        x=df1['Stationing'], 
        y=df1['Offset'], 
        z=df1['Elevation'],
        name='val1',
        mode='lines'
        )
    trace2 = go.Scatter3d(
        x=df2['Stationing'], 
        y=df2['Offset'], 
        z=df2['Elevation'],
        name='val2',
        mode='lines'
        )
    data = go.Data([trace1, trace2])
    layout = go.Layout(
        showlegend=True,
        scene=go.Scene(
            xaxis=go.XAxis(title='x axis title'),
            yaxis=go.YAxis(title='y axis title'),
            zaxis=go.ZAxis(title='z axis title')            
        ),
        aspectmode='cube'
    )
    
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)
    
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
    if type1 == 'Single Conductor':
        with col1:
            form_SC('val1')
    if type2 == 'Single Conductor':
        with col2:
            form_SC('val2')
    if ((len(st.session_state.wiring_values['val1']) > 1) & (len(st.session_state.wiring_values['val2']) > 1)):
        if st.button('Draw'):
            draw()
