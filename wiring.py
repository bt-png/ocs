import streamlit as st
import pandas as pd
import numpy as np
import conductor
import pyvista as pv
import panel as pn
from stpyvista import stpyvista

def system_types():
    return ('Structure', 'Single Conductor', 'Catenary System')

@st.cache_resource
def intro(dummy: str = "robot"):
    plotter = pv.Plotter()
    
    df1 = st.session_state.wiring_values['val1']['DataFrame']
    df2 = st.session_state.wiring_values['val2']['DataFrame']
    
    df = pd.concat([df1, df2], ignore_index=True)
    st.dataframe(df)
    
    spline1 = pv.Spline(df1)#.tube(radius=0.5*st.session_state.wiring_values['val1']['Diameter'])
    spline2 = pv.Spline(df2)#.tube(radius=0.5*st.session_state.wiring_values['val2']['Diameter'])
    plotter.add_mesh(spline1, color="green")
    plotter.add_mesh(spline2, color="red")

    
    head = pv.Cylinder(radius=3.5, height=8)
    #nose = pv.Cylinder(radius=0.5, height=8, direction=(0, 0, 1), center=(0, 0, 1.7))
    #eye_left = pv.Cylinder(
    #    radius=1.0, height=4, direction=(0, 0, 1), center=(-2.0, 1, 2)
    #)
    #eye_left_p = pv.Cylinder(
    #    radius=0.3, height=4.1, direction=(0, 0, 1), center=(-2.0, 1, 2)
    #)
    #eye_right = pv.Cylinder(
    #    radius=1.0, height=4, direction=(0, 0, 1), center=(2.0, 1, 2)
    #)
    #eye_right_p = pv.Cylinder(
    #    radius=0.3, height=4.1, direction=(0, 0, 1), center=(2.0, 1, 2)
    #)

    plotter.add_mesh(head, color="grey")
    #plotter.add_mesh(nose, color="red")
    #plotter.add_mesh(eye_left, color="white")
    #plotter.add_mesh(eye_right, color="white")
    #plotter.add_mesh(eye_left_p, color="green")
    #plotter.add_mesh(eye_right_p, color="green")

    #plane = pv.Plane(center=[0, -3.65, 0], direction=[0, 1, 0], i_size=12, j_size=12)
    #plotter.add_mesh(plane, color="#09ab3b", show_edges=True)

    plotter.background_color = "white"
    plotter.view_xy()
    #plotter.camera.azimuth = 25
    #plotter.camera.elevation = 15

    plotter.window_size = [450, 300]
    return plotter
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
            new_dict['DataFrame'] = _df
            st.session_state.wiring_values[_key] = new_dict
            
def draw():
    pv.start_xvfb()
    
    #stpyvista(plotter)
    plotter = intro()

    stpyvista(
        plotter,
        panel_kwargs=dict(
            orientation_widget=True, interactive_orientation_widget=True
        ),
    )
    
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
