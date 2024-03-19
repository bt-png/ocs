import streamlit as st
import numpy as np
import pandas as pd
import general as GenFun

def dataframe_with_selections(df, inputkey):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    df_with_selections.iloc[0,0] = True
    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
        key=inputkey
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

st.session_state.accesskey = st.session_state.accesskey

st.set_page_config(page_title="Simple Calcs", page_icon="📹")
st.markdown("# Simple Reference Calculations")
st.sidebar.header("Simple Calculations")
st.write(
    """Use this app for simple reference calculations.  
    You are welcome!."""
)

#if st.session_state['accesskey'] != st.secrets['accesskey']:
#    st.stop()

conductor, wiring, resetting = st.tabs(['Conductor Data', 'Wiring Plan', 'Cantilever Resetting Force'])
with conductor:
    cdd0, cdd1 = st.columns([0.05, 0.95])
    with cdd1:
        with st.expander('Cable Dimensions'):
            _c_weight = st.number_input(label='Cable Weight (pounds per foot)', value=1.063, min_value=0.1)
            _c_diameter = st.number_input(label='Cable Diameter (inches)', value=0.86, min_value=0.1)
            _c_iced_radius = st.number_input(label='Iced Radius (inches)', value=0.0, min_value=0.0)
            _c_iced_weight = GenFun.IcedWeight(_c_diameter, _c_iced_radius)
            st.write('Calculated Iced Weight = ', str(_c_iced_weight), '(pounds per foot)')
        with st.expander('Wind Data'):
            _c_windspeed = st.number_input(label='Wind Speed (mph)', value=40, min_value=0)
            _c_shapecoeff = st.number_input('Conductor Shape Coefficient', value = 1.0, min_value=0.1)
            _c_windpressure = GenFun.WindPressure(_c_windspeed, _c_shapecoeff)
            st.write('Calculated Wind Pressure = ', str(_c_windpressure), '(pounds per square feet)')
        with st.expander('Cable Tension'):
            _c_tension = st.number_input(label='Nominal Tension (pounds)', value = 3000, min_value=0)
            _c_area = st.number_input(label='Cross Sectional Area (square inches)', value = 0.276, min_value=0.0)
            _c_equiv_span = st.number_input(label='Equivalent Span', value = 160, min_value = 0)
            _c_temp_diff = st.number_input(label='Temperature Variation', value = 0)
            _c_tension_final = GenFun.ConductorTension(_c_tension,_c_weight,_c_iced_weight,_c_temp_diff,_c_equiv_span,_c_area)
            #st.write('Calculated Final Tension = ', str(_c_tension_final), '(pounds)')
        with st.expander('Sag & Wind Blowoff'):
            _c_span = st.number_input(label='Span Length', value = 150.0, min_value=0.0, step = 10.0)
            _c_span_loc = st.slider('Location of interest in Span', min_value=0.0, max_value=_c_span, value=0.5*_c_span)
            _c_sag = GenFun.sag(_c_span, _c_tension, (_c_weight+_c_iced_weight), 0, _c_span_loc)
            _ft, _in , _ftin = GenFun.FtIn(_c_sag)
            st.write('Sag at location = ', _ftin)
            _c_blowoff = GenFun.sag(_c_span, _c_tension, _c_windpressure*((_c_diameter+2*_c_iced_radius)/12),0, _c_span_loc)
            _ft, _in , _ftin = GenFun.FtIn(_c_blowoff)
            st.write('Blow-off at location = ', _ftin)
