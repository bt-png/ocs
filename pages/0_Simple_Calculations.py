import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

import general as GenFun

def plotSag(df):
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Span'], empty=False)
    line = alt.Chart(df).mark_line().encode(
        alt.X('Span:Q').scale(zero=False), 
        alt.Y('Offset (in):Q').scale(zero=False),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Span:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Offset (in):Q', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=600, height=300)
    st.write(chart)

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
            st.write('Calculated Iced Weight = ' + str(round(_c_iced_weight,2)) + ' (pounds per foot)')
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
            _c_tension_final = GenFun.ConductorTension(_c_tension,_c_weight,_c_weight+_c_iced_weight,_c_temp_diff,_c_equiv_span,_c_area)
            st.write('Calculated Final Tension = ' + str(round(_c_tension_final,2)) + ' (pounds)')
        with st.expander('Sag & Wind Blowoff'):
            _c_span = st.number_input(label='Span Length', value = 150.0, min_value=0.0, step = 10.0)
            #_c_span_loc = st.slider('Location of interest in Span', min_value=0.0, max_value=_c_span, value=0.5*_c_span)
            _x = range(0, int(_c_span)+1, 1)
            _c_sag = [-12 * GenFun.sag(_c_span, _c_tension_final, (_c_weight+_c_iced_weight), 0, x) for x in _x]
            _c_sag_bare = [-12 * GenFun.sag(_c_span, _c_tension, (_c_weight), 0, x) for x in _x]
            dfa = pd.DataFrame({
                'Span': _x,
                'Offset (in)': _c_sag,
                'type': 'Sag'
            })
            dfa_bare = pd.DataFrame({
                'Span': _x,
                'Offset (in)': _c_sag_bare,
                'type': 'Sag (Bare)'
            })
            _c_blowoff = [12 * GenFun.sag(_c_span, _c_tension_final, _c_windpressure*((_c_diameter+2*_c_iced_radius)/12),0, x) for x in _x]
            _c_blowoff_bare = [12 * GenFun.sag(_c_span, _c_tension, _c_windpressure*((_c_diameter)/12),0, x) for x in _x]
            dfb = pd.DataFrame({
                'Span': _x,
                'Offset (in)': _c_blowoff,
                'type': 'BlowOff (+)'
            })
            dfc = pd.DataFrame({
                'Span': _x,
                'Offset (in)': [-x for x in _c_blowoff],
                'type': 'BlowOff (-)'
            })
            dfb_bare = pd.DataFrame({
                'Span': _x,
                'Offset (in)': _c_blowoff_bare,
                'type': 'BlowOff (+Bare)'
            })
            dfc_bare = pd.DataFrame({
                'Span': _x,
                'Offset (in)': [-x for x in _c_blowoff_bare],
                'type': 'BlowOff (-Bare)'
            })
            df = pd.concat([dfa, dfb, dfc], ignore_index=True)
            plotSag(df)
            _ft, _in , _ftin = GenFun.FtIn(-min(_c_sag)/12)
            st.write('Max Sag = ', _ftin)
            _ft, _in , _ftin = GenFun.FtIn(max(_c_blowoff)/12)
            st.write('Max Blow-off = ', _ftin)
            