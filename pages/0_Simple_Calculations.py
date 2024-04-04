import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

import general as GenFun

def plotSag(df):
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Span'], empty=False)
    line = alt.Chart(df).mark_line().encode(
        alt.X('Span:Q').scale(zero=False), 
        alt.Y('Sag (in):Q').scale(zero=False),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Span:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Sag (in):Q', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=600, height=300)
    st.write(chart)

def plotBlowOff(df_pos, df_neg):
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Span'], empty=False)
    linepos = alt.Chart(df_pos).mark_line().encode(
        alt.X('Span:Q').scale(zero=False), 
        alt.Y('Offset (in):Q').scale(zero=False),
        alt.Color('type')
    )
    lineneg = alt.Chart(df_neg).mark_line().encode(
        alt.X('Span:Q').scale(zero=False), 
        alt.Y('Offset (in):Q').scale(zero=False),
        alt.Color('type')
    )
    selectors = alt.Chart(df_pos).mark_point().encode(
        alt.X('Span:Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    pointspos = linepos.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    textpos = linepos.mark_text(align='right', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Offset (in):Q', alt.value(' ')))
    rulespos = alt.Chart(df_pos).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    pointsneg = lineneg.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    textneg = lineneg.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Offset (in):Q', alt.value(' ')))
    rulesneg = alt.Chart(df_neg).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    chart = alt.layer(linepos + lineneg + selectors + pointspos + rulespos + textpos + pointsneg + rulesneg + textneg).properties(width=600, height=300)
    st.write(chart)

def plotSection(df):
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Wind Speed'], empty=False)
    line = alt.Chart(df).mark_line().encode(
        alt.X('Offset (in):Q').scale(zero=False), 
        alt.Y('Sag (in):Q').scale(zero=False),
        alt.Color('type')
    )
    selectors = alt.Chart(df).mark_point().encode(
        alt.X('Offset (in):Q'),
        alt.Y('Sag (in):Q'),
        opacity=alt.value(0)
    ).add_params(nearest)
    points = line.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'Wind Speed:Q', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Offset (in):Q').transform_filter(nearest)
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

if 'accesskey' not in st.session_state:
    st.session_state.accesskey = ''
else:
    st.session_state.accesskey = st.session_state.accesskey

st.set_page_config(page_title="Simple Calcs", page_icon="📹")
st.markdown("# Simple Reference Calculations")
st.sidebar.header("Simple Calculations")
st.write(
    """Use this app for simple reference calculations.  
    You are welcome!."""
)

conductor, wiring, resetting = st.tabs(['Conductor Data', 'Wiring Plan', 'Cantilever Resetting Force'])
with conductor:
    cdd0, cdd1 = st.columns([0.05, 0.95])
    with cdd1:
        with st.expander('Cable Dimensions'):
            _c_weight = st.number_input(label='Cable Weight (pounds per foot)', value=1.063, min_value=0.1)
            _c_diameter = st.number_input(label='Cable Diameter (inches)', value=0.86, min_value=0.1)
            _c_iced_radius = st.number_input(label='Iced Radius (inches)', value=0.25, min_value=0.0)
            _c_iced_weight = GenFun.IcedWeight(_c_diameter, _c_iced_radius)
            if _c_iced_radius > 0:
                st.caption('Calculated Iced Weight = ' + str(round(_c_iced_weight,2)) + ' (pounds per foot)')
            _c_chk_ice = st.checkbox(label='Include Iced Weight in Nominal Tension Condition', value=False)
        with st.expander('Wind Data'):
            _c_windspeed = st.number_input(label='Wind Speed (mph)', value=40, min_value=0)
            _c_shapecoeff = st.number_input('Conductor Shape Coefficient', value = 1.0, min_value=0.1)
            _c_windpressure = GenFun.WindPressure(_c_windspeed, _c_shapecoeff)
            st.caption('Calculated Wind Pressure = ' + str(_c_windpressure) + ' (pounds per square feet)')
            _c_windload_bare = _c_windpressure*((_c_diameter)/12)
            _c_windload_iced = _c_windpressure*((_c_diameter+2*_c_iced_radius)/12)
            if _c_iced_radius == 0:
                 st.caption('Wind Load = ' + str(round(_c_windload_bare,2)) + ' (pounds per foot)')
            else:
                st.caption('Wind Load:')
                st.caption('Bare conductor = ' + str(round(_c_windload_bare,2)) + ' (pounds per foot)')
                st.caption('Iced conductor = ' + str(round(_c_windload_iced,2)) +' (pounds per foot)')
            _c_chk_wind = st.checkbox(label='Wind Load Influences Cable Tension', value=True)
        with st.expander('Cable Tension'):
            _c_tension = st.number_input(label='Nominal Tension (pounds)', value = 3000, min_value=0)
            _c_equiv_span = st.number_input(label='Equivalent Span', value = 160, min_value = 0)
            _c_temp_diff = st.number_input(label='Temperature Variation', value = 0)
            _c_area = st.number_input(label='Cross Sectional Area (square inches)', value = 0.276, min_value=0.0)
            _c_section_loss = st.number_input(label='Conductor Section Loss (%)', value = 0, min_value=0)
            _c_E = st.number_input(label='Modulus of Elasticity', value = 1.7*10**7, format='%2e')
            _c_alpha = st.number_input(label='Thermal Elongation Coefficient', value = 9.4*10**-6, format='%2e')
            _c_initialweight = _c_weight+_c_chk_ice*_c_iced_weight
            _c_initialwindweight = (_c_weight**2 + (_c_windload_bare*_c_chk_wind)**2)**0.5
            _c_finalweight = (_c_weight*(100-_c_section_loss)/100)+_c_iced_weight
            _c_finalwindweight = (_c_finalweight**2 + (_c_windload_iced*_c_chk_wind)**2)**0.5
            _c_tension_initial_wind = GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_weight, _c_initialwindweight, 0, _c_equiv_span, _c_area, _c_area)
            _c_tension_final = GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalweight, _c_temp_diff, _c_equiv_span, _c_area, _c_area*(100-_c_section_loss)/100)
            _c_tension_final_wind = GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalwindweight, _c_temp_diff, _c_equiv_span, _c_area, _c_area*(100-_c_section_loss)/100)
            if _c_chk_wind:
                st.caption('Calculated Final Tensions:')
                st.caption('No Wind = ' + str(round(_c_tension_final,2)) + ' (pounds)')
                st.caption('Wind = ' + str(round(_c_tension_final_wind,2)) +' (pounds)')
            else:
                st.caption('Calculated Final Tension = ' + str(round(_c_tension_final,2)) + ' (pounds)')
        with st.expander('Sag & Wind Blowoff', expanded=True):
            _c_span = st.number_input(label='Span Length', value = 150.0, min_value=0.0, step = 10.0)
            #_c_span_loc = st.slider('Location of interest in Span', min_value=0.0, max_value=_c_span, value=0.5*_c_span)
            _x = range(0, int(_c_span)+1, 1)
            
            st.subheader('Conductor Sag')
            _c_sag_initial = [-12 * GenFun.sag(_c_span, _c_tension, _c_initialweight, 0, x) for x in _x]
            dfa = pd.DataFrame({
                'Span': _x,
                'Sag (in)': _c_sag_initial,
                'type': '1. Initial'
            })
            _c_sag_final = [-12 * GenFun.sag(_c_span, _c_tension_final, _c_finalweight, 0, x) for x in _x]
            dfb = pd.DataFrame({
                'Span': _x,
                'Sag (in)': _c_sag_final,
                'type': '2. Final'
            })
            df = pd.concat([dfa, dfb], ignore_index=True)
            plotSag(df)
            _ft, _in , _ftin = GenFun.FtIn(-min(_c_sag_initial)/12)
            st.caption('Initial Tension = ' + str(round(_c_tension,2)) + ' (pounds), Initial Weight = ' + str(round(_c_initialweight,2)) + ' (pounds per foot), Max Sag = ' + _ftin)
            _ft, _in , _ftin = GenFun.FtIn(-min(_c_sag_final)/12)
            st.caption('Final Tension = ' + str(round(_c_tension_final_wind,2)) + ' (pounds), Final Weight = ' + str(round(_c_finalweight,2)) + ' (pounds per foot), Max Sag = ' + _ftin)
            
            st.subheader('Conductor Blow-Off')
            _c_blowoff_initial = [12 * GenFun.sag(_c_span, _c_tension_initial_wind, _c_windload_bare,0, x) for x in _x]
            dfb = pd.DataFrame({
                'Span': _x,
                'Offset (in)': _c_blowoff_initial,
                'type': '1. Initial Bare'
            })
            dfc = pd.DataFrame({
                'Span': _x,
                'Offset (in)': [-x for x in _c_blowoff_initial],
                'type': '1. Initial Bare'
            })
            _c_blowoff_final = [12 * GenFun.sag(_c_span, _c_tension_final_wind, _c_windload_iced,0, x) for x in _x]
            dfd = pd.DataFrame({
                'Span': _x,
                'Offset (in)': _c_blowoff_final,
                'type': '2. Final'
            })
            dfe = pd.DataFrame({
                'Span': _x,
                'Offset (in)': [-x for x in _c_blowoff_final],
                'type': '2. Final'
            })
            df_p = pd.concat([dfb, dfd], ignore_index=True)
            df_n = pd.concat([dfc, dfe], ignore_index=True)
            plotBlowOff(df_p, df_n)
            _ft, _in , _ftin = GenFun.FtIn(max(_c_blowoff_initial)/12)
            st.caption('Initial Tension = ' + str(round(_c_tension_initial_wind,2)) + ' (pounds), Wind Force = ' + str(round(_c_windload_bare,2)) + ' (pounds per foot), Max Blow-Off = ' + _ftin)
            _ft, _in , _ftin = GenFun.FtIn(max(_c_blowoff_final)/12)
            st.caption('Final Tension = ' + str(round(_c_tension_final_wind,2)) + ' (pounds), Wind Force = ' + str(round(_c_windload_iced,2)) + ' (pounds per foot), Max Blow-Off = ' + _ftin)
            
            st.subheader('Section View at Mid-Span')
            _windrange = range(0,_c_windspeed+1,1)
            _windpressure = [GenFun.WindPressure(ws, _c_shapecoeff) for ws in _windrange]
            _windload_bare = [wp*((_c_diameter)/12) for wp in _windpressure]
            _windweight = [(_c_weight**2 + (wl*_c_chk_wind)**2)**0.5 for wl in _windload_bare]
            _tensions = [GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_weight, ww, 0, _c_equiv_span, _c_area, _c_area) for ww in _windweight]
            _x = [12 * GenFun.sag(_c_span, t, wl, 0, 0.5*_c_span) for t, wl in zip(_tensions, _windload_bare)]
            _y = [-12 * GenFun.sag(_c_span, t, _c_weight, 0, 0.5*_c_span) for t in _tensions]
            dff = pd.DataFrame({
                'Offset (in)': _x,
                'Sag (in)': _y,
                'Wind Speed': _windrange,
                'type': '1. Initial (Bare)'
            })
            dfg = pd.DataFrame({
                'Offset (in)': [-val for val in _x],
                'Sag (in)': _y,
                'Wind Speed': [-val for val in _windrange],
                'type': '1. Initial (Bare)'
            })
            df1 = pd.concat([dfg[::-1],dff])
            _windload_iced = [wp*((_c_diameter+2*_c_iced_radius)/12) for wp in _windpressure]
            _windweight = [(_c_finalweight**2 + (wl*_c_chk_wind)**2)**0.5 for wl in _windload_iced]
            _tensions = [GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_initialweight, ww, _c_temp_diff, _c_equiv_span, _c_area, _c_area*(100-_c_section_loss)/100) for ww in _windweight]
            _x = [12 * GenFun.sag(_c_span, t, wl, 0, 0.5*_c_span) for t, wl in zip(_tensions, _windload_iced)]
            _y = [-12 * GenFun.sag(_c_span, t, _c_finalweight, 0, 0.5*_c_span) for t in _tensions]
            dfh = pd.DataFrame({
                'Offset (in)': _x,
                'Sag (in)': _y,
                'Wind Speed': _windrange,
                'type': '2. Final'
            })
            dfi = pd.DataFrame({
                'Offset (in)': [-val for val in _x],
                'Sag (in)': _y,
                'Wind Speed': [-val for val in _windrange],
                'type': '2. Final'
            })
            df2 = pd.concat([dfi[::-1],dfh])
            df = pd.concat([df1,df2])
            plotSection(df)