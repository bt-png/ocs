import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import general as GenFun

def sample_df_conductor_data():
    _df = pd.DataFrame({
        'Load Condition': ['OP', 'OP-Loaded', 'OP-UnLoaded', ],
        'Conductor Type': ['350 KCMIL HD', '500 KCMIL 19-HD', '500 KCMIL 19-HD'],
        'Bare Conductor Weight': [1.063, 2.633, 1.544],
        'Bare Conductor Diameter': [0.620, 0.811, 0.811],
        'Bare Conductor Area': [0.2758, 0.3926, 0.3926],
        'Nominal Conductor Tension': [3000, 6000, 5353],
        'Nominal Temperature': [70, 70, 70],
        'Equivalent Span': [21, 150, 150],
        'Conductor Modulus of Elasticity': [1.6*10**7, 1.6*10**7, 1.6*10**7],
        'Conductor Coefficient of Thermal Expansion': [9.4*10**-6, 9.4*10**-6, 9.4*10**-6],
        'Conductor Nominal Breaking Load': [11810, 21590, 21590],
        'Condition Radial Thickness of Ice': [0.0, 0.0, 0.0],
        'Condition Wind Speed': [40, 40, 40],
        'Condition Conductor Section Loss': [0, 0, 0],
        'Condition Temperature': [110, 110, 110],
        'Calc Wind Shape Coefficient': [1.0, 1.0, 1.0],
        'Calc Ice Load is Nominal State': [False, False, False],
        'Calc Wind Load Influences Cable Tension': [True, True, True]
    })
    return _df

@st.cache_data()
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

#@st.cache_data()
def conductorData(conductorfile):
    return pd.read_csv(conductorfile)

def plotSag(_df):
    df = _df.copy()
    df['FtInText'] = df['Sag (in)'].apply(lambda x : GenFun.FtIn_Simple(x/12))
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
    text = line.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'FtInText:N', alt.value(' ')))
    rules = alt.Chart(df).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    chart = alt.layer(line + selectors + points + rules + text).properties(width=600, height=300)
    st.write(chart)

def plotBlowOff(_df_pos, _df_neg):
    df_pos = _df_pos.copy()
    df_neg = _df_neg.copy()
    df_pos['FtInText'] = df_pos['Offset (in)'].apply(lambda x : GenFun.FtIn_Simple(x/12))
    df_neg['FtInText'] = df_neg['Offset (in)'].apply(lambda x : GenFun.FtIn_Simple(x/12))
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
    textpos = linepos.mark_text(align='right', dx=5, dy=-5).encode(text=alt.condition(nearest, 'FtInText:N', alt.value(' ')))
    rulespos = alt.Chart(df_pos).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    pointsneg = lineneg.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    textneg = lineneg.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'FtInText:N', alt.value(' ')))
    rulesneg = alt.Chart(df_neg).mark_rule(color='gray').encode(x='Span:Q').transform_filter(nearest)
    chart = alt.layer(linepos + lineneg + selectors + pointspos + rulespos + textpos + pointsneg + rulesneg + textneg).properties(width=600, height=300)
    st.write(chart)

def plotSection(_df):
    df = _df.copy()
    df['FtInText_O'] = df['Offset (in)'].apply(lambda x : GenFun.FtIn_Simple(x/12))
    df['FtInText_S'] = df['Sag (in)'].apply(lambda x : GenFun.FtIn_Simple(x/12))
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

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with st.container():
        to_filter_columns = ('Load Condition', 'Conductor Type')#st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            #left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            user_cat_input = st.multiselect(
                f"Values for {column}",
                df[column].unique(),
                default=list(df[column].unique())
            )
            if len(user_cat_input) > 0:
                df = df[df[column].isin(user_cat_input)]
            
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)
    df_with_selections.iloc[0,0] = True
    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_order=[
            'Select',
            'Load Condition',
            'Conductor Type',
            'Condition Temperature',
            'Condition Wind Speed',
            'Condition Radial Thickness of Ice',
            'Equivalent Span',
            'Condition Conductor Section Loss'
            ],
        column_config={
            'Select': st.column_config.CheckboxColumn(required=True),
            'Load Condition': st.column_config.TextColumn(label='Condition'),
            'Conductor Type': st.column_config.TextColumn(label='Type'),
            'Condition Temperature': st.column_config.NumberColumn(label='Temp', width='small'),
            'Condition Wind Speed': st.column_config.NumberColumn(label='Wind', width='small'),
            'Condition Radial Thickness of Ice': st.column_config.NumberColumn(label='Ice', width='small'),
            'Equivalent Span': st.column_config.NumberColumn(label='EQ Span', width='small'),
            'Condition Conductor Section Loss': st.column_config.NumberColumn(label='Wear', width='small'),
            },
        disabled=df.columns
    )

    # Filter the dataframe using the temporary column, then drop the column
    if edited_df.Select.any():
        selected_rows = df[edited_df.Select]
        selected_rows = selected_rows.reset_index(drop=True)
    else:
        selected_rows = df
    return selected_rows

@st.cache_data
def CalcTension(
        _c_E, 
        _c_alpha,
        _c_tension,
        _c_initialweight,
        _c_finalwindweight, 
        _c_temp, 
        _c_equiv_span, 
        _c_area,
        _c_section_loss,
        _c_temp_nom=0):
    try:
        if len(_c_temp) > 1:
            tension = [
                GenFun.ConductorTension(
                _c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalwindweight,
                (_t-_c_temp_nom), _c_equiv_span, _c_area, 
                _c_area*(100-_c_section_loss)/100) 
                for _t in _c_temp]
        else:
            tension = [
                GenFun.ConductorTension(
                _c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalwindweight,
                (_c_temp-_c_temp_nom), _c_equiv_span, _c_area, 
                _c_area*(100-_c_section_loss)/100)
                ]
    except:
        tension = GenFun.ConductorTension(
            _c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalwindweight,
            (_c_temp-_c_temp_nom), _c_equiv_span, _c_area, 
            _c_area*(100-_c_section_loss)/100)
    return pd.DataFrame({'Temperature': _c_temp, 'Tension': tension})

class SingleWire():
    """ A simple container for the single wire system containing all design data
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, _simple_dict):
        self._dict = _simple_dict
        self._solved = False
        self._sag = None
        self.empty = False
        self._sag = None
        self._solve()

    def _solve(self):
        STAStart = self._dict['Start STA']
        STAEnd = self._dict['End STA']
        HTStart = self._dict['Start Height']
        HTEnd = self._dict['End Height']
        yStart = self._dict['Start Offset']
        yEnd = self._dict['End Offset']
        UnitCableWeight = self._dict['Weight']
        UnitCableTension = self._dict['Tension']
        STA_Step=1
        STARound=1
        
        StationList = GenFun.Spans(STAStart, STAEnd, STA_Step, STARound)
        LoadList = GenFun.LoadSpanWeight(StationList, UnitCableWeight)
        TensionList = GenFun.LoadSpanTension(StationList, UnitCableTension)
        HangerStationing = np.array([STAStart, STAEnd])
        HangerElevation = np.array([HTStart, HTEnd])
        Elevations = GenFun.SagSpanData(StationList, LoadList, TensionList, HangerStationing, HangerElevation, STARound)
        Offset = GenFun.SpanOffset(StationList, yStart, yEnd)
        self._sag = pd.DataFrame({
            'Stationing': StationList,
            'Offset': Offset,
            'Elevation': Elevations
            })
        
    def dataframe(self):
        """ return a pandas dataFrame of the wire """
        if not self._solved:
            self._solve()
        return self._sag

def run():
    with st.expander('Load from File'):
        ##Cable Data
        #st.markdown("#### Load conductor design data")
        sample_conductor_csv = convert_df(sample_df_conductor_data())
        st.download_button(
            label="### press to download sample csv file for cable data",
            data=sample_conductor_csv,
            file_name="_conductor_Data.csv",
            mime="text/csv"
        )
        conductorfile = st.file_uploader(
                'Upload a properly formatted csv file containing conductor data',
                type={'csv', 'txt'},
                accept_multiple_files = False,
                key='_ddfile'
                )
        if conductorfile is not None:
            _conductordata = conductorData(conductorfile)
        else:
            _conductordata = sample_df_conductor_data()
            #_df_cd, _df_acd, _df_hd, _df_bd, _df_sl = ddData(_dd)
            #preview_ddfile(ddfile)
        st.markdown("#### Select saved cable condition data")
        _conductor = filter_dataframe(_conductordata).to_dict()
    if len(_conductor['Bare Conductor Weight']) != 1:
        st.warning('Please select ONE entry')
    else:
        _c_weight = _conductor['Bare Conductor Weight'][0]
        _c_diameter = _conductor['Bare Conductor Diameter'][0]
        _c_iced_radius = _conductor['Condition Radial Thickness of Ice'][0]
        _c_windspeed = int(_conductor['Condition Wind Speed'][0])
        _c_shapecoeff = float(_conductor['Calc Wind Shape Coefficient'][0])
        _c_tension = int(_conductor['Nominal Conductor Tension'][0])
        _c_equiv_span = int(_conductor['Equivalent Span'][0])
        _c_temp_diff = _conductor['Condition Temperature'][0]-_conductor['Nominal Temperature'][0]
        _c_area = _conductor['Bare Conductor Area'][0]
        _c_section_loss = int(_conductor['Condition Conductor Section Loss'][0])
        _c_E = _conductor['Conductor Modulus of Elasticity'][0]
        _c_alpha = _conductor['Conductor Coefficient of Thermal Expansion'][0]
        _c_BS = _conductor['Conductor Nominal Breaking Load'][0]
        _c_chk_ice = _conductor['Calc Ice Load is Nominal State'][0]
        _c_chk_wind = _conductor['Calc Wind Load Influences Cable Tension'][0]
        _c_span = _c_equiv_span
        st.markdown('''---''')
        with st.expander('Cable Dimensions'):
            _c_weight = st.number_input(label='Cable Weight (pounds per foot)', value=_c_weight, min_value=0.1)
            _c_diameter = st.number_input(label='Cable Diameter (inches)', value=_c_diameter, min_value=0.1)
            _c_iced_radius = st.number_input(label='Iced Radius (inches)', value=_c_iced_radius, min_value=0.0)
            _c_iced_weight = GenFun.IcedWeight(_c_diameter, _c_iced_radius)
            if _c_iced_radius > 0:
                st.caption('Calculated Iced Weight = ' + str(round(_c_iced_weight,2)) + ' (pounds per foot)')
            _c_chk_ice = st.checkbox(label='Include Iced Weight in Nominal Tension Condition', value=_c_chk_ice)
        with st.expander('Wind Data'):
            _c_windspeed = st.number_input(label='Wind Speed (mph)', value=_c_windspeed, min_value=0)
            _c_shapecoeff = st.number_input('Conductor Shape Coefficient', value=_c_shapecoeff, min_value=0.1)
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
            _c_chk_wind = st.checkbox(label='Wind Load Influences Cable Tension', value=_c_chk_wind)
        with st.expander('Cable Tension'):
            _c_tension = st.number_input(label='Nominal Tension (pounds)', value=_c_tension, min_value=0)
            _c_equiv_span = st.number_input(label='Equivalent Span', value=_c_equiv_span, min_value = 0)
            _c_temp_diff = st.number_input(label='Temperature Variation', value=_c_temp_diff)
            _c_area = st.number_input(label='Cross Sectional Area (square inches)', value=_c_area, min_value=0.0)
            _c_section_loss = st.number_input(label='Conductor Section Loss (%)', value=_c_section_loss, min_value=0)
            _c_E = st.number_input(label='Modulus of Elasticity', value=_c_E, format='%2e')
            _c_alpha = st.number_input(label='Thermal Elongation Coefficient', value=_c_alpha, format='%2e')
            _c_BS = st.number_input(label='Breaking Strength (pounds)', value=_c_BS)
            _c_initialweight = _c_weight+_c_chk_ice*_c_iced_weight
            _c_initialwindweight = (_c_weight**2 + (_c_windload_bare*_c_chk_wind)**2)**0.5
            _c_finalweight = (_c_weight*(100-_c_section_loss)/100)+_c_iced_weight
            _c_finalwindweight = (_c_finalweight**2 + (_c_windload_iced*_c_chk_wind)**2)**0.5
            _c_tension_initial_wind = GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_weight, _c_initialwindweight, 0, _c_equiv_span, _c_area, _c_area)
            _c_tension_final = GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalweight, _c_temp_diff, _c_equiv_span, _c_area, _c_area*(100-_c_section_loss)/100)
            _c_tension_final_wind = GenFun.ConductorTension(_c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalwindweight, _c_temp_diff, _c_equiv_span, _c_area, _c_area*(100-_c_section_loss)/100)
            _c_BS_final = _c_BS*(100-_c_section_loss)/100
            if _c_chk_wind:
                st.caption('Calculated Final Tensions:')
                st.caption(f'No Wind = {str(round(_c_tension_final,2))} (pounds), F.O.S = {str(round(_c_BS_final/_c_tension_final,2))}')
                st.caption(f'Wind = {str(round(_c_tension_final_wind,2))} (pounds), F.O.S = {str(round(_c_BS_final/_c_tension_final_wind,2))}')
            else:
                st.caption('Calculated Final Tension = ' + str(round(_c_tension_final,2)) + ' (pounds)')
        with st.expander('Sag & Wind Blowoff', expanded=True):
            _c_span = st.number_input(label='Span Length', value=_c_span, min_value=0, step = 10)
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
            st.caption(
                f'Initial Tension = {str(round(_c_tension,2))} (pounds), \
                Initial Weight = {str(round(_c_initialweight,2))} (pounds per foot), \
                Max Sag = {_ftin}, \
                F.O.S = {str(round(_c_BS/_c_tension,2))}'
                )
            _ft, _in , _ftin = GenFun.FtIn(-min(_c_sag_final)/12)
            st.caption(
                f'Final Tension = {str(round(_c_tension_final_wind,2))} (pounds), \
                Final Weight = {str(round(_c_finalweight,2))} (pounds per foot), \
                Max Sag = {_ftin}, \
                F.O.S = {str(round(_c_BS_final/_c_tension_final_wind,2))}'
                )
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
            st.caption(
                f'Initial Tension = {str(round(_c_tension_initial_wind,2))} (pounds), \
                Wind Force = {str(round(_c_windload_bare,2))} (pounds per foot), \
                Max Blow-Off = {_ftin}, \
                F.O.S = {str(round(_c_BS/_c_tension_initial_wind,2))}'
                )
            _ft, _in , _ftin = GenFun.FtIn(max(_c_blowoff_final)/12)
            st.caption(
                f'Final Tension = {str(round(_c_tension_final_wind,2))} (pounds), \
                Wind Force = {str(round(_c_windload_iced,2))} (pounds per foot), \
                Max Blow-Off = {_ftin} \
                F.O.S = {str(round(_c_BS_final/_c_tension_final_wind,2))}'
                )
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
        with st.expander('Export', expanded=False):
            _c_temp_nom = st.number_input(label='Nominal Temperature', value=70)
            _c_temp_min = st.number_input(label='Min Temperature', value=0)
            _c_temp_max = st.number_input(label='Max Temperature', value=110)
            _c_temp_step = st.number_input(label='Temperature Step', value=10)
            if st.button(label='Populate Tension Table'):
                temp = np.arange(_c_temp_min, (_c_temp_max+_c_temp_step), _c_temp_step)
                CalcTension.clear()
                tension = CalcTension(
                    _c_E, _c_alpha, _c_tension, _c_initialweight, _c_finalwindweight, 
                    temp, _c_equiv_span, _c_area, _c_section_loss, _c_temp_nom)
                st.dataframe(tension, hide_index=True, column_config={
                    'Temperature': st.column_config.TextColumn(label='Temp', width='small'),
                    'Tension': st.column_config.NumberColumn(width='medium', step=1),
                })
                    