import pandas as pd
import numpy as np
import altair as alt

import streamlit as st
import system as OCS
import general as GenFun

st.session_state.accesskey = st.session_state.accesskey

@st.cache_data()
def SagData(data, wirerun) -> None:
    return OCS.CatenaryFlexible(data, wirerun)

@st.cache_data()
def altSagData(data, _ORIG) -> None:
    return OCS.AltCondition_Series(data, _ORIG)

@st.cache_data()
def elasticity(_df_cd, _BASE, pUplift, stepSize, startSPT, endSPT) -> None:
    EL = OCS.Elasticity(_df_cd, _BASE, pUplift, stepSize, startSPT, endSPT)
    df = EL.dataframe()
    return df

@st.cache_data()
def elasticityalt(_df_cd, _df_acd, _BASE, pUplift, stepSize, startSPT, endSPT) -> None:
    EL = OCS.Elasticity_series(_df_cd, _df_acd, _BASE, pUplift, stepSize, startSPT, endSPT)
    df = EL.dataframe()
    return df

@st.cache_data()
def preview_ddfile(ddfile) -> None:
    cdd0, cdd1, cdd2 = st.columns([0.1, 0.6, 0.3])
    with cdd1:
        #st.write(_dd)
        st.write('###### Loaded conductor particulars data:')
        st.dataframe(_df_cd, hide_index=True)
        st.write('###### Loaded alternate conductor particulars data:')
        st.dataframe(_df_acd, hide_index=True)
        st.write('###### Loaded span point loads:')
        st.dataframe(_df_sl, hide_index=True)
    with cdd2:
        st.write('###### Loaded hanger design variables:')
        st.dataframe(_df_hd, hide_index=True)
        st.write('###### Loaded calculation constants:')
        st.dataframe(_df_bd, hide_index=True)

@st.cache_data()
def preview_wrfile(wrfile) -> None:
    cwr0, cwr1 = st.columns([0.1, 0.9])
    with cwr1:
        st.write('###### First several rows of input file:')
        st.dataframe(wr, hide_index=True)  

@st.cache_data()
def PlotSag(_BASE) -> None:
    df = _BASE.dataframe()
    st.write('### Catenary Wire Sag Plot')
    chart = st.line_chart(
            data = df, x='Stationing', y='Elevation', color='cable'
        )
       
@st.cache_data()
def PlotSagSample(_BASE) -> None:
    df = _BASE.dataframe()
    st.markdown('### SAMPLE DATA')
    st.write('### Catenary Wire Sag Plot')
    chart = st.line_chart(
            data = df, x='Stationing', y='Elevation', color='cable'
        )

@st.cache_data()
def PlotSagaltCond(_BASE) -> None:
    LC = altSagData(_df_acd, _BASE)
    dfa = LC.dataframe()
    st.write(_df_acd)
    st.write('### Alternate Condition Sag Plot')
    chart = st.line_chart(
        data = dfa, x='Stationing', y='Elevation', color ='type'
    )

@st.cache_data()
def Plotelasticity(df) -> None:
    st.write('### Elasticity')
    chart = st.line_chart(
        data = df, x='Stationing', y='Rise (in)', color ='type'
    )

@st.cache_data()
def OutputSag(_BASE) -> None:
    st.write('#### Sag Data ', _BASE.dataframe())
    st.write('#### HA Data', _BASE.dataframe_ha())

@st.cache_data()
def OutputAltCond(_BASE) -> None:
    LC = altSagData(_df_acd, _BASE)
    dfa = LC.dataframe()
    st.write('#### Alternate Conductor Data', dfa)

@st.cache_data()
def Outputelasticity(df) -> None:
    st.write('#### Elasticity', df)

@st.cache_data()
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

st.set_page_config(page_title="CAT SAG", page_icon="📹")
st.markdown("# Simple Catenary Sag")
st.sidebar.header("CAT SAG")

st.write(
    """This app shows you the simple sag curves for a flexible catenary system
    utilizing a **sum of moments** method. The CW is assumed to be supported only
    at hanger locations, with elevations calculated based on designed elevation at 
    supports and the designated pre-sag."""
    )

if st.session_state['accesskey'] != st.secrets['accesskey']:
        st.stop()

tab1, tab2, tab3, tab4 = st.tabs(['Input', 'Sag Plot', 'Elasticity', 'Output'])

st.sidebar.checkbox('Pause Calculation', key='pauseCalc', value=False)
st.sidebar.checkbox('Consider Alt Conductors', key='altConductors', value=False)
st.sidebar.checkbox('Perform Elasticity Check', key='elasticity', value=False)

with tab1:
    cdd = st.container(border=True)
    cdd_val = st.container(border=False)
    cwr = st.container(border=True)
    cwr_val = st.container(border=False)
    with cdd:
        ##Design Data
        st.markdown("#### Load catenary system design data")
        sample_dd_csv = convert_df(OCS.sample_df_dd())
        st.download_button(
            label="### press to download sample csv file for Design data",
            data=sample_dd_csv,
            file_name="_dd_Data.csv",
            mime="text/csv"
        )

        ddfile = st.file_uploader(
                'Upload a properly formatted csv file containing Design data',
                type={'csv', 'txt'},
                accept_multiple_files = False,
                key='_ddfile'
                )
        if ddfile is not None:
            _dd = pd.read_csv(ddfile)
            _df_cd, _df_acd, _df_hd, _df_bd, _df_sl = OCS.create_df_dd(_dd)
            preview_ddfile(ddfile)
    with cwr:
        ##Wire Run Data
        st.markdown("#### Load layout design data for wire run")
        sample_wr_csv = convert_df(OCS.sample_df_wr())
        st.download_button(
            label="### press to download sample csv file for Wire Run data",
            data=sample_wr_csv,
            file_name="_WR_Data.csv",
            mime="text/csv"
        )
        wrfile = st.file_uploader(
                'Upload a properly formatted csv file containing Wire Run data',
                type={'csv', 'txt'},
                accept_multiple_files = False,
                key='_wrfile'
                )
        if wrfile is not None:
            wr = OCS.wire_run(wrfile)
            preview_wrfile(wrfile)
with tab2:
    if ddfile is not None and wrfile is not None:
        Nom = SagData(_dd, wr)
        PlotSag(Nom)
        if st.session_state['altConductors']:
            PlotSagaltCond(Nom)
    elif wrfile is None and ddfile is None:
        Nom = SagData(OCS.sample_df_dd(), OCS.sample_df_wr())
        PlotSagSample(Nom)
with tab3:
    if ddfile is not None and wrfile is not None and st.session_state['elasticity']:
        ec = None
        with st.form('Input values'):
            pUplift = st.number_input(label='Uplift Force (lbf)', value=25)
            stepSize = st.number_input(label='Resolution (ft)', min_value = 1, step=1, value=1)
            startSPT = st.number_input(label='Starting structure', value=1)
            endSPT = st.number_input(label='Ending structure', value=2)
            submit_form = st.form_submit_button('Calculate')
            if submit_form:
                if st.session_state['altConductors']:
                    ec = elasticityalt(_df_cd, _df_acd, Nom, pUplift, stepSize, startSPT, endSPT)
                else:
                    ec = elasticity(_df_cd, Nom, pUplift, stepSize, startSPT, endSPT)
        if ec is not None:
            Plotelasticity(ec)

with tab4:
    if ddfile is not None and wrfile is not None:
        OutputSag(Nom)
        if st.session_state['altConductors']:
            OutputAltCond(Nom)
        if st.session_state['elasticity'] and ec is not None:
            Outputelasticity(ec)

