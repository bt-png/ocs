import pandas as pd
import numpy as np
import altair as alt

import streamlit as st
import system as OCS
import general as GenFun

st.session_state.accesskey = st.session_state.accesskey

def calc1() -> None:
    with tab1:
        if ddfile is not None:
            with cdd_val:
                cdd0, cdd1, cdd2 = st.columns([0.1, 0.6, 0.3])
                _dd = pd.read_csv(ddfile)
                _df_cd, _df_acd, _df_hd, _df_bd, _df_sl = OCS.create_df_dd(_dd)
                with cdd1:
                    #st.write(_dd)
                    st.write('###### Loaded conductor particulars data:')
                    st.dataframe(_df_cd, hide_index=True)
                    st.write('###### Loaded alternate conductor particulars data:')
                    st.dataframe(_df_acd, hide_index=True)
                    st.write('###### Loaded span point loads:')
                    st.dataframe(_df_sl, hide_index=True)
                    P_DiscreteLoad_CW, STA_DiscreteLoad_CW, P_DiscreteLoad_MW, STA_DiscreteLoad_MW = [tuple(_df_sl.iloc[:,i].tolist()) for i in (2,1,3,1)]
                    #P=(0,0,0,0,0,0)
                    #st.write(P_DiscreteLoad_CW)
                with cdd2:
                    st.write('###### Loaded hanger design variables:')
                    st.dataframe(_df_hd, hide_index=True)
                    st.write('###### Loaded calculation constants:')
                    st.dataframe(_df_bd, hide_index=True)
                    #xStep, xRound, xMultiplier, yMultiplier, SteadyArmLength = [_df_bd.iloc[i,1] for i in (0,1,2,3,4)]
        if wrfile is not None:
            with cwr_val:
                cwr0, cwr1 = st.columns([0.1, 0.9])
                with cwr1:
                    wr = OCS.wire_run(wrfile)
                    st.write('###### First several rows of input file:')
                    st.dataframe(wr, hide_index=True)
    if not st.session_state['pauseCalc']:
        ## LAYOUT DESIGN
        if wrfile is not None and ddfile is not None:
            Nom = OCS.CatenaryFlexible(_dd, wr)
            df = Nom.dataframe()
            if st.session_state['altConductors']:
                LC = OCS.AltCondition_Series(_df_acd, Nom)
        elif wrfile is None and ddfile is None:
            Nom = OCS.CatenaryFlexible(OCS.sample_df_dd(), OCS.sample_df_wr())
            df = Nom.dataframe()
        else:
            with tab2:
                st.markdown('#### Provide data for both system design and wire run!')
                st.markdown('Sample data is only provided if no files have been uploaded.')
            st.stop()
        
        with tab2:
            if wrfile is None and ddfile is None:
                st.markdown('### SAMPLE DATA')
            st.write('### Catenary Wire Sag Plot')
            chart = st.line_chart(
                    data = df, x='Stationing', y='Elevation', color='cable'
                )
            if ddfile is not None and st.session_state['altConductors']:
                st.write(_df_acd)
                dfa = LC.dataframe()
                st.write('### Alternate Condition Sag Plot')
                chart = st.line_chart(
                    data = dfa, x='Stationing', y='Elevation', color ='type'
                )
        with tab3:
            if ddfile is not None and st.session_state['elasticity']:
                ec = OCS.Elasticity(_df_cd, Nom, pUplift, stepSize, startSPT, endSPT)
                dfe = ec.dataframe()
                st.write('### Elasticity')
                chart = st.line_chart(
                    data = dfe, x='Stationing', y='Rise (in)', color ='cable'
                )

        with tab4:
            st.write('#### Sag Data ', df)
            st.write('#### HA Data', Nom.dataframe_ha())
            if ddfile is not None and st.session_state['altConductors']:
                st.write('#### Alternate Conductor Data', dfa)
            if ddfile is not None and st.session_state['elasticity']:
                st.write('#### Elasticity', dfe)

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
    #with cdd_val:
        
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
with tab3:
    if ddfile is not None:
        e0, e1 = st.columns([0.5, 0.5])
        with e0:
            pUplift = st.number_input(label='Uplift Force (lbf)', value=25)
            stepSize = st.number_input(label='Resolution (ft)', min_value = 1, step=1, value=1)
        with e1:
            startSPT = st.number_input(label='Starting structure', value=1)
            endSPT = st.number_input(label='Ending structure', value=2)
calc1()
