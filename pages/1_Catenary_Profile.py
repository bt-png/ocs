import pandas as pd
import numpy as np

import streamlit as st
import system as OCS
import linebline as LL

st.session_state.accesskey = st.session_state.accesskey

def calc1() -> None:
    with tab1:
        if ddfile is not None:
            with cdd_val:
                cdd0, cdd1, cdd2 = st.columns([0.1, 0.6, 0.3])
                _dd = pd.read_csv(ddfile)
                with cdd1:
                    #st.write(_dd)
                    st.write('###### Loaded conductor particulars data:')
                    #st.dataframe(_df_cd, hide_index=True)
                    st.write('###### Loaded alternate conductor particulars data:')
                    #st.dataframe(_df_cd, hide_index=True)
                    st.write('###### Loaded span point loads:')
                    #st.dataframe(_df_cd, hide_index=True)
                with cdd2:
                    st.write('###### Loaded system design variables:')
                    #st.dataframe(_df_sd, hide_index=True)
                    st.write('###### Loaded calculation constants:')
                    #st.dataframe(_df_cc, hide_index=True)
                st.write(_dd)
        if wrfile is not None:
            with cwr_val:
                cwr0, cwr1 = st.columns([0.1, 0.9])
                with cwr1:
                    wr = OCS.wire_run(wrfile)
                    st.write('###### First several rows of input file:')
                    st.dataframe(wr.head(), hide_index=True)
    if not st.session_state['pauseCalc']:
        ## LAYOUT DESIGN
        if wrfile is not None:
            Nom = OCS.CatenaryFlexible(OCS.sample_df_cp(), wr)
            Nom.resetloads(OCS.sample_df_sl())
        else:
            Nom = OCS.CatenaryFlexible(OCS.sample_df_cp(), OCS.sample_df_wr())
            Nom.resetloads(OCS.sample_df_sl())
        df = Nom.dataframe()
        with tab2:
            st.write('### Catenary Wire Sag Plot')
            chart = st.line_chart(
                data = df, x='Stationing', y='Elevation', color='cable'
                )
        with tab3:
            st.write('#### Sag Data ', df)
            st.write('#### HA Data', Nom.dataframe_ha())

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

tab1, tab2, tab3 = st.tabs(['Input', 'Results', 'Output'])

st.sidebar.checkbox('Pause Calculation', key='pauseCalc', value=False)

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

calc1()
