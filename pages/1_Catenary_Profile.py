import pandas as pd
import numpy as np

import streamlit as st
import system as OCS

st.session_state.accesskey = st.session_state.accesskey
df_cd = None

def calc1() -> None:
    with tab1:
         ## CONDUCTOR PARTICULARS AND LOADING CONDITIONS
        st.write('##### Input conductor particulars:')
        cond_data = pd.DataFrame({
            'Cable': ['MW', 'CW', 'HA'],
            'Weight': [1.544, 1.063, 0.2],
            'Tension': [5000, 3300, 0]
            })
        econd_data = st.data_editor(cond_data, hide_index=True)
        mwWeight = econd_data.Weight[cond_data.Cable == 'MW'].iloc[0]
        cwWeight = float(econd_data.Weight[cond_data.Cable == 'CW'].iloc[0])
        haWeight = float(econd_data.Weight[cond_data.Cable == 'HA'].iloc[0])
        mwTension = float(econd_data.Tension[cond_data.Cable == 'MW'].iloc[0])
        cwTension = float(econd_data.Tension[cond_data.Cable == 'CW'].iloc[0])
        cN = OCS.conductor_particulars(
            (mwWeight, mwTension), (cwWeight, cwTension), haWeight
            )
        ## POINT LOADS
        st.write('##### Input point loads:')
        pl_data = pd.DataFrame({
            'Description': ['Description of point load for reference'],
            'Stationing': [0],
            'CW Loading': [0],
            'MW Loading': [0]
            })
        epl_data = st.data_editor(pl_data, hide_index=True, num_rows="dynamic")
        df_loading = epl_data[~epl_data['Stationing'].isnull()]
        df_loading.replace(to_replace=np.nan, value=0, inplace=True)
        if df_loading.empty:
            df_loading = pl_data
        #df_loading = df_loading.transpose()
        df_loading = (
            df_loading['Description'].values,
            df_loading['Stationing'].values,
            df_loading['CW Loading'].values,
            df_loading['MW Loading'].values
            )

    if wrfile is not None and not st.session_state['pauseCalc']:
        ## LAYOUT DESIGN       
        Nom = OCS.CatenaryFlexible(cN, wr)
        Nom.resetloads(df_loading)
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

def design_data(path):
    df_cd = design_data_cd(path)

def design_data_cd(path):
    _df = pd.DataFrame({
        'Cable': ['MW', 'CW', 'HA'],
        'Weight': [0, 0, 0],
        'Tension': [0, 0, 0]
    })
    header = False
    for line in path:
        val = line.decode('utf-8')
        head = 'Conductor Particulars'
        if header:
            if val[:1] != ',':
                header = False
                break
            _lineval = val.split(',')
            if _lineval[1] == 'MW':
                _df.Weight.iloc[0] = float(_lineval[2])
                _df.Tension.iloc[0] = float(_lineval[3])
            if _lineval[1] == 'CW':
                _df.Weight.iloc[1] = float(_lineval[2])
                _df.Tension.iloc[1] = float(_lineval[3])
            if _lineval[1] == 'HA':
                _df.Weight.iloc[2] = float(_lineval[2])
        if val[:len(head)] == head:
            st.write(val)
            header = True
    st.dataframe(_df, hide_index=True)
    return _df

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
    with st.container(border=True):
        ##Design Data
        #sample_dd_csv = convert_df(OCS.sample_wr_df())
        st.markdown("#### Load catenary system design data - not yet functional")
        #st.download_button(
        #    label="### press to download sample csv file for Design data",
        #    data=sample_dd_csv,
        #    file_name="_dd_Data.csv",
        #    mime="text/csv"
        #)

        ddfile = st.file_uploader(
                'Upload a properly formatted csv file containing Catenary Design data',
                type={'csv', 'txt'},
                accept_multiple_files = False,
                key='_ddfile'
                )
        if ddfile is not None:
            with st.container(border=True):
                st.write('###### Loaded design data:')
                st.dataframe(dd.head(), hide_index=True)

    with st.container(border=True):
        ##Wire Run Data
        sample_wr_csv = convert_df(OCS.sample_wr_df())
        st.markdown("#### Load layout design data for wire run")
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
            with st.container(border=True):
                wr = OCS.wire_run(wrfile)
                st.write('###### First several rows of input file:')
                st.dataframe(wr.head(), hide_index=True)

calc1()
