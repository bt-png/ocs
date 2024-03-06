# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import pandas as pd
import numpy as np

import streamlit as st
#from streamlit.hello.utils import show_code
import system as OCS

def calc1() -> None:
    
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

    ## WIRE RUN
    file = st.file_uploader(
        'Upload a formatted "*.csv" file containing your WR data.',
        type={'csv', 'txt'},
        accept_multiple_files = False
        )

    if file is not None and not st.session_state['pauseCalc']:
        ## LAYOUT DESIGN
        wr = OCS.wire_run(file)
        
        Nom = OCS.CatenaryFlexible(cN, wr)
        Nom.resetloads(df_loading)
        df = Nom.dataframe()
        st.write('### Catenary Wire Sag Plot')
        chart = st.line_chart(
            data = df, x='Stationing', y='Elevation', color='cable'
            )
        st.write('#### Sag Data ', df)
        st.write('#### HA Data', Nom.dataframe_ha())

st.set_page_config(page_title="CAT SAG", page_icon="📹")
st.markdown("# Simple Catenary Sag")
st.sidebar.header("CAT SAG")

checkbox = st.sidebar.checkbox('Pause Calculation', key='pauseCalc', value=False)

st.write(
    """This app shows you the simple sag curves for a flexible catenary system
    utilizing a **sum of moments** method. The CW is assumed to be supported only
    at hanger locations, with elevations calculated based on designed elevation at 
    supports and the designated pre-sag."""
    )

calc1()
