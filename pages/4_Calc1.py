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

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code
import system as OCS

def calc1() -> None:
    @st.cache_data
    ## CONDUCTOR PARTICULARS AND LOADING CONDITIONS
    cN = OCS.conductor_particulars((1.544, 5000), (1.063, 3300), 0.2)
    
    ## LAYOUT DESIGN
    #filepath_none = '..\\input\\InputData_none.csv'
    file = st.file_uploader("upload WR data, type={'csv', 'txt'})
    wr = OCS.wire_run(filepath_none)
    
    
    ## POINT LOADS
    pN = (
    ('term span', 'uninsulated overlap', 'mpa z-anchor', 'mpa z-anchor', 'SI and feeders', 'insulated overlap', 'term span'),
    (162442+30, 162442-15, 164755-25, 164755+25, 166379-15, 166980-15, 167160-30),
    (15, 5, -15, -15, 30, 5, 15),
    (15, 5, 0, 0, 85, 5, 15)
    )
    
    Nominal = OCS.CatenaryFlexible(cN, wr)
    Nominal.resetloads(pN)
    return Nominal.dataframe()

st.set_page_config(page_title="Calculation Set 1", page_icon="📹")
st.markdown("# Calculation Set 1")
st.sidebar.header("Calculation Set 1")
st.write(
    """This app shows how you can use Streamlit to build cool animations.
It displays an animated fractal based on the the Julia Set. Use the slider
to tune different parameters."""
)

calc1()

show_code(calc1)
